# -*- coding:utf-8 -*-
"""
Author:
    Wanrui chen

Reference:
    多目标排序模型在腾讯QQ看点推荐中的应用实践(https://mp.weixin.qq.com/s/RwMYLZRsX2TGSQsU8PPhig)

"""


from itertools import chain

import tensorflow as tf
from deepctr.feature_column import (
    DEFAULT_GROUP_NAME,
    build_input_features,
    get_linear_logit,
    input_from_feature_columns,
)
from deepctr.layers.core import DNN, PredictionLayer
from deepctr.layers.interaction import FM
from deepctr.layers.utils import add_func, combined_dnn_input, concat_func, reduce_sum
from tensorflow.python.keras.layers import Input

from .mtl_loss import MultiLossLayer


class CustomModel(tf.keras.Model):
    def train_step(self, data):
        x = data
        y = None

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred[0], regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        y = y_pred.copy()
        y[1] = [x[key] for key in x.keys() if key.endswith("_loss")]
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics if not m.name.startswith("multi")}

    def test_step(self, data):
        x = data
        y = None

        y_pred = self(x, training=True)  # Forward pass
        # Compute the loss value
        # (the loss function is configured in `compile()`)
        self.compiled_loss(y, y_pred[0], regularization_losses=self.losses)
        # Update metrics (includes the metric that tracks the loss)
        y = y_pred.copy()
        y[1] = [x[key] for key in x.keys() if key.endswith("_loss")]
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics if not m.name.startswith("multi")}


def DeepFM_MMOE_ESMM(
    linear_feature_columns,
    dnn_feature_columns,
    fm_group=(DEFAULT_GROUP_NAME,),
    l2_reg_linear=0.00001,
    l2_reg_embedding=0.00001,
    l2_reg_dnn=0,
    seed=1024,
    dnn_dropout=0,
    dnn_activation="relu",
    dnn_use_bn=False,
    num_experts=3,
    expert_dnn_hidden_units=(256, 128),
    tower_dnn_hidden_units=(64,),
    gate_dnn_hidden_units=(),
    task_types=("binary", "binary", "binary"),
    task_names=("ctr", "ctcvr", "ctvoi"),
):
    """Instantiates the DeepFM_MMOE_ESMM Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param fm_group: list, group_name of features that will be used to do feature interactions.
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
    :param num_experts: integer, number of experts.
    :param expert_dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of expert DNN.
    :param tower_dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of task-specific DNN.
    :param gate_dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of gate DNN.
    :param task_types: list of str, indicating the loss of each tasks, ``"binary"`` for  binary logloss, ``"regression"`` for regression loss. e.g. ['binary', 'regression']
    :param task_names: list of str, indicating the predict target of each tasks
    :return: A Keras model instance.
    """

    features = build_input_features(linear_feature_columns + dnn_feature_columns)

    inputs_list = list(features.values())

    linear_logit = get_linear_logit(
        features,
        linear_feature_columns,
        seed=seed,
        prefix="linear",
        l2_reg=l2_reg_linear,
    )

    group_embedding_dict, dense_value_list = input_from_feature_columns(
        features, dnn_feature_columns, l2_reg_embedding, seed, support_group=True
    )

    fm_logit = add_func(
        [
            FM()(concat_func(v, axis=1))
            for k, v in group_embedding_dict.items()
            if k in fm_group
        ]
    )

    dnn_input = combined_dnn_input(
        list(chain.from_iterable(group_embedding_dict.values())), dense_value_list
    )

    # mmoe
    num_tasks = len(task_names)
    expert_outs = []
    for i in range(num_experts):
        expert_network = DNN(
            expert_dnn_hidden_units,
            dnn_activation,
            l2_reg_dnn,
            dnn_dropout,
            dnn_use_bn,
            seed=seed,
            name="expert_" + str(i),
        )(dnn_input)
        expert_outs.append(expert_network)

    expert_concat = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=1))(
        expert_outs
    )  # None,num_experts,dim

    mmoe_outs = []
    for i in range(num_tasks):  # one mmoe layer: nums_tasks = num_gates
        # build gate layers
        gate_input = DNN(
            gate_dnn_hidden_units,
            dnn_activation,
            l2_reg_dnn,
            dnn_dropout,
            dnn_use_bn,
            seed=seed,
            name="gate_" + task_names[i],
        )(dnn_input)
        gate_out = tf.keras.layers.Dense(
            num_experts,
            use_bias=False,
            activation="softmax",
            name="gate_softmax_" + task_names[i],
        )(gate_input)
        gate_out = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(
            gate_out
        )

        # gate multiply the expert
        gate_mul_expert = tf.keras.layers.Lambda(
            lambda x: reduce_sum(x[0] * x[1], axis=1, keep_dims=False),
            name="gate_mul_expert_" + task_names[i],
        )([expert_concat, gate_out])
        mmoe_outs.append(gate_mul_expert)

    task_outs = []
    for i, (task_type, task_name, mmoe_out) in enumerate(
        zip(task_types, task_names, mmoe_outs)
    ):
        # build tower layer
        tower_output = DNN(
            tower_dnn_hidden_units,
            dnn_activation,
            l2_reg_dnn,
            dnn_dropout,
            dnn_use_bn,
            seed=seed,
            name="tower_" + task_name,
        )(mmoe_out)

        dnn_logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(
            tower_output
        )
        final_logit = add_func([linear_logit, fm_logit, dnn_logit])
        output = PredictionLayer(
            task_type, name="tower_" + task_name + "_out" if i > 0 else task_name
        )(final_logit)
        task_outs.append(output)

    # esmm
    ctr_pred = task_outs[0]
    cvr_pred = task_outs[1]
    voi_pred = task_outs[2]

    # CTCVR = CTR * CVR
    ctcvr_pred = tf.keras.layers.Multiply(name=task_names[1])([ctr_pred, cvr_pred])
    ctvoi_pred = tf.keras.layers.Multiply(name=task_names[2])([ctr_pred, voi_pred])
    task_outputs = [ctr_pred, ctcvr_pred, ctvoi_pred]

    prediction_model = tf.keras.models.Model(inputs=inputs_list, outputs=task_outputs)
    ys_true = [Input(shape=(1,), name=task_name + "_loss") for task_name in task_names]
    loss_layer_inputs = ys_true + task_outputs
    model_out = MultiLossLayer(num_tasks, task_types)(loss_layer_inputs)
    model_inputs = inputs_list + ys_true
    train_model = CustomModel(model_inputs, [model_out, task_outputs])
    return prediction_model, train_model
