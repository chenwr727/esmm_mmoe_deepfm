"""
Reference:
    Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics
    https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html
"""

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import Constant
from tensorflow.python.keras.layers import Layer


class MultiLossLayer(Layer):
    def __init__(self, num_tasks, tasks, **kwargs):
        self.num_tasks = num_tasks
        self.tasks = tasks
        super(MultiLossLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # initialise log_vars
        self.log_vars = []
        for i in range(self.num_tasks):
            self.log_vars += [
                self.add_weight(
                    name="log_var" + str(i),
                    shape=(1,),
                    initializer=Constant(0.0),
                    trainable=True,
                )
            ]
        super(MultiLossLayer, self).build(input_shape)

    def multi_loss(self, ys_true, ys_pred, tasks):
        assert len(ys_true) == self.num_tasks and len(ys_pred) == self.num_tasks
        total_loss = 0
        for y_true, y_pred, task, log_var in zip(
            ys_true, ys_pred, tasks, self.log_vars
        ):
            precision = K.exp(-log_var)
            if task == "binary":
                loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
            else:
                loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
            total_loss += K.sum(precision * loss + log_var, -1)
        return K.mean(total_loss)

    def call(self, inputs):
        ys_true = inputs[: self.num_tasks]
        ys_pred = inputs[self.num_tasks :]
        loss = self.multi_loss(ys_true, ys_pred, self.tasks)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return K.concatenate(inputs, -1)

    def get_config(self):
        config = {"tasks": self.tasks, "num_tasks": self.num_tasks}
        base_config = super(MultiLossLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
