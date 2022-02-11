import tensorflow as tf
from deepctr.feature_column import DenseFeat, SparseFeat
from tensorflow.python.ops.parsing_ops import FixedLenFeature

from model.esmm_mmoe_deepfm import DeepFM_MMOE_ESMM
from utils.preprocess import input_fn_tfrecord


if __name__ == "__main__":

    # 1.generate feature_column for linear part and dnn part

    sparse_features = [
        "workclass",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    dense_features = ["age", "fnlwgt", "capital-gain", "capital-loss", "hours-per-week"]
    labels = ["label_income", "label_marital", "label_education"]

    dnn_feature_columns = []
    linear_feature_columns = []

    for feat in sparse_features:
        dnn_feature_columns.append(SparseFeat(feat, 50, embedding_dim=4))
        linear_feature_columns.append(SparseFeat(feat, 50, embedding_dim=4))
    for feat in dense_features:
        dnn_feature_columns.append(DenseFeat(feat, 1))
        linear_feature_columns.append(DenseFeat(feat, 1))

    # 2.generate input data for model

    feature_description = {
        k: FixedLenFeature(dtype=tf.int64, shape=1) for k in sparse_features
    }
    feature_description.update(
        {k: FixedLenFeature(dtype=tf.float32, shape=1) for k in dense_features}
    )
    feature_description.update(
        {k: FixedLenFeature(dtype=tf.float32, shape=1) for k in labels}
    )

    train_model_input = input_fn_tfrecord(
        "train", "./data/adult.tr.tfrecords", feature_description, labels
    )
    test_model_input = input_fn_tfrecord(
        "test", "./data/adult.te.tfrecords", feature_description, labels, batch_size=2048
    )
    validation_model_input = input_fn_tfrecord(
        "validation", "./data/adult.va.tfrecords", feature_description, labels, batch_size=2048
    )

    # 3.Define Model,train,predict and evaluate

    prediction_model, train_model = DeepFM_MMOE_ESMM(
        linear_feature_columns, dnn_feature_columns, task_names=labels
    )
    metrics = [
        tf.keras.metrics.BinaryCrossentropy(name="loss"),
        tf.keras.metrics.AUC(name="auc"),
    ]
    optimizer = tf.keras.optimizers.Adam(
        tf.keras.optimizers.schedules.InverseTimeDecay(
            0.001, decay_steps=3000, decay_rate=1, staircase=False
        )
    )
    prediction_model.compile(optimizer, loss=None, metrics=metrics)
    train_model.compile("adam", loss=None, metrics=metrics)

    # callback
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
    ]
    train_model.fit(
        train_model_input,
        epochs=10,
        verbose=1,
        callbacks=callbacks,
        validation_data=validation_model_input,
    )

    # test
    eval_result = prediction_model.evaluate(test_model_input)
    scores = {name: loss for name, loss in zip(prediction_model.metrics_names, eval_result)}
    print(scores)
