import tensorflow as tf


def make_example(row, sparse_feature_names, dense_feature_names, label_names):
    features = {
        feat: tf.train.Feature(int64_list=tf.train.Int64List(value=[int(row[feat])]))
        for feat in sparse_feature_names
    }
    features.update(
        {
            feat: tf.train.Feature(float_list=tf.train.FloatList(value=[row[feat]]))
            for feat in dense_feature_names
        }
    )
    features.update(
        {
            feat: tf.train.Feature(float_list=tf.train.FloatList(value=[row[feat]]))
            for feat in label_names
        }
    )
    return tf.train.Example(features=tf.train.Features(feature=features))


def write_tfrecord(
    filename, df, sparse_feature_names, dense_feature_names, label_names
):
    writer = tf.io.TFRecordWriter(filename)
    for _, row in df.iterrows():
        ex = make_example(row, sparse_feature_names, dense_feature_names, label_names)
        writer.write(ex.SerializeToString())
    writer.close()


def input_fn_tfrecord(
    train_or_test,
    filenames,
    feature_description,
    label_description=None,
    batch_size=256,
    num_epochs=1,
    num_parallel_calls=8,
    shuffle_factor=10,
):
    def _parse_examples(serial_exmp):
        try:
            features = tf.parse_single_example(
                serial_exmp, features=feature_description
            )
        except AttributeError:
            features = tf.io.parse_single_example(
                serial_exmp, features=feature_description
            )
        if train_or_test in ("train", "validation"):
            for label in label_description:
                features[label + "_loss"] = features.pop(label)
            return features
        else:
            labels = {}
            for label in label_description:
                labels[label] = features.pop(label)
            return features, labels

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_examples, num_parallel_calls=num_parallel_calls)
    if shuffle_factor > 0:
        dataset = dataset.shuffle(buffer_size=batch_size * shuffle_factor)

    dataset = dataset.repeat(num_epochs).batch(batch_size)

    return dataset
