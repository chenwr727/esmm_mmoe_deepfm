import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from utils.preprocess import write_tfrecord


if __name__ == "__main__":
    column_names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]
    data = pd.read_csv("./data/adult.data", header=None, names=column_names)

    data["label_income"] = data["income"].map({" <=50K": 0, " >50K": 1})
    data["label_marital"] = (
        data["marital-status"].apply(lambda x: 1 if x == " Never-married" else 0)
        * data["label_income"]
    )
    data["label_education"] = (
        data["education-num"].apply(lambda x: 1 if x >= 10 else 0)
        * data["label_income"]
    )
    data.drop(
        labels=["income", "marital-status", "education", "education-num"],
        axis=1,
        inplace=True,
    )

    columns = data.columns.values.tolist()
    sparse_features = [
        "workclass",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    labels = ["label_income", "label_marital", "label_education"]
    dense_features = [
        col for col in columns if col not in sparse_features and col not in labels
    ]

    data[sparse_features] = data[sparse_features].fillna(
        "-1",
    )
    data[dense_features] = data[dense_features].fillna(
        0,
    )
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    train, test = train_test_split(data, test_size=0.2, random_state=2020)
    train, validation = train_test_split(train, test_size=0.25, random_state=2020)

    write_tfrecord(
        "./data/adult.tr.tfrecords", train, sparse_features, dense_features, labels
    )
    write_tfrecord(
        "./data/adult.te.tfrecords", test, sparse_features, dense_features, labels
    )
    write_tfrecord(
        "./data/adult.va.tfrecords", test, sparse_features, dense_features, labels
    )
