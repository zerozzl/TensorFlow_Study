import pandas as pd
import tensorflow as tf

TRAIN_DATA = "Census_data/adult.data"
TEST_DATA = "Census_data/adult.test"
MODEL_DIR = "Census_model/"
COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]
LABEL_COLUMN = "label"
CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
                       "relationship", "race", "gender", "native_country"]
CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss",
                      "hours_per_week"]


def build_estimator(model_type, model_dir):
    # Sparse base columns.
    gender = tf.contrib.layers.sparse_column_with_keys(column_name="gender",
                                                       keys=["female", "male"])
    education = tf.contrib.layers.sparse_column_with_hash_bucket(
        "education", hash_bucket_size=1000)
    relationship = tf.contrib.layers.sparse_column_with_hash_bucket(
        "relationship", hash_bucket_size=100)
    workclass = tf.contrib.layers.sparse_column_with_hash_bucket(
        "workclass", hash_bucket_size=100)
    occupation = tf.contrib.layers.sparse_column_with_hash_bucket(
        "occupation", hash_bucket_size=1000)
    native_country = tf.contrib.layers.sparse_column_with_hash_bucket(
        "native_country", hash_bucket_size=1000)

    # Continuous base columns.
    age = tf.contrib.layers.real_valued_column("age")
    education_num = tf.contrib.layers.real_valued_column("education_num")
    capital_gain = tf.contrib.layers.real_valued_column("capital_gain")
    capital_loss = tf.contrib.layers.real_valued_column("capital_loss")
    hours_per_week = tf.contrib.layers.real_valued_column("hours_per_week")

    # Transformations.
    age_buckets = tf.contrib.layers.bucketized_column(age,
                                                      boundaries=[
                                                          18, 25, 30, 35, 40, 45,
                                                          50, 55, 60, 65
                                                      ])

    # Wide columns and deep columns.
    wide_columns = [gender, native_country, education, occupation, workclass,
                    relationship, age_buckets,
                    tf.contrib.layers.crossed_column([education, occupation],
                                                     hash_bucket_size=int(1e4)),
                    tf.contrib.layers.crossed_column(
                        [age_buckets, education, occupation],
                        hash_bucket_size=int(1e6)),
                    tf.contrib.layers.crossed_column([native_country, occupation],
                                                     hash_bucket_size=int(1e4))]
    deep_columns = [
        tf.contrib.layers.embedding_column(workclass, dimension=8),
        tf.contrib.layers.embedding_column(education, dimension=8),
        tf.contrib.layers.embedding_column(gender, dimension=8),
        tf.contrib.layers.embedding_column(relationship, dimension=8),
        tf.contrib.layers.embedding_column(native_country, dimension=8),
        tf.contrib.layers.embedding_column(occupation, dimension=8),
        age,
        education_num,
        capital_gain,
        capital_loss,
        hours_per_week]

    if model_type == "wide":
        m = tf.contrib.learn.LinearClassifier(model_dir=model_dir,
                                              feature_columns=wide_columns)
    elif model_type == "deep":
        m = tf.contrib.learn.DNNClassifier(model_dir=model_dir,
                                           feature_columns=deep_columns,
                                           hidden_units=[100, 50])
    else:
        m = tf.contrib.learn.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=[100, 50])
    return m


def input_fn(df):
    # Creates a dictionary mapping from each continuous feature column name (k) to
    #  the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    categorical_cols = {k: tf.SparseTensor(
        indices=[[i, 0] for i in range(df[k].size)],
        values=df[k].values,
        shape=[df[k].size, 1])
            for k in CATEGORICAL_COLUMNS}
    # Merges the two dictionaries into one.
    feature_cols = dict(continuous_cols)
    feature_cols.update(categorical_cols)
    # Converts the label column into a constant Tensor.
    label = tf.constant(df[LABEL_COLUMN].values)
    # Returns the feature columns and the label.
    return feature_cols, label


def train_and_eval(train_data, test_data, model_dir, train_steps=200):
    train_file_name = train_data
    test_file_name = test_data
    df_train = pd.read_csv(
        tf.gfile.Open(train_file_name),
        names=COLUMNS,
        skipinitialspace=True,
        engine="python")
    df_test = pd.read_csv(
        tf.gfile.Open(test_file_name),
        names=COLUMNS,
        skipinitialspace=True,
        skiprows=1,
        engine="python")

    df_train[LABEL_COLUMN] = (
        df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
    df_test[LABEL_COLUMN] = (
        df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)

    print "model directory = %s" % model_dir
    m = build_estimator("wide_n_deep", model_dir)
    m.fit(input_fn=lambda: input_fn(df_train), steps=train_steps)
    results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
    for key in sorted(results):
        print "%s: %s" % (key, results[key])


if __name__ == '__main__':
    train_and_eval(TRAIN_DATA, TEST_DATA, MODEL_DIR)
