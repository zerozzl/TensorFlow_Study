import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age",
           "dis", "tax", "ptratio", "medv"]
FEATURES = ["crim", "zn", "indus", "nox", "rm", "age",
            "dis", "tax", "ptratio"]
LABEL = "medv"


def input_fn(data_set):
    feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
    labels = tf.constant(data_set[LABEL].values)
    return feature_cols, labels


if __name__ == "__main__":
    # Load datasets
    training_set = pd.read_csv("Boston_Housing_data/boston_train.csv",
                               skipinitialspace=True,
                               skiprows=1,
                               names=COLUMNS)
    test_set = pd.read_csv("Boston_Housing_data/boston_test.csv",
                           skipinitialspace=True,
                           skiprows=1,
                           names=COLUMNS)

    # Set of 6 examples for which to predict median house values
    prediction_set = pd.read_csv("Boston_Housing_data/boston_predict.csv",
                                 skipinitialspace=True,
                                 skiprows=1,
                                 names=COLUMNS)

    # Feature cols
    feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]

    # Build 2 layer fully connected DNN with 10, 10 units respectively
    regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                              hidden_units=[10, 10],
                                              model_dir="Boston_Housing_model")

    # Fit
    regressor.fit(input_fn=lambda: input_fn(training_set), steps=5000)

    # Score accuracy
    ev = regressor.evaluate(input_fn=lambda: input_fn(test_set), steps=1)
    loss_score = ev["loss"]
    print "Loss: {0:f}".format(loss_score)

    # Print out predictions
    y = regressor.predict(input_fn=lambda: input_fn(prediction_set))
    print "Predictions: {}".format(str(y))
