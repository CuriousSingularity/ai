import h5py
import numpy as np
import autokeras as ak
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from sklearn.metrics import classification_report
import os

def load_data(filename, subset, verbose=True):
    arrays = {}
    with h5py.File(filename, "r") as h5file:
        for name in ["samples", "labels"]:
            key = f"data/{subset}/{name}"
            if key not in h5file:
                raise DataFormatError(f"No dataset {key} in file {filename}")
            arrays[name] = np.array(h5file[key])

    if verbose:
        print(f"Loading {subset} data \t: {arrays['samples'].shape} {arrays['labels'].shape}")

    return (arrays["samples"], arrays["labels"])

def ImageClassification():
    filename = "mnist.h5"
    X_train, Y_train = load_data(filename, "train")
    X_test, Y_test = load_data(filename, "test")
    X_valid, Y_valid = load_data(filename, "valid")

    model = ak.ImageClassifier(max_trials=3, overwrite=False)
    model.fit(X_train, Y_train, epochs=1, batch_size=64, validation_data=(X_valid, Y_valid))
    loss = model.evaluate(X_test, Y_test)
    print("Loss : {loss}")


def CreateSupergraph(output_dir, hp_tuner):
    input_node = ak.Input()
    conv2d_1 = ak.ConvBlock(num_blocks=1, num_layers=3,
                            max_pooling=True, dropout=0)(input_node)
    dense_1 = ak.DenseBlock(dropout=0)(conv2d_1)
    output_node = ak.ClassificationHead(num_classes=4,
                                        metrics=['accuracy'])(dense_1)

    automodel = ak.AutoModel(inputs=input_node, outputs=output_node,
                             max_trials=3, directory=output_dir, project_name="autoML",
                             tuner=hp_tuner, seed=123)

    return automodel

def Main():

    output_dir = "./output/"
    filename = "mnist.h5"
    X_train, Y_train = load_data(filename, "train")
    X_test, Y_test = load_data(filename, "test")
    X_valid, Y_valid = load_data(filename, "valid")

    hp_tuner = "greedy"
    # create supergraph
    automodel = CreateSupergraph(output_dir, hp_tuner)

    # run automl
    #TODO: early stopping with callbacks
    automodel.fit(x=X_train, y=Y_train, validation_data=(X_valid, Y_valid),
                  batch_size=64, epochs=5)

    score = automodel.evaluate(X_test, Y_test)

    # export best model
    model = automodel.export_model()
    model_json = model.to_json()

    # save model
    with open("autoML.json", "w") as json_file:
        json_file.write(model_json)
    # save weights
    model.save("best_model.h5")

    new_model = tf.keras.models.load_model("best_model.h5")
    new_model.summary()

    print("Saving Model to disk")

    predicted = automodel.predict(X_test)
    """
    report = classification_report(Y_test, predicted, target_names=["1", "2", "3", "4"])

    path = os.path.join(os.path.dirname(output_dir), f"results_{hp_tuner}.txt")
    with open(path, "w") as f:
        f.write(report)
    """

if __name__ == '__main__':
    Main()
