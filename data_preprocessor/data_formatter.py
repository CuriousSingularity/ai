import numpy as np
import h5py
from sklearn.model_selection import train_test_split

class CreateDatasetPackage():
    def __init__(self, filename, dataset_name = "FullData", keys=["X", "Y"]):
        """
        Creates a new dataset h5 file with train, test and validation datasets with
        the requested shape
        Parameters
        ----------
        filename : name of the file
        dataset_name : key name in the h5 file
        keys : keys inside dataset_name
        """
        self._filename = filename
        self._dataset_name = dataset_name
        self._array = {}
        self._keys = ["X", "Y"]
        self.train = {}
        self.test = {}
        self.valid = {}

        with h5py.File(filename, "r") as h5_f:
            for index, name in zip(self._keys, keys):
                key = f"{self._dataset_name}/{name}"

                if key not in h5_f:
                    raise LookupError(f"Data {key} is not available")

                self._array[index] = np.array(h5_f[key])

    def modify_shape(self, new_sample_shape):
        """
        Modify the shape of the input samples X
        Parameters
        ----------
        new_sample_shape

        Returns
        -------

        """
        for key, shape in zip(self._array.keys(), new_sample_shape):
            if shape is not None:
                sample_size = (self._array[key].shape[0],)
                self._array[key] = self._array[key].reshape((sample_size + shape))

    def split(self, validation_split = 0.1, test_split = 0.1):
        """
        Split the dataset into train, test and valid sets given the percentage
        Parameters
        ----------
        validation_split : percentage of the dataset used for validation
        test_split : percentage of the dataset used for test

        Returns
        -------

        """
        X_train, X_test, Y_train, Y_test = train_test_split(self._array["X"], self._array["Y"], test_size=test_split)
        X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=validation_split)

        self.train = {"X": X_train, "Y":Y_train}
        self.valid = {"X": X_valid, "Y":Y_valid}
        self.test = {"X": X_test, "Y": Y_test}

    def save(self, filename="data.h5"):
        with h5py.File(filename, "w") as h5_f:
            main_group = h5_f.create_group("data")
            sub_g1 = main_group.create_group('train')
            sub_g2 = main_group.create_group('valid')
            sub_g3 = main_group.create_group('test')

            sub_g1.create_dataset("samples", data=self.train["X"])
            sub_g1.create_dataset("labels", data=self.train["Y"])

            sub_g2.create_dataset("samples", data=self.test["X"])
            sub_g2.create_dataset("labels", data=self.test["Y"])

            sub_g3.create_dataset("samples", data=self.valid["X"])
            sub_g3.create_dataset("labels", data=self.valid["Y"])

    def get_class_probability(self, verbose=True):

        def get_label_probability(labels):
            each_class = np.sum(labels, axis=0)
            total = np.sum(each_class)

            class_probability = each_class / total
            inverse_class_proability = class_probability ** -1

            return class_probability, inverse_class_proability

        class_probability = {}
        inverse_class_proability = {}
        class_probability["train"], inverse_class_proability["train"] = get_label_probability(self.train["Y"])
        class_probability["valid"], inverse_class_proability["valid"] = get_label_probability(self.valid["Y"])
        class_probability["test"], inverse_class_proability["test"] = get_label_probability(self.test["Y"])

        return class_probability, inverse_class_proability


if __name__ == "__main__":
    file = "example.h5"

    dp = CreateDatasetPackage(filename=file, dataset_name = "FullData", keys=["X_full", "y_full"])

    dp.modify_shape([(64, 64, 1)])
    dp.split()
    dp.save()

    class_probability, inverse_class_proability = dp.get_class_probability()

    for (key_cp, val_cp), (key_icp, val_icp) in zip(class_probability.items(), inverse_class_proability.items()):
        print(f"Class Probability {key_cp} : {val_cp}")
        print(f"Inverse Class Probability {key_icp} : {val_icp}")
