import os
import random
from .build import DATASET_REGISTRY
from .base_class.DatasetClsBase import DatasetClsBase, Datum
from utils import listdir_nohidden


IGNORED = ["BACKGROUND_Google", "Faces_easy"] # List of categories to ignore

NEW_CNAMES = { # Mapping of new category names
    "airplanes": "airplane",
    "Faces": "face",
    "Leopards": "leopard",
    "Motorbikes": "motorbike",
}


@DATASET_REGISTRY.register()
class Caltech101(DatasetClsBase):
    """
    Caltech101 dataset class.
    Inherits from DatasetClsBase class.

    Public attributes (@property):
        - Attributes from the parent class DatasetClsBase:
            - Attributes from the parent class DatasetBase:
                - train (list): Labeled training data.
                - val (list): Validation data (optional).
                - test (list): Test data.

            - lab2cname (dict): Mapping from label to category name.
            - classnames (list): List of category names.
            - num_classes (int): Number of categories.
            - impath (str): Image path.
            - label (int): Category label.
            - classname (str): Category name.

    Internal attributes:
        - Attributes from the parent class DatasetClsBase:
            - Attributes from the parent class DatasetBase:
                - num_shots (int): Number of few-shot samples.
                - seed (int): Random seed.
                - p_trn (float): Proportion of training set.
                - p_val (float): Proportion of validation set.
                - p_tst (float): Proportion of test set.
                - dataset_dir (str): Dataset directory.

            - image_dir (str): Image directory.

    Implements abstract methods/interfaces from DatasetClsBase:
        - read_and_split_data: Reads data and splits it into train, val, and test datasets (customized for each dataset format).

    """

    def read_and_split_data(self):
        """
        Reads data and splits it into train, val, and test datasets (implements the abstract method from the parent class).
        
        Attributes used:
            - p_trn (float): Proportion of training set.
            - p_val (float): Proportion of validation set.
            - p_tst (float): Proportion of test set.
            - image_dir (str): Image directory.
            - IGNORED (List[str]): List of categories to ignore.
            - NEW_CNAMES (Dict[str, str]): Mapping of new category names.

        Returns:
            - Training, validation, and test data (list of Datum objects).

        Main steps:
            1. Calculate the proportion of the test set.
            2. Get the list of categories.
            3. Iterate through the list of categories:
                - Get the list of images in the category directory.
                - Update the category name if a new name mapping exists.
                - Shuffle the list of images randomly and calculate the number of training, validation, and test samples.
                - Package the data into (list of Datum objects) for training, validation, and test datasets.
            4. Return the training, validation, and test data (list of Datum objects).
        """
        print(f"Splitting into {self.p_trn:.0%} train, {self.p_val:.0%} val, and {self.p_tst:.0%} test")  # Print split proportions
        new_cnames = NEW_CNAMES
        ignored_categories = IGNORED

        # ---Get the list of categories---
        categories = listdir_nohidden(self.image_dir)  # Get the list of categories
        categories = [c for c in categories if c not in ignored_categories]  # Filter out ignored categories
        categories.sort()

        train, val, test = [], [], []

        # ---Iterate through the list of categories---
        for label, category in enumerate(categories):
            
            # ---Get the list of images in the category directory---
            category_dir = os.path.join(self.image_dir, category)  # Get the category directory
            images = listdir_nohidden(category_dir)  # Get the list of images
            images = [os.path.join(category_dir, im) for im in images]  # Join image paths

            # ---Update the category name if a new name mapping exists---
            if new_cnames is not None and category in new_cnames:
                category = new_cnames[category]  # Update the category name
            
            # ---Shuffle the list of images randomly and calculate the number of training, validation, and test samples---
            random.shuffle(images)  # Shuffle the list of images randomly
            n_total = len(images)
            n_train = round(n_total * self.p_trn)  # Calculate the number of training samples
            n_val = round(n_total * self.p_val)  # Calculate the number of validation samples
            n_test = n_total - n_train - n_val  # Calculate the number of test samples
            assert n_train > 0 and n_val > 0 and n_test > 0  # Assert that the number of training, validation, and test samples is greater than 0

            # ---Package the data into (list of Datum objects) for training, validation, and test datasets---
            def _collate(imgs, label, classname):
                """ Package the imgs data list into a list of Datum objects """
                return [Datum(impath=imgpath, label=label, classname=classname) for imgpath in imgs]
            train.extend(_collate(images[:n_train], label, category))  # Collect training data
            val.extend(_collate(images[n_train : n_train + n_val], label, category))  # Collect validation data
            test.extend(_collate(images[n_train + n_val :], label, category))  # Collect test data
            
        return train, val, test  # Return training, validation, and test data (list of Datum objects)
