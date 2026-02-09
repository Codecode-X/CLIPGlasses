import random
from collections import defaultdict
from ..build import DATASET_REGISTRY
from .DatasetBase import DatasetBase
from utils import read_json, write_json
import os
from utils import check_isfile

@DATASET_REGISTRY.register()
class DatasetClsBase(DatasetBase):
    """
    Base class for classification datasets.
    Inherits from the DatasetBase class.

    Publicly accessible attributes (@property):
        - Attributes from the parent class DatasetBase:
            - train (list): Labeled training data.
            - val (list): Validation data (optional).
            - test (list): Test data.

        - lab2cname (dict): Mapping from labels to class names.
        - classnames (list): List of class names.
        - num_classes (int): Number of classes.

    Internally accessible attributes:
        - Attributes from the parent class DatasetBase:
            - num_shots (int): Number of few-shot samples.
            - seed (int): Random seed.
            - p_trn (float): Proportion of training set.
            - p_val (float): Proportion of validation set.
            - p_tst (float): Proportion of test set.
            - dataset_dir (str): Dataset directory.

        - image_dir (str): Image directory.

    Basic methods:
        - Implements abstract methods/interfaces from the DatasetBase class:
            - read_split: Reads the data split file.
            - save_split: Saves the data split file.
            - generate_fewshot_dataset: Generates a few-shot dataset (usually for the training set).

    Abstract methods/interfaces (to be implemented by specific dataset subclasses):
        - read_and_split_data: Reads and splits data into train, val, and test datasets (customized based on the format of each classification dataset).
    """

    def __init__(self, cfg):
        """ 
        Initializes the basic attributes of the dataset.

        Parameters:
            - cfg (CfgNode): Configuration.

        Configuration:
            - Currently read:
                - image_dir (str): cfg.DATASET.IMAGE_DIR: Image directory. Example: /root/autodl-tmp/caltech-101/101_ObjectCategories

            - Read by the parent class:
                - dataset_dir (str): cfg.DATASET.DATASET_DIR: Dataset directory.
                - num_shots (int): cfg.DATASET.NUM_SHOTS: Number of few-shot samples | -1 means using all data, 0 means zero-shot, 1 means one-shot, and so on.
                - seed (int): cfg.SEED: Random seed.
                - p_trn (float): cfg.DATASET.SPLIT[0]: Proportion of training set.
                - p_val (float): cfg.DATASET.SPLIT[1]: Proportion of validation set.
                - p_tst (float): cfg.DATASET.SPLIT[2]: Proportion of test set.
        
        Main steps:
            1. Read additional configuration.
            2. Call the parent class DatasetBase constructor:
                1. Read configuration.
                2. Read and split data into train, val, and test datasets (subclass needs to implement get_data and get_fewshot_data methods).
                3. If num_shots >= 0, perform few-shot sampling from train and val to generate few-shot train and val datasets.
            3. Obtain additional attributes: number of classes, mapping from labels to class names, and list of class names.
        """

        # ---Read configuration---
        self.image_dir = cfg.DATASET.IMAGE_DIR  # Get image directory, e.g., /root/autodl-tmp/caltech-101/101_ObjectCategories
        
        # ---Call parent class constructor to obtain self.train, self.val, self.test, etc.---
        super().__init__(cfg)  # Call parent class constructor
        
        # ---Obtain attributes: number of classes, mapping from labels to class names, and list of class names---
        self._num_classes = _get_num_classes(self.train)
        self._lab2cname, self._classnames = _get_lab2cname(self.val)


    # -------------------Attributes-------------------
        
    @property
    def lab2cname(self):
        """Returns the mapping from labels to class names."""
        return self._lab2cname

    @property
    def classnames(self):
        """Returns the list of class names."""
        return self._classnames

    @property
    def num_classes(self):
        """Returns the number of classes."""
        return self._num_classes
    

    # -----------------------Abstract methods required by the parent class DatasetBase-----------------------

    def read_split(self, split_file):
        """
        Reads the data split file (implements the abstract method of the parent class).
        
        Parameters:
            - img_path_prefix (str): Image path prefix, usually the directory where images are located.
        
        Returns:
            - Training, validation, and test data (list of Datum objects).
        """
        img_path_prefix = self.image_dir  # Image path prefix
        def _convert(items):
            out = []
            for impath, label, classname in items:
                impath = os.path.join(img_path_prefix, impath)  # Join image path
                item = Datum(impath=impath, label=int(label), classname=classname)  # Create Datum object
                out.append(item)
            return out

        print(f"Reading split from {split_file}")  # Print information about reading the split file
        split = read_json(split_file)  # Read JSON file
        train = _convert(split["train"])  # Convert training data
        val = _convert(split["val"])  # Convert validation data
        test = _convert(split["test"])  # Convert test data

        return train, val, test  # Return training, validation, and test data (list of Datum objects)
    

    def save_split(self, train, val, test, split_file):
        """
        Saves the data split file (implements the abstract method of the parent class).
        
        Parameters:
            - train (list): Training dataset.
            - val (list): Validation dataset.
            - test (list): Test dataset.
            - split_file (str): Path to the data split file.

        Returns:
            - None
        """
        img_path_prefix = self.image_dir  # Image path prefix
        def _extract(items):
            out = []
            for item in items:
                impath = item.impath
                label = item.label
                classname = item.classname
                impath = impath.replace(img_path_prefix, "")  # Remove path prefix
                if impath.startswith("/"):
                    impath = impath[1:]
                out.append((impath, label, classname))
            return out
        train = _extract(train)  # Extract training data
        val = _extract(val)  # Extract validation data
        test = _extract(test)  # Extract test data

        split = {"train": train, "val": val, "test": test}  # Create split dictionary

        write_json(split, split_file)  # Write to JSON file
        print(f"Saved split to {split_file}")  # Print information about saving the split file
    

    def generate_fewshot_dataset(self, dataset, num_shots=-1, repeat=False):
        """Generates a few-shot dataset, containing only a few images per class (implements the abstract method of the parent class).

        Parameters:
            - dataset (list): List of Datum objects.
            - num_shots (int): Number of instances sampled per class. | Default -1 means returning the original dataset.
            - repeat (bool): Whether to repeat images if needed (default: False).
        
        Returns:
            - fewshot_dataset (list): Dataset containing a few images.
        """
    
        # Non-few-shot learning case, implemented by the parent class method
        if num_shots <= 0:
            super().generate_fewshot_dataset(dataset, num_shots, repeat)

        print(f"Creating a {num_shots}-shot dataset...")

        # Organize the dataset by label
        tracker = defaultdict(list)
        for item in dataset:
            tracker[item.label].append(item)

        # Randomly sample num_shots samples for each class
        fewshot_dataset = []  # Few-shot dataset
        for label, items in tracker.items():
            # If there are enough samples, randomly sample num_shots samples
            if len(items) >= num_shots:
                sampled_items = random.sample(items, num_shots)  # Randomly sample num_shots samples for each class
            else:
                # If samples are insufficient, decide whether to repeat sampling based on the repeat parameter
                if repeat:
                    sampled_items = random.choices(items, k=num_shots)
                else:
                    sampled_items = items  # If not repeating, directly use all samples
            # Add the sampled samples to the dataset
            fewshot_dataset.extend(sampled_items)  # Contains num_shots samples for all classes in the current dataset

        return fewshot_dataset

    
    # -------------------Abstract methods/interfaces to be implemented by specific dataset subclasses-------------------

    def read_and_split_data(self):
        """
        Reads and splits data into train, val, and test datasets (customized based on the format of each classification dataset).

        Needs to return:
            - Training, validation, and test data (list of Datum objects).
        """
        raise NotImplementedError





# -------------------Helper classes and functions-------------------

def _get_num_classes(dataset):
    """Counts the number of classes. Implemented in init.

    Parameters:
        - data_source (list): List of Datum objects.
    
    Returns:
        - num_classes (int): Number of classes.
    """
    # Use a set to store unique labels
    label_set = set()
    for item in dataset:
        # Add the label of each data instance to the set
        label_set.add(item.label)
    # Return the maximum label value plus 1 as the number of classes
    return max(label_set) + 1

def _get_lab2cname(dataset):
    """
    Gets the mapping from labels to class names (dictionary).

    Parameters:
        - data_source (list): List of Datum objects.
    
    Returns:
        - mapping (dict): Mapping from labels to class names. | {label: classname}
        - classnames (list): List of class names. | [classname1, classname2, ...]
    """
    # Get the mapping of all class labels and class names in the dataset
    container = {(item.label, item.classname) for item in dataset}
    mapping = {label: classname for label, classname in container}
    # Get all labels and sort them
    labels = list(mapping.keys())
    labels.sort()
    # Generate the list of class names based on the sorted labels
    classnames = [mapping[label] for label in labels]
    return mapping, classnames

class Datum:
    """Data instance class, defines basic attributes.

    Parameters:
        impath (str): Image path.
        label (int): Class label.
        classname (str): Class name.
    """

    def __init__(self, impath="", label=0, classname=""):
        """Initializes a data instance."""
        # Ensure the image path is of string type
        assert isinstance(impath, str)
        # Check if the image path is a valid file
        assert check_isfile(impath)

        # Initialize the image path
        self._impath = impath  # str
        # Initialize the class label
        self._label = label  # int
        # Initialize the class name
        self._classname = classname  # str

    @property
    def impath(self):
        """Returns the image path."""
        return self._impath
    @property
    def label(self):
        """Returns the class label."""
        return self._label
    @property
    def classname(self):
        """Returns the class name."""
        return self._classname
