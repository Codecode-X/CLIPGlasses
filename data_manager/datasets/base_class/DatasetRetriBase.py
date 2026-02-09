import random
from collections import defaultdict
from ..build import DATASET_REGISTRY
from .DatasetBase import DatasetBase
from utils import read_json, write_json
from utils import check_isfile

@DATASET_REGISTRY.register()
class DatasetRetriBase(DatasetBase):
    """
    Base class for retrieval datasets.
    Inherits from the DatasetBase class.

    Publicly accessible attributes (@property):
        - Attributes from the parent class DatasetBase:
            - train (list): Labeled training data.
            - val (list): Validation data (optional).
            - test (list): Test data.
        - Currently read: None

    Internally accessible attributes:
        - Attributes from the parent class DatasetBase:
            - num_shots (int): Number of few-shot samples.
            - seed (int): Random seed.
            - p_trn (float): Training set proportion.
            - p_val (float): Validation set proportion.
            - p_tst (float): Test set proportion.
            - dataset_dir (str): Dataset directory.
        - Currently read:
            - csv_file (str): cfg.DATASET.CSV_FILE: Path to the annotation file | e.g., /root/autodl-tmp/NegBench/images/Retrieval/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv

    Basic methods:
        - Implements abstract methods/interfaces from the DatasetBase class:
            - read_split: Reads the data split file.
            - save_split: Saves the data split file.
            - generate_fewshot_dataset: Generates a few-shot dataset (usually for the training set).

    Abstract methods/interfaces (to be implemented by specific dataset subclasses):
        - read_and_split_data: Reads data and splits it into train, val, and test datasets (customized for each retrieval dataset format).
    """

    def __init__(self, cfg):
        """ 
        Initializes the basic attributes of the dataset.

        Parameters:
            - cfg (CfgNode): Configuration.

        Configuration:
            - Currently read:
                - csv_path (str): cfg.DATASET.CSV_PATH: Path to the annotation file | e.g., /root/autodl-tmp/NegBench/images/Retrieval/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv
            - Read by the parent class:
                - dataset_dir (str): cfg.DATASET.DATASET_DIR: Dataset directory. | e.g., /root/autodl-tmp/NegBench/images/Retrieval
                - num_shots (int): cfg.DATASET.NUM_SHOTS: Number of few-shot samples | -1 means using all data, 0 means zero-shot, 1 means one-shot, and so on.
                - seed (int): cfg.SEED: Random seed.
                - p_trn (float): cfg.DATASET.SPLIT[0]: Training set proportion.
                - p_val (float): cfg.DATASET.SPLIT[1]: Validation set proportion.
                - p_tst (float): cfg.DATASET.SPLIT[2]: Test set proportion.
        
        Main steps:
            1. Read additional configurations.
            2. Call the constructor of the parent class DatasetBase:
                1. Read configurations.
                2. Read data and split it into train, val, and test datasets (subclasses need to implement the get_data and get_fewshot_data methods).
                3. If num_shots >= 0, perform few-shot sampling from train and val to generate few-shot train and val datasets.
        """
        # ---Read configurations---
        self.csv_file = cfg.DATASET.CSV_FILE 
        self.num_choices = cfg.DATASET.NUM_CHOICES

        # ---Call the parent class constructor to obtain self.train, self.val, self.test, etc.---
        super().__init__(cfg)  # Call the parent class constructor

    # -----------------------Abstract methods required by the parent class DatasetBase-----------------------

    def read_split(self, split_file):
        """
        Reads the data split file (implements the abstract method of the parent class).
        
        Parameters:
            - split_file (str): Path to the (newly saved/read) split file.
        
        Returns:
            - Training, validation, and test data (list of RetrievalDatum objects).
        """
        def _convert(items):
            out = []
            for impath, num_choices, choices, correct_answer, correct_answer_type in items:
                item = RetrievalDatum(impath=impath, num_choices=num_choices, choices=choices, correct_answer=correct_answer, correct_answer_type=correct_answer_type)
                out.append(item)
            return out

        print(f"Reading split from {split_file}")  # Print information about reading the split file
        split = read_json(split_file)  # Read the JSON file
        train = _convert(split["train"])  # Convert training data
        val = _convert(split["val"])  # Convert validation data
        test = _convert(split["test"])  # Convert test data

        return train, val, test  # Return training, validation, and test data (list of RetrievalDatum objects)
    

    def save_split(self, train, val, test, split_file):
        """
        Saves the data split results to split_file (implements the abstract method of the parent class).
        
        Parameters:
            - train (list): Training dataset.
            - val (list): Validation dataset.
            - test (list): Test dataset.
            - split_file (str): Path to the data split file.

        Returns:
            - None
        """
        def _extract(items):
            out = []
            for item in items:
                impath = item.impath
                caption = item.num_choices
                out.append((impath, caption))
            return out
        train = _extract(train)  # Extract training data
        val = _extract(val)  # Extract validation data
        test = _extract(test)  # Extract test data

        split = {"train": train, "val": val, "test": test}  # Create a split dictionary

        write_json(split, split_file)  # Write to a JSON file
        print(f"Saved split to {split_file}")  # Print information about saving the split file
    

    def generate_fewshot_dataset(self, dataset, num_shots=-1, repeat=False):
        """Generates a few-shot dataset, containing only a small number of images per category (implements the abstract method of the parent class).

        Parameters:
            - dataset (list): List containing RetrievalDatum objects.
            - num_shots (int): Number of instances to sample per category. | Default -1 means returning the original dataset directly.
            - repeat (bool): Whether to repeat images if needed (default: False).
        
        Returns:
            - fewshot_dataset (list): Dataset containing a small number of images.
        """
        if num_shots <= 0:  # No few-shot sampling
            return dataset

        print(f"Creating a {num_shots}-shot dataset...")

        # Shuffle the dataset randomly
        random.shuffle(dataset)

        # Sample num_shots instances
        if len(dataset) >= num_shots:
            fewshot_dataset = random.sample(dataset, num_shots)
        else:
            if repeat:
                # If instances are insufficient and repetition is allowed, sample with replacement
                fewshot_dataset = random.choices(dataset, k=num_shots)
            else:
                # If instances are insufficient and repetition is not allowed, return all instances
                fewshot_dataset = dataset

        print(f"Few-shot dataset created with {len(fewshot_dataset)} instances.")
        return fewshot_dataset

    
    # -------------------Abstract methods/interfaces to be implemented by specific dataset subclasses-------------------

    def read_and_split_data(self):
        """
        Reads data from the CSV file and splits it into train, val, and test datasets (customized for each retrieval dataset format).

        CSV file format:
        positive_objects	negative_objects	filepath	image_id	captions
        ['person', 'bottle', 'cup', 'knife', 'spoon', 'bowl', 'broccoli', 'carrot', 'dining table', 'oven', 'sink']	['chair', 'fork', 'car']	data/coco/images/val2017/000000397133.jpg	397133	['A man in a kitchen is making pizzas, but there is no chair in sight.', 'A man in an apron stands in front of an oven with pans and bakeware nearby, without a chair in sight.', 'No fork is present, but a baker is busy working in the kitchen, rolling out dough.', 'A person stands by a stove in the kitchen, but a fork is noticeably absent.', "At the table, pies are being crafted, while a person stands by a wall adorned with pots and pans, and noticeably, there's no fork."]
        ['banana', 'orange', 'chair', 'potted plant', 'dining table', 'oven', 'sink', 'refrigerator']	['person', 'cup', 'bottle']	data/coco/images/val2017/000000037777.jpg	37777	['No cup can be seen; however, the dining table near the kitchen is set with a bowl of fruit.', 'No cup is visible in the image, but the small kitchen is equipped with various appliances and a table.', 'The kitchen is clean and empty, with no person in sight.', 'In this pristine white kitchen and dining area, no bottle is in sight.', 'A kitchen scene featuring a bowl of fruit on the table, but noticeably absent is a bottle.']
        ['person', 'traffic light', 'umbrella', 'handbag', 'cup']	['chair', 'car', 'dining table']	data/coco/images/val2017/000000252219.jpg	252219	['A person with a shopping cart is on a city street, with no car in sight.', 'No dining table is visible in the image, which shows city dwellers passing by a homeless man begging for cash.', 'On a city street, people walk past a homeless man begging, with no chair in sight.', 'On the street, a homeless man holds a cup and stands beside a shopping cart, with no sign of a car.', 'People walk by a homeless person standing on the street, with no cars in the surrounding area.']

        Needs to return:
            - Training, validation, and test data (list of RetrievalDatum objects).
        """
        raise NotImplementedError("Subclasses need to implement the read_and_split_data method") 

        

# -------------------Auxiliary classes and functions-------------------


class RetrievalDatum:
    """Retrieval data instance class, defining basic attributes.

    Parameters:
        - impath (str): Image path.
        - captions (int): Image description text.
    """

    def __init__(self, impath, captions):
        """Initializes a data instance."""
        # Ensure the image path is a string type & check if the image path is a valid file
        assert isinstance(impath, str) and check_isfile(impath)
        self._impath = impath
        self._captions = captions
    @property
    def impath(self):
        return self._impath
    
    @property
    def captions(self):
        return self.captions