import random
from collections import defaultdict
from ..build import DATASET_REGISTRY
from .DatasetBase import DatasetBase
from utils import read_json, write_json
from utils import check_isfile

@DATASET_REGISTRY.register()
class DatasetMcqBase(DatasetBase):
    """
    Base class for MCQ datasets.
    Inherits from the DatasetBase class.

    Public attributes (@property):
        - Attributes from the parent class DatasetBase:
            - train (list): Labeled training data.
            - val (list): Validation data (optional).
            - test (list): Test data.
        - Current read: None

    Internal attributes:
        - Attributes from the parent class DatasetBase:
            - num_shots (int): Few-shot sample count.
            - seed (int): Random seed.
            - p_trn (float): Training set ratio.
            - p_val (float): Validation set ratio.
            - p_tst (float): Test set ratio.
            - dataset_dir (str): Dataset directory.
        - Current read:
            - csv_file (str): cfg.DATASET.CSV_FILE: Annotation file path | e.g., /root/autodl-tmp/NegBench/images/MCQ/COCO_val_mcq_llama3.1_rephrased.csv
            - num_choices (int): cfg.DATASET.NUM_CHOICES: Number of answer options. e.g., 4

    Basic methods:
        - Implements abstract methods/interfaces from DatasetBase:
            - read_split: Reads data split files.
            - save_split: Saves data split files.
            - generate_fewshot_dataset: Generates few-shot datasets (usually for training sets).

    Abstract methods/interfaces (to be implemented by specific dataset subclasses):
        - read_and_split_data: Reads data and splits it into train, val, and test datasets (customized for each MCQ dataset format).
    """

    def __init__(self, cfg):
        """ 
        Initializes the basic attributes of the dataset.

        Parameters:
            - cfg (CfgNode): Configuration.

        Configuration:
            - Current read:
                - csv_path (str): cfg.DATASET.CSV_PATH: Annotation file path | e.g., /root/autodl-tmp/NegBench/images/MCQ/COCO_val_mcq_llama3.1_rephrased.csv
                - num_choices (int): cfg.DATASET.NUM_CHOICES: Number of answer options. e.g., 4
            - Parent class read:
                - dataset_dir (str): cfg.DATASET.DATASET_DIR: Dataset directory. | e.g., /root/autodl-tmp/NegBench/images/MCQ
                - num_shots (int): cfg.DATASET.NUM_SHOTS: Few-shot sample count | -1 means use all data, 0 means zero-shot, 1 means one-shot, and so on.
                - seed (int): cfg.SEED: Random seed.
                - p_trn (float): cfg.DATASET.SPLIT[0]: Training set ratio.
                - p_val (float): cfg.DATASET.SPLIT[1]: Validation set ratio.
                - p_tst (float): cfg.DATASET.SPLIT[2]: Test set ratio.
        
        Main steps:
            1. Read additional configurations.
            2. Call the parent class DatasetBase constructor:
                1. Read configurations.
                2. Read data and split it into train, val, and test datasets. (Subclasses need to implement get_data and get_fewshot_data methods)
                3. If num_shots >= 0, perform few-shot sampling from train and val to generate few-shot train and val datasets.
        """
        # ---Read configurations---
        self.csv_file = cfg.DATASET.CSV_FILE 
        self.num_choices = cfg.DATASET.NUM_CHOICES

        # ---Call the parent class constructor to get self.train, self.val, self.test, etc.---
        super().__init__(cfg)  # Call the parent class constructor

    # -----------------------Abstract methods required by the parent class DatasetBase-----------------------

    def read_split(self, split_file):
        """
        Reads data split files (implements the abstract method from the parent class).
        
        Parameters:
            - split_file (str): Path to the split file (newly saved/read).
        
        Returns:
            - Training, validation, and test data (list of MCQDatum objects).
        """
        def _convert(items):
            out = []
            for impath, num_choices, choices, correct_answer, correct_answer_type in items:
                item = MCQDatum(impath=impath, num_choices=num_choices, choices=choices, correct_answer=correct_answer, correct_answer_type=correct_answer_type)
                out.append(item)
            return out

        print(f"Reading split from {split_file}")  # Print information about reading the split file
        split = read_json(split_file)  # Read the JSON file
        train = _convert(split["train"])  # Convert training data
        val = _convert(split["val"])  # Convert validation data
        test = _convert(split["test"])  # Convert test data

        return train, val, test  # Return training, validation, and test data (list of MCQDatum objects)
    

    def save_split(self, train, val, test, split_file):
        """
        Saves data split files (implements the abstract method from the parent class).
        
        Parameters:
            - train (list): Training dataset.
            - val (list): Validation dataset.
            - test (list): Test dataset.
            - split_file (str): Path to the split file.

        Returns:
            - None
        """
        def _extract(items):
            out = []
            for item in items:
                impath = item.impath
                num_choices = item.num_choices
                choices = item.choices
                correct_answer = item.correct_answer
                correct_answer_type = item.correct_answer_type
                out.append((impath, num_choices, choices, correct_answer, correct_answer_type))
            return out
        train = _extract(train)  # Extract training data
        val = _extract(val)  # Extract validation data
        test = _extract(test)  # Extract test data

        split = {"train": train, "val": val, "test": test}  # Create a split dictionary

        write_json(split, split_file)  # Write to a JSON file
        print(f"Saved split to {split_file}")  # Print information about saving the split file
    

    def generate_fewshot_dataset(self, dataset, num_shots=-1, repeat=False):
        """Generates a few-shot dataset, containing only a few images per category (implements the abstract method from the parent class).

        Parameters:
            - dataset (list): List of MCQDatum objects.
            - num_shots (int): Number of instances to sample per category. | Default -1 means directly return the original dataset.
            - repeat (bool): Whether to repeat images if needed (default: False).
        
        Returns:
            - fewshot_dataset (list): Dataset containing a few images.
        """
        if num_shots <= 0: # No few-shot sampling
            super().generate_fewshot_dataset(dataset, num_shots, repeat)

        print(f"Creating a {num_shots}-shot dataset....")

        # Group by correct_answer_type
        grouped_data = defaultdict(list)
        for item in dataset:
            grouped_data[item.correct_answer_type].append(item)

        fewshot_dataset = []
        for correct_answer_type, items in grouped_data.items():
            if len(items) <= num_shots:
                if repeat:
                    # If samples are insufficient and repetition is allowed, repeat sampling
                    fewshot_dataset.extend(
                        items * (num_shots // len(items)) +  # Items need to be fully repeated this many times
                        items[:num_shots % len(items)]  # Remaining part that cannot be evenly divided
                    )
                else:
                    fewshot_dataset.extend(items)  # Use all samples if insufficient
            else:
                # Randomly sample num_shots samples
                fewshot_dataset.extend(random.sample(items, num_shots))

        print(f"Few-shot dataset created, containing {len(fewshot_dataset)} samples.")
        return fewshot_dataset

    
    # -------------------Specific dataset subclass implementation - Abstract methods/interfaces-------------------

    def read_and_split_data(self):
        """
        Reads data from the CSV file and splits it into train, val, and test datasets (customized for each MCQ dataset format).

        Needs to return:
            - Training, validation, and test data (list of MCQDatum objects).
        """
        raise NotImplementedError("Subclasses need to implement the read_and_split_data method") 

        

# -------------------Helper classes and functions-------------------


class MCQDatum:
    """MCQ data instance class, defining basic attributes.

    Parameters:
        - impath (str): Image path.
        - num_choices (int): Number of answer options.
        - choices (list<str>): List of answer option texts.
        - correct_answer (int): Index of the correct answer.
        - correct_answer_template (str): Type of the correct answer template (negation, hybrid, positive).
    """

    def __init__(self, impath, num_choices, choices, correct_answer, correct_answer_type):
        """Initializes a data instance."""
        # Ensure the image path is a string type & check if the image path is a valid file
        assert isinstance(impath, str) and check_isfile(impath)
        self._impath = impath
        self._num_choices = num_choices
        self._choices = choices
        self._correct_answer = correct_answer
        self._correct_answer_type = correct_answer_type

    @property
    def impath(self):
        return self._impath
    
    @property
    def num_choices(self):
        return self._num_choices
    
    @property
    def choices(self):
        return self._choices
    
    @property
    def correct_answer(self):
        return self._correct_answer
    
    @property
    def correct_answer_type(self):
        return self._correct_answer_type
