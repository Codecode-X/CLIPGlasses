import os
from ..build import DATASET_REGISTRY
import pickle
from utils import mkdir_if_missing, set_random_seed

@DATASET_REGISTRY.register()
class DatasetBase:
    """
    Base class for datasets.

    Public attributes (@property): 
        - train (list): Labeled training data.
        - val (list): Validation data (optional).
        - test (list): Test data.
    
    Internal attributes:
        - num_shots (int): Number of few-shot samples.
        - seed (int): Random seed.
        - p_trn (float): Training set proportion.
        - p_val (float): Validation set proportion.
        - p_tst (float): Test set proportion.
        - dataset_dir (str): Dataset directory.
    
    Basic methods:
        - get_data: Read data and split into train, val, and test datasets.
        - get_fewshot_data: Perform few-shot sampling from train and val to generate few-shot train and val datasets.
    
    Abstract methods/interfaces:
        - To be implemented by subclasses based on task type:   
            - read_split: Read data split file.
            - save_split: Save data split file.
            - generate_fewshot_dataset: Generate few-shot dataset.

        - To be implemented by specific dataset subclasses based on data storage format:
            - read_and_split_data: Read and split data.

    """

    def __init__(self, cfg):
        """ 
        Initialize basic dataset attributes:

        Parameters:
            - cfg (CfgNode): Configuration.

        Configuration:
            - dataset_dir (str): cfg.DATASET.DATASET_DIR: Dataset directory.
            - num_shots (int): cfg.DATASET.NUM_SHOTS: Number of few-shot samples | -1 means use all data, 0 means zero-shot, 1 means one-shot, and so on.
            - seed (int): cfg.SEED: Random seed.
            - p_trn (float): cfg.DATASET.SPLIT[0]: Training set proportion.
            - p_val (float): cfg.DATASET.SPLIT[1]: Validation set proportion.
            - p_tst (float): cfg.DATASET.SPLIT[2]: Test set proportion.
            - output_dir (str): cfg.OUTPUT_DIR: Output directory.

        Main steps:
            1. Read configuration.
            2. Read data and split into train, val, and test datasets (subclass needs to implement get_data and get_fewshot_data methods).
            3. If num_shots >= 0, perform few-shot sampling from train and val to generate few-shot train and val datasets.

        """
        # ---Read configuration---
        self.num_shots = cfg.DATASET.NUM_SHOTS  # Get number of few-shot samples
        self.seed = cfg.SEED  # Get random seed
        set_random_seed(self.seed)
        self.p_trn, self.p_val, self.p_tst = cfg.DATASET.SPLIT  # Get proportions of train, val, and test sets
        assert self.p_trn + self.p_val + self.p_tst == 1  # Assert that the sum of proportions equals 1
        self.dataset_dir = cfg.DATASET.DATASET_DIR  # Get dataset directory, e.g., /root/autodl-tmp/caltech-101
        self.output_dir = cfg.OUTPUT_DIR  # Get output directory, e.g., /root/autodl-tmp/X/output

        # ---Read data and split into train, val, and test datasets---
        self._train, self._val, self._test = self.get_data() 
        
        # ---If num_shots >= 0, perform few-shot sampling from train and val to generate few-shot train and val datasets---
        if self.num_shots >= 0: # -1: Use all data, 0: zero-shot, 1: one-shot, and so on
            self._train, self._val = self.get_fewshot_data(self._train, self._val)

    # -------------------Attributes-------------------
    @property
    def train(self):
        """Return labeled training data"""
        return self._train

    @property
    def val(self):
        """Return validation data"""
        return self._val

    @property
    def test(self):
        """Return test data"""
        return self._test

    # -------------------Methods-------------------
    def get_data(self):
        """
        Read data and split into train, val, and test datasets.

        Parameters:
            - None
        
        Returns:
            - Training, validation, and test datasets.
        """
        
        split_path = os.path.join(self.output_dir, "split.json")  # Set data split file path

        # Split data based on p_trn, p_val and save split file
        train, val, test = self.read_and_split_data()  # Read and split data
        self.save_split(train, val, test, split_path)  # Save data split
        
        return train, val, test  # Return training, validation, and test datasets
    
    def get_fewshot_data(self, train, val):
        """
        Perform few-shot sampling from train and val to generate few-shot train and val datasets.
        
        Returns:
            - Few-shot training and validation datasets.
        """
        # Set few-shot dataset directory path (create if not exists) and few-shot dataset file save path
        split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")  # Set few-shot split directory path
        mkdir_if_missing(split_fewshot_dir)  # Create directory if not exists
        fewshot_dataset = os.path.join(split_fewshot_dir, f"shot_{self.num_shots}-seed_{self.seed}.pkl")  # Few-shot dataset file save path

        # If few-shot dataset file exists, load it directly; otherwise, generate few-shot dataset and save the file
        if os.path.exists(fewshot_dataset): # If few-shot dataset file exists, load it directly
            print(f"Loading preprocessed few-shot data from {fewshot_dataset}")
            with open(fewshot_dataset, "rb") as file:
                data = pickle.load(file)  # Load dataset
                train, val = data["train"], data["val"]  # Get training and validation data
        
        else: # If few-shot dataset file does not exist, generate few-shot dataset and save (generate_fewshot_dataset method needs to be implemented by subclass)
            train = self.generate_fewshot_dataset(train, num_shots=self.num_shots)  # Generate few-shot training data
            val = self.generate_fewshot_dataset(val, num_shots=min(self.num_shots, 4))  # Generate few-shot validation data
            data = {"train": train, "val": val}  # Create data dictionary
            print(f"Saving preprocessed few-shot data to {fewshot_dataset}")
            with open(fewshot_dataset, "wb") as file:
                pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)  # Save
        
        return train, val  # Return few-shot training and validation datasets
    

    # -------------------To be implemented by subclasses - Abstract methods/interfaces-------------------

    def read_split(self, split_file):
        """
        Read data split file (abstract method to be implemented by subclasses).
        Subclasses should customize based on task type (classification, regression).
        
        Parameters:
            - split_file (str): Data split file path.
        
        Returns:
            - Training, validation, and test datasets.
        """
        raise NotImplementedError
    

    def save_split(self, train, val, test, split_file):
        """
        Save data split file (abstract method to be implemented by subclasses).
        Subclasses should customize based on task type (classification, regression).
        
        Parameters:
            - train (list): Training dataset.
            - val (list): Validation dataset.
            - test (list): Test dataset.
            - split_file (str): Data split file path.
        
        Returns:
            - None
        """
        raise NotImplementedError
    
    def generate_fewshot_dataset(self, dataset, num_shots=-1, repeat=False):
        """
        Perform random sampling to generate a few-shot dataset (abstract method to be implemented by subclasses).

        Parameters:
            - dataset (list): Dataset.
            - num_shots (int): Number of few-shot samples.
            - repeat (bool): Whether to allow repeated sampling.

        Returns:
            - If num_shots < 0, return the original dataset.
            - If num_shots == 0, return an empty list.
            - If num_shots > 0, raise NotImplementedError.
        """
        # If num_shots < 0, return the original dataset
        if num_shots < 0:  # No few-shot sampling
            return dataset
        
        # If num_shots == 0, return an empty list
        if num_shots == 0:  
            print(f"Creating a zero-shot dataset....")
            return [] # zero-shot learning
        
        raise NotImplementedError # Subclass needs to implement specific few-shot sampling logic
    

    # -------------------To be implemented by specific dataset subclasses - Abstract methods/interfaces-------------------

    def read_and_split_data(self):
        """
        Read and split data (abstract method to be implemented by subclasses).
        """
        
        raise NotImplementedError