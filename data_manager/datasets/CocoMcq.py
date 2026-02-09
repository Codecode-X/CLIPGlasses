from .build import DATASET_REGISTRY
from .base_class.DatasetMcqBase import DatasetMcqBase, MCQDatum
import pandas as pd

@DATASET_REGISTRY.register()
class CocoMcq(DatasetMcqBase):
    """
    CocoMCQ dataset class.
    Inherits from the DatasetMCQBase class.

    Publicly accessible attributes (@property):
        - Attributes from the parent class DatasetClsBase:
            - Attributes from the parent class DatasetBase:
                - train (list): Labeled training data.
                - val (list): Validation data (optional).
                - test (list): Test data.
        - Currently read: None

    Internally accessible attributes:
        - Attributes from the parent class DatasetClsBase:
            - Attributes from the parent class DatasetBase:
                - num_shots (int): Few-shot quantity.
                - seed (int): Random seed.
                - p_trn (float): Training set proportion.
                - p_val (float): Validation set proportion.
                - p_tst (float): Test set proportion.
                - dataset_dir (str): Dataset directory.
            - csv_file (str): cfg.DATASET.CSV_FILE: Annotation file path | e.g., /root/autodl-tmp/NegBench/images/MCQ/COCO_val_mcq_llama3.1_rephrased.csv
            - num_choices (int): cfg.DATASET.NUM_CHOICES: Number of answer options. e.g., 4
        - Currently read: None

    Implements abstract methods/interfaces from DatasetMCQBase:
        - read_and_split_data: Reads data and splits it into train, val, and test datasets (needs to be customized based on the format of each MCQ dataset).
    """

    def read_and_split_data(self):
        """
        Reads data and splits it into train, val, and test datasets (needs to be customized based on the format of each MCQ dataset).

        CSV file format:
        image_path	correct_answer	caption_0	caption_1	caption_2	caption_3	correct_answer_template	
        data/coco/images/val2017/000000397133.jpg	0	This image features a knife, but no car is present.	A car is present in this image, but there is no knife.	This image features a car	This image does not feature a knife.	hybrid	
        data/coco/images/val2017/000000397133.jpg	0	A chair is not present in this image.	This image shows a chair, but no spoon is present.	A chair is present in this image.	A spoon is not included in this image.	negative	
        data/coco/images/val2017/000000397133.jpg	0	A person is present in this image, but there's no fork.	This image shows a fork, with no person in sight.	A fork is shown in this image.	No person is present in this image.	hybrid	

        Needs to return:
            - Training, validation, and test data (list of Datum objects).
        """
        df = pd.read_csv(self.csv_file, sep=',')  # Read the CSV file

        # Convert each row into an MCQDatum object
        data = []
        for _, row in df.iterrows():
            impath = row['image_path']
            num_choices = self.num_choices  # 4 options per row
            choices = [row[f'caption_{i}'] for i in range(num_choices)]
            correct_answer = row['correct_answer']
            correct_answer_type = row['correct_answer_template']
            datum = MCQDatum(impath, num_choices, choices, correct_answer, correct_answer_type)
            data.append(datum)

        # Split the dataset according to proportions
        num_train = int(len(data) * self.p_trn)
        num_val = int(len(data) * self.p_val)
        train = data[:num_train]
        val = data[num_train:num_train + num_val]
        test = data[num_train + num_val:]

        return train, val, test