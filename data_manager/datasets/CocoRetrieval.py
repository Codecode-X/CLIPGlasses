from .build import DATASET_REGISTRY
from .base_class.DatasetRetriBase import DatasetRetriBase, RetrievalDatum
import pandas as pd

@DATASET_REGISTRY.register()
class CocoRetrieval(DatasetRetriBase):
    """
    CocoRetrieval dataset class.
    Inherits from the DatasetRetrievalBase class.

    External access properties (@property):
        - Properties of the parent class DatasetClsBase:
            - Properties of the parent class DatasetBase:
                - train (list): Labeled training data.
                - val (list): Validation data (optional).
                - test (list): Test data.
        - Currently reading: None

    Internal access properties:
        - Properties of the parent class DatasetClsBase:
            - Properties of the parent class DatasetBase:
                - num_shots (int): Few-shot quantity.
                - seed (int): Random seed.
                - p_trn (float): Training set proportion.
                - p_val (float): Validation set proportion.
                - p_tst (float): Test set proportion.
                - dataset_dir (str): Dataset directory.
            - csv_file (str): cfg.DATASET.CSV_FILE: Annotation file path | Example: /root/autodl-tmp/NegBench/images/Retrieval/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv
            - Currently reading: None

    Implements abstract methods/interfaces of the DatasetRetrievalBase class:
        - read_and_split_data: Reads data and splits it into train, val, and test datasets (needs to be customized based on the format of each Retrieval dataset).
    """

    def read_and_split_data(self):
        """
        Reads data and splits it into train, val, and test datasets (needs to be customized based on the format of each Retrieval dataset).

        CSV file format:
        positive_objects	negative_objects	filepath	image_id	captions
        ['person', 'bottle', 'cup', 'knife', 'spoon', 'bowl', 'broccoli', 'carrot', 'dining table', 'oven', 'sink']	['chair', 'fork', 'car']	data/coco/images/val2017/000000397133.jpg	397133	['A man in a kitchen is making pizzas, but there is no chair in sight.', 'A man in an apron stands in front of an oven with pans and bakeware nearby, without a chair in sight.', 'No fork is present, but a baker is busy working in the kitchen, rolling out dough.', 'A person stands by a stove in the kitchen, but a fork is noticeably absent.', "At the table, pies are being crafted, while a person stands by a wall adorned with pots and pans, and noticeably, there's no fork."]
        ['banana', 'orange', 'chair', 'potted plant', 'dining table', 'oven', 'sink', 'refrigerator']	['person', 'cup', 'bottle']	data/coco/images/val2017/000000037777.jpg	37777	['No cup can be seen; however, the dining table near the kitchen is set with a bowl of fruit.', 'No cup is visible in the image, but the small kitchen is equipped with various appliances and a table.', 'The kitchen is clean and empty, with no person in sight.', 'In this pristine white kitchen and dining area, no bottle is in sight.', 'A kitchen scene featuring a bowl of fruit on the table, but noticeably absent is a bottle.']
        ['person', 'traffic light', 'umbrella', 'handbag', 'cup']	['chair', 'car', 'dining table']	data/coco/images/val2017/000000252219.jpg	252219	['A person with a shopping cart is on a city street, with no car in sight.', 'No dining table is visible in the image, which shows city dwellers passing by a homeless man begging for cash.', 'On a city street, people walk past a homeless man begging, with no chair in sight.', 'On the street, a homeless man holds a cup and stands beside a shopping cart, with no sign of a car.', 'People walk by a homeless person standing on the street, with no cars in the surrounding area.']

        Needs to return:
            - Training, validation, and test data (list of Datum objects).
        """
        df = pd.read_csv(self.csv_file, sep=',')  # Read the CSV file

        # Convert each row into a RetrievalDatum object
        data = []
        for _, row in df.iterrows():
            impath = row['image_path']
            captions = row['captions']
            datum = RetrievalDatum(impath, captions)
            data.append(datum)

        # Split the dataset according to proportions
        num_train = int(len(data) * self.p_trn)
        num_val = int(len(data) * self.p_val)
        train = data[:num_train]
        val = data[num_train:num_train + num_val]
        test = data[num_train + num_val:]

        return train, val, test