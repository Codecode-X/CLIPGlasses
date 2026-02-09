import torch
import torchvision.transforms as T
from tabulate import tabulate
from torch.utils.data import Dataset as TorchDataset
from utils import read_image, transform_image
from .datasets import build_dataset
from .samplers import build_train_sampler, build_test_sampler
from .transforms import build_train_transform, build_test_transform
from .transforms import TRANSFORM_REGISTRY
from torchvision.transforms import Normalize, Compose

class DataManager:
    """
    Data manager for loading datasets and building data loaders.
    
    Parameters:
        - cfg (CfgNode): Configuration.
        - custom_tfm_train (list): Custom training data augmentation.
        - custom_tfm_test (list): Custom testing data augmentation.
        - dataset_wrapper (DatasetWrapper): Dataset wrapper.

    Attributes:
        - dataset (DatasetBase): Dataset.
        - train_loader (DataLoader): Training data loader.
        - val_loader (DataLoader): Validation data loader.
        - test_loader (DataLoader): Testing data loader. 
        
        - CLS classification task
            - num_classes (int): Number of classes.
            - lab2cname (dict): Mapping from label to class name.
        - MCQ multiple-choice task
            - num_choices (int): Number of choices.
        

    Methods:
        - show_dataset_summary: Print dataset summary information.
        
    """
    def __init__(self, cfg, custom_tfm_train=None, custom_tfm_test=None, dataset_transform=None):
        """ 
        Initialize the data manager: build dataset and data loaders.
        
        Parameters:
            - cfg (CfgNode): Configuration.
            - custom_tfm_train (list): Custom training data augmentation.
            - custom_tfm_test (list): Custom testing data augmentation.
            - dataset_transform (TransformedDataset): Dataset transformer | Operations like tensor conversion and data augmentation.

        Main steps:
            1. Build the dataset object.
            2. Build data augmentation.
            3. Build data loaders (dataset + data augmentation).
            4. Record attributes: number of classes, label-to-class-name mapping.
            5. Record objects: dataset, training data loader, validation data loader, testing data loader.
            6. If verbose is enabled, print dataset summary information.
        """
        # --- Build the dataset object ---
        dataset = build_dataset(cfg)
        if cfg.TASK_TYPE == "CLS":  # Classification task
            self.num_classes = dataset.num_classes  # Number of classes
            self.lab2cname = dataset.lab2cname  # Mapping from label to class name
        elif cfg.TASK_TYPE == "MCQ":  # Multiple-choice task
            self.num_choices = dataset.num_choices  # Number of choices
        
        zero_shot = True if dataset.p_tst == 1.0 else False  # If test set ratio is 1.0, it indicates zero-shot evaluation

        # --- Build data augmentation ---
        if not zero_shot and custom_tfm_train is None: # Build training data augmentation
            tfm_train = build_train_transform(cfg)  # Use default training data augmentation from config
        else:
            print("* Using custom training data augmentation")
            tfm_train = custom_tfm_train  # Use custom training data augmentation
        
        if custom_tfm_test is None: # Build testing data augmentation
            tfm_test = build_test_transform(cfg)  # Use default testing data augmentation from config
        else:
            print("* Using custom testing data augmentation")
            tfm_test = custom_tfm_test  # Use custom testing data augmentation

        # --- Build data loaders (dataset + sampler + data augmentation) ---
        train_loader, val_loader, test_loader = None, None, None  # Initialize data loaders
        if not zero_shot:
            train_sampler = build_train_sampler(cfg, dataset.train) # Build training sampler
            train_loader = _build_data_loader( # Build training data loader train_loader based on config
                cfg,
                sampler=train_sampler,  # Training sampler
                data_source=dataset.train,  # Data source
                batch_size=cfg.DATALOADER.BATCH_SIZE_TRAIN,  # Batch size
                tfm=tfm_train,  # Training data augmentation
                is_train=True,  # Training mode
                dataset_transform=dataset_transform  # Dataset transformer for data conversion and augmentation
            )

            if dataset.val:  # Build validation data loader val_loader (if validation data exists)
                val_sampler = build_test_sampler(cfg, dataset.val) # Build validation sampler
                val_loader = _build_data_loader(
                    cfg,
                    sampler=val_sampler,  # Validation sampler
                    data_source=dataset.val,  # Data source
                    batch_size=cfg.DATALOADER.BATCH_SIZE_TEST,  # Batch size
                    tfm=tfm_test,  # Validation data augmentation
                    is_train=False,  # Testing mode
                    dataset_transform=dataset_transform # Dataset transformer for data conversion and augmentation
                )

        test_sampler = build_test_sampler(cfg, dataset.test) # Build testing sampler
        test_loader = _build_data_loader( # Build testing data loader test_loader
            cfg,
            sampler=test_sampler,  # Testing sampler
            data_source=dataset.test,  # Data source
            batch_size=cfg.DATALOADER.BATCH_SIZE_TEST,  # Batch size
            tfm=tfm_test,  # Testing data augmentation
            is_train=False,  # Testing mode
            dataset_transform=dataset_transform # Dataset transformer for data conversion and augmentation
        )
        
        # --- Record objects: dataset, training data loader, validation data loader, testing data loader ---
        self.dataset = dataset # Dataset
        self.train_loader = train_loader # Training data loader
        self.val_loader = val_loader # Validation data loader
        self.test_loader = test_loader # Testing data loader

        if cfg.VERBOSE:  # If verbose is enabled in config
            self.show_dataset_summary(cfg)


    def show_dataset_summary(self, cfg):
        """Print dataset summary information."""
        dataset_name = cfg.DATASET.NAME  # Dataset name

        # Build summary table
        table = []
        table.append(["Dataset", dataset_name])
        table.append(["Training Data", f"{len(self.dataset.train):,}"])
        if self.dataset.val:
            table.append(["Validation Data", f"{len(self.dataset.val):,}"])
        table.append(["Testing Data", f"{len(self.dataset.test):,}"])

        # Print table
        print(tabulate(table))


def _build_data_loader(cfg, sampler, data_source=None, batch_size=64, tfm=None, is_train=True, dataset_transform=None):
    """Build data loader.
    
    Parameters:
        - cfg (CfgNode): Configuration.
        - sampler (Sampler): Sampler.
        - data_source (list): Data source.
        - batch_size (int): Batch size.
        - tfm (list): Data augmentation.
        - is_train (bool): Whether it is training mode.
        - dataset_transform (TransformeWrapper): Data transformer | Operations like tensor conversion and data augmentation.
    Returns:
        - DataLoader: Data loader.

    Main steps:
        1. Build data loader using torch.utils.data.DataLoader with data transformer, data source, batch size, and sampler.
        2. Assert that the data loader length is greater than 0.
    """

    # Data transformer
    if dataset_transform is None:
        dataset_transform = TransformeWrapper(cfg, data_source, transform=tfm, is_train=is_train)

    # Build data loader
    data_loader = torch.utils.data.DataLoader(
        dataset_transform, # Data transformer (tensor conversion, data augmentation, etc.)
        batch_size=batch_size, 
        sampler=sampler, # Sampler
        num_workers=cfg.DATALOADER.NUM_WORKERS, # Number of worker processes
        drop_last=(is_train and len(data_source) >= batch_size), # Drop the last batch only in training mode and if the data source length is greater than or equal to the batch size
        pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA) # Pin memory only if CUDA is available and being used
    )

    assert len(data_loader) > 0
    return data_loader



class TransformeWrapper(TorchDataset):
    """
    Dataset transformation wrapper for applying transformations and augmentations to the dataset.
    
    Parameters:
        - cfg (CfgNode): Configuration.
        - data_source (list): Data source.
        - transform (list): Data augmentation.
        - is_train (bool): Whether it is training mode.

    Main functionality:
        - Apply data transformations (resize + RGB + toTensor + normalize) to each item in the data source.
        - (Optional) Apply data augmentations to each item in the data source.
        
    """

    def __init__(self, cfg, data_source, transform=None, is_train=False):
        """ 
        Initialize the dataset transformation wrapper.
        
        Main steps:
            1. Initialize attributes and retrieve relevant configuration information.
            3. Build a preprocessing pipeline that does not apply any data augmentation (resize + RGB + toTensor + normalize).
        """
        # Initialize attributes
        self.data_source = data_source  # Data source
        self.transform = transform  # Data augmentation, accepts list or tuple as input
        self.is_train = is_train  # Whether it is training mode
        # Retrieve relevant configuration information
        self.cfg = cfg  # Configuration
        self.task_type = cfg.TASK_TYPE # "CLS" classification; "MCQ" multiple-choice
        self.k_tfm = cfg.DATALOADER.K_TRANSFORMS if is_train else 1 # Number of augmentations | If training mode, get the number of augmentations; default to 1 in testing mode
        self.return_img0 = cfg.DATALOADER.RETURN_IMG0  # Whether to return the original unaugmented image | Default is False

        # Build a preprocessing pipeline that does not apply any data augmentation (resize + RGB)
        self.no_aug = Compose([TRANSFORM_REGISTRY.get("StandardNoAugTransform")(cfg),  # Standard no-augmentation preprocessing pipeline
                                T.ToTensor(),
                                Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD) if cfg.INPUT.NORMALIZE else None])

        # If K augmentations are required but no transform is provided, raise an exception
        if self.k_tfm > 1 and transform is None:
            raise ValueError("Cannot perform {} augmentations because transform is None".format(self.k_tfm))


    def __len__(self):
        """Return the length of the data source."""
        return len(self.data_source)

    def __getitem__(self, idx):
        """Retrieve a data item by index.
        
        Parameters:
        - idx (int): Index.
        
        Returns: Dictionary output: 
            - label: Class label
            - impath: Image path
            - index: Index
            - img or imgi: Augmented image (i-th augmentation) | img: First augmentation | img1: Second augmentation | ...
            - img0: Unaugmented image | Preprocessed image (resize + RGB + toTensor + normalize)
        """
        item = self.data_source[idx]  # Retrieve data item

        # Initialize output dictionary
        if self.task_type == 'CLS':
            output = {
                "index": idx,  # Index
                "impath": item.impath,  # Image path | str
                "label": item.label,  # Class label | int
            }
        elif self.task_type == 'MCQ':
            output = {
                "index": idx,  # Index | int
                "impath": item.impath,  # Image path | str
                "num_choices": item.num_choices,  # Number of choices | int
                "choices": item.choices,  # Choices | list[str]
                "correct_answer": item.correct_answer,  # Correct answer index | int
                "correct_answer_type": item.correct_answer_type,  # Correct answer type | str
            }
        # Read image and convert to tensor, store in output dictionary
        img0 = read_image(item.impath)  # Original image
        # If transform is provided, apply augmentation; otherwise, return the unprocessed original image
        if self.transform is not None:  
            self.transform = [self.transform] if not isinstance(self.transform, (list, tuple)) else self.transform # If transform is not a list or tuple, convert to list
            for i, tfm in enumerate(self.transform):  # Iterate over each transform
                img = transform_image(tfm, img0, self.k_tfm) # Apply (K times) tfm augmentation to the original image
                keyname = f"img{i + 1}" if (i + 1) > 1 else "img"  # Key name: "img", "img1", "img2", ... 
                output[keyname] = img  # Augmented image
        else: 
            output["img"] = self.no_aug(img0) # Preprocessed original image

        if self.return_img0:  # If the original unaugmented image is required
            output["img0"] = self.no_aug(img0) # Preprocessed original image

        return output  # Return output dictionary