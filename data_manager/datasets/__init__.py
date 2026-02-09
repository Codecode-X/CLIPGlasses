"""Specific implementation classes for datasets"""
from .build import DATASET_REGISTRY, build_dataset 

# -------Classification Dataset-------
from .Caltech101 import Caltech101 

# -------MCQ Dataset-------
from .CocoMcq import CocoMcq

# -------Retrieval Dataset-------
from .CocoRetrieval import CocoRetrieval
