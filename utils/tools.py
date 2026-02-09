"""
Modified from https://github.com/KaiyangZhou/deep-person-reid
"""
import os
import sys
import json
import time
import errno
import numpy as np
import random
import os.path as osp
import warnings
from difflib import SequenceMatcher
import PIL
import torch
from PIL import Image
from six.moves import urllib
from yacs.config import CfgNode

__all__ = [
    "mkdir_if_missing",  # Create directory if missing
    "check_isfile",  # Check if the path is a file
    "read_json",  # Read JSON file
    "write_json",  # Write JSON file
    "set_random_seed",  # Set random seed
    "download_url",  # Download file from URL
    "read_image",  # Read image
    "collect_env_info",  # Collect environment information
    "listdir_nohidden",  # List non-hidden items
    "get_most_similar_str_to_a_from_b",  # Get the most similar string
    "check_availability",  # Check availability
    "tolist_if_not",  # Convert to list
    "load_yaml_config",  # Load YAML configuration
]


def mkdir_if_missing(dirname):
    """Create directory if missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def check_isfile(fpath):
    """Check if the given path is a file.
    Args:
        fpath (str): File path.
    Returns:
       bool
    """
    isfile = osp.isfile(fpath)
    if not isfile:
        warnings.warn('File not found at "{}"'.format(fpath))
    return isfile


def read_json(fpath):
    """Read JSON file from path."""
    with open(fpath, "r") as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    """Write JSON file."""
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, "w") as f:
        json.dump(obj, f, indent=4, separators=(",", ": "))


def set_random_seed(seed):
    """Set random seed."""
    random.seed(seed)  # Set seed for Python
    np.random.seed(seed)  # Set seed for NumPy
    torch.manual_seed(seed)  # Set seed for CPU
    torch.cuda.manual_seed_all(seed)  # Set seed for all GPUs


def download_url(url, dst):
    """Download file from URL to destination path.

    Args:
        url (str): URL of the file to download.
        dst (str): Destination path.
    """
    print('* url="{}"'.format(url))
    print('* destination="{}"'.format(dst))

    def _reporthook(count, block_size, total_size):
        """ Callback function: used for download progress reporting."""
        global start_time
        if count == 0:
            start_time = time.time()
            return
        duration = time.time() - start_time
        progress_size = int(count * block_size)
        speed = int(progress_size / (1024 * duration))
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(
            "\r...%d%%, %d MB, %d KB/s, %d seconds elapsed" %
            (percent, progress_size / (1024 * 1024), speed, duration)
        )
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dst, _reporthook)  # Download file
    sys.stdout.write("\n")


def read_image(path):
    """Read image from path using ``PIL.Image`` and convert to RGB mode.
    Args:
        path (str): Image path.
    Returns:
        PIL Image
    """
    return Image.open(path).convert("RGB")


def collect_env_info():
    """Return environment information string.
    Includes PyTorch and Pillow version information."""
    # Code source: github.com/facebookresearch/maskrcnn-benchmark
    from torch.utils.collect_env import get_pretty_env_info

    env_str = get_pretty_env_info()
    env_str += "\n        Pillow ({})".format(PIL.__version__)
    return env_str


def listdir_nohidden(path, sort=False):
    """List non-hidden files in a directory.
    Args:
         path (str): Directory path.
         sort (bool): Whether to sort the items.
    """
    items = [f for f in os.listdir(path) if not f.startswith(".")]  # List non-hidden items
    if sort:
        items.sort()
    return items


def get_most_similar_str_to_a_from_b(a, b):
    """Return the most similar string to `a` from list `b`.
    Args:
        a (str): Target string.
        b (list): List of candidate strings.
    """
    highest_sim = 0  # Highest similarity
    chosen = None  # Chosen string
    for candidate in b:
        sim = SequenceMatcher(None, a, candidate).ratio()  # Compute similarity
        if sim >= highest_sim:
            highest_sim = sim
            chosen = candidate 
    return chosen  # Return the most similar string


def check_availability(requested, available):
    """Check if an element is available in the list.
    Args:
        requested (str): Target string.
        available (list): List of available strings.
    """
    if requested not in available:  # If the requested string is not in the available list
        psb_ans = get_most_similar_str_to_a_from_b(requested, available)
        raise ValueError(f"The requested string should be one of [{available}], but got [{requested}], (Did you mean [{psb_ans}]?)")


def tolist_if_not(x):
    """Convert to list."""
    if not isinstance(x, list):
        x = [x]
    return x



def load_yaml_config(config_path, save=False, modify_fn=None):
    """
    Load YAML configuration file into a CfgNode object and freeze the configuration.
    
    Args:
        - config_path (str): Path to the configuration file.
        - save (bool, optional): Whether to save the configuration file to the output directory.
        - modify_fn (callable, optional): Optional modification function to modify the configuration object.
    
    Returns:
        - cfg (CfgNode): Configuration object.
    """   
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = CfgNode.load_cfg(f)

    if modify_fn is not None:
        modify_fn(cfg)

    if save:
        # Save configuration file to output directory
        mkdir_if_missing(cfg.OUTPUT_DIR)
        save_path = osp.join(cfg.OUTPUT_DIR, 'config.yaml')
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(cfg.dump())  # Write directly as string

    # Freeze configuration to prevent modification   
    cfg.freeze()
    return cfg