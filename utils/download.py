import hashlib
import os
import os.path as osp
import tarfile
import zipfile
import gdown
import urllib
import warnings
from tqdm import tqdm

__all__ = [
    "download_weight", # Download model weight file via URL
    "download_data" # Download data and extract
]

def download_weight(url: str, root: str):
    """
    Download the weight file from the specified URL to the specified directory and return the downloaded file path.
       
    Parameters:
        - url: str, the URL of the file.
        - root: str, the directory path to download the file.
        
    Returns:
        - str, the path of the downloaded file.

    Main steps:
        1. Create the path to save the downloaded file (download_target)
        2. Get the SHA256 value of the file
        3. Check if the target path already contains the downloaded file
        4. If the file exists, verify its SHA256 value; if it matches, return the file path, otherwise re-download the file
        5. Download the file and display a progress bar
        6. After downloading, verify the SHA256 value again
    """

    # ---Create the path to save the downloaded file (download_target)---
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)
    download_target = os.path.join(root, filename)

    # ---Download the file---
    # Get the SHA256 value of the file
    expected_sha256 = url.split("/")[-2]
    # Check if the target path already contains the downloaded file
    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists but is not a regular file")
    # If the file exists, verify its SHA256 value; if it matches, return the file path, otherwise re-download the file
    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists but SHA256 checksum does not match; re-downloading the file")
    # Download the file and display a progress bar
    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    # ----File verification----
    # After downloading, verify the SHA256 value again
    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError("File downloaded but SHA256 checksum does not match")

    return download_target


def download_data(url, dst, from_gdrive=True):
    """
    Download data and extract it. Supports zip, tar, and tar.gz files. Extracted files are stored in the folder of the target path.

    Parameters:
        - url (str): The download link for the data.
        - dst (str): The target path for the downloaded file.
        - from_gdrive (bool): Whether to download from Google Drive.
    
    Returns:
        - None
    """
    # Create the parent directory of the target path if it does not exist
    if not osp.exists(osp.dirname(dst)):
        os.makedirs(osp.dirname(dst))

    if from_gdrive:
        # Use gdown to download the file
        gdown.download(url, dst, quiet=False)
    else:
        raise NotImplementedError

    print("Extracting file ...")

    # Extract zip files
    if dst.endswith(".zip"):
        zip_ref = zipfile.ZipFile(dst, "r")
        zip_ref.extractall(osp.dirname(dst))
        zip_ref.close()

    # Extract tar files
    elif dst.endswith(".tar"):
        tar = tarfile.open(dst, "r:")
        tar.extractall(osp.dirname(dst))
        tar.close()

    # Extract tar.gz files
    elif dst.endswith(".tar.gz"):
        tar = tarfile.open(dst, "r:gz")
        tar.extractall(osp.dirname(dst))
        tar.close()

    else:
        raise NotImplementedError

    print("File extracted to {}".format(osp.dirname(dst)))