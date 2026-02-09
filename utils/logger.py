import os
import sys
import time
import os.path as osp

from .tools import mkdir_if_missing

__all__ = [
    "Logger",  # Class to write console output to an external text file
    "setup_logger"  # Function to set up standard output logging
]


class Logger:
    """
    A class to write console output to an external text file.
    Args:
        fpath (str): Directory to save the log file.

    Example::
       >>> import sys
       >>> import os.path as osp
       >>> save_dir = 'output/experiment-1'
       >>> log_name = 'train.log'
       >>> sys.stdout = Logger(osp.join(save_dir, log_name))
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout  # Save the current standard output
        self.file = None  # Initialize file as None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))  # Create directory if it doesn't exist
            self.file = open(fpath, "w")  # Open file for write operations

    def __del__(self):
        self.close()  # Close the file when the object is destroyed

    def __enter__(self):
        pass  # Enter method for context manager

    def __exit__(self, *args):
        self.close()  # Exit method for context manager, close the file

    def write(self, msg):
        """Write the message to the console and, if a file exists, also to the file."""
        self.console.write(msg)  # Write the message to the console
        if self.file is not None:
            self.file.write(msg)  # If a file exists, write the message to the file

    def flush(self):
        """
        Force flush the buffer to ensure data is immediately written to the console and file.
        Ensures timely logging for each thread, preventing log overwrites or loss in multi-threaded environments.
        
        Note: When writing to a file, Python may first store data in a buffer and write it to disk later (e.g., when the file is closed or the buffer is full).
        """
        self.console.flush()  # Flush console output
        if self.file is not None:
            self.file.flush()  # Flush file output
            os.fsync(self.file.fileno())  # Ensure file content is written to disk

    def close(self):
        """Close the console and file output."""
        self.console.close()  # Close console output
        if self.file is not None:
            self.file.close()  # Close the file


def setup_logger(output=None):
    """Set up standard output logging.
    Args:
        output (str): Path to the log file (ending with .txt or .log).
    """
    if output is None:
        return  # Return if no output path is provided

    if isinstance(output, str) and (output.endswith(".txt") or output.endswith(".log")):
        fpath = output  # Use the path directly if it ends with .txt or .log
    else:
        fpath = osp.join(output, "log.txt")  # Otherwise, append "log.txt" to the output path

    # if osp.exists(fpath):
    # Ensure existing log files are not overwritten
    fpath += time.strftime("-%Y-%m-%d-%H-%M-%S")  # Add a timestamp to the file name

    sys.stdout = Logger(fpath)  # Redirect standard output to a Logger instance
