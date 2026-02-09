import torch
from torch.nn import functional as F
import numpy as np

__all__ = [
    "compute_distance_matrix",  # Function to compute the distance matrix
    "compute_accuracy",  # Function to compute accuracy
    "compute_ci95"  # Function to compute the 95% confidence interval
]

def compute_distance_matrix(input1, input2, metric):
    """Function to compute the distance matrix.

    Each input matrix has the shape (n_data, feature_dim).

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.
        metric (str, optional): "euclidean" or "cosine".

    Returns:
        torch.Tensor: Distance matrix.
    """
    # Check inputs
    assert isinstance(input1, torch.Tensor)
    assert isinstance(input2, torch.Tensor)
    assert input1.dim() == 2, "Expected 2-D tensor, but got {}-D".format(input1.dim())
    assert input2.dim() == 2, "Expected 2-D tensor, but got {}-D".format(input2.dim())
    assert input1.size(1) == input2.size(1)

    if metric == "euclidean":
        distmat = _euclidean_squared_distance(input1, input2)  # Compute Euclidean distance
    elif metric == "cosine":
        distmat = _cosine_distance(input1, input2)  # Compute cosine distance
    else:
        raise ValueError(
            "Unknown distance metric: {}. "
            'Choose "euclidean" or "cosine"'.format(metric)
        )

    return distmat


def compute_accuracy(output, target, topks=(1, )):
    """Compute the accuracy for the specified top-k values (Top K).

    Args:
        output (torch.Tensor): Prediction matrix with shape (batch_size, num_classes).
        target (torch.LongTensor): Ground truth labels with shape (batch_size).
        topks (tuple, optional): Compute top-k accuracy. For example, topk=(1, 5) computes top-1 and top-5 accuracy.

    Returns:
        list: Top-k accuracy values.
    """
    maxk = max(topks)  # Get the maximum value in topk
    batch_size = target.size(0)  # Get the batch size

    if isinstance(output, (tuple, list)):  # If output is a tuple or list
        output = output[0]  # Take the first element, usually the prediction matrix of shape (batch_size, num_classes)

    _, pred = output.topk(maxk, 1, True, True)  # Get the top maxk predictions along dimension 1 (columns)
    pred = pred.t()  # Transpose the predictions
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # Compare pred and target to get a boolean matrix

    res = []  # Initialize the result list
    for k in topks:  # Iterate over each value in topk
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)  # Compute the number of correct predictions for top-k
        acc = correct_k.mul_(100.0 / batch_size)  # Compute the accuracy
        res.append(acc)  # Append the accuracy to the result list

    return res  # Return the list of TopK accuracy values


# ------Helper Functions------

def _euclidean_squared_distance(input1, input2):
    """Compute the Euclidean squared distance, i.e., the square of the L2 norm.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: Distance matrix.
    """
    m, n = input1.size(0), input2.size(0)  # Get the number of rows in the input matrices
    mat1 = torch.pow(input1, 2).sum(dim=1, keepdim=True).expand(m, n)  # Compute the square of input1
    mat2 = torch.pow(input2, 2).sum(dim=1, keepdim=True).expand(n, m).t()  # Compute the square of input2
    distmat = mat1 + mat2  # Initialize the distance matrix
    distmat.addmm_(1, -2, input1, input2.t())  # Compute the final Euclidean distance matrix
    return distmat


def _cosine_distance(input1, input2):
    """Compute the cosine distance.
    (The smaller the cosine similarity, the larger the cosine distance)

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: Distance matrix.
    """
    input1_normed = F.normalize(input1, p=2, dim=1)  # Normalize input1
    input2_normed = F.normalize(input2, p=2, dim=1)  # Normalize input2
    distmat = 1 - torch.mm(input1_normed, input2_normed.t())  # Compute the cosine distance matrix
    return distmat

def compute_ci95(results):
    """ 
    Compute the 95% confidence interval.

    Args:
        - results (list): A list containing multiple values.
    Returns:
        - float: 95% confidence interval.
    """
    return 1.96 * np.std(results) / np.sqrt(len(results))