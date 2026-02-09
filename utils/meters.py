from collections import defaultdict
import torch

__all__ = [
    "AverageMeter",  # Compute and store the average and current value
    "MetricMeter"  # Store a set of metric values
]


class AverageMeter:
    """Compute and store the average and current value.

    Example::
        >>> # 1. Initialize a meter to track loss
        >>> losses = AverageMeter()
        >>> # 2. Update the meter after each mini-batch
        >>> losses.update(loss_value, batch_size)
    """

    def __init__(self, ema=False):
        """
        Args:
            ema (bool, optional): Whether to apply Exponential Moving Average (EMA),
                                  which is more sensitive to new data and reflects changes faster.
        """
        self.ema = ema  # Whether to use Exponential Moving Average
        self.reset()

    def reset(self):
        """Reset all values: val (current loss value), avg, sum, count = 0."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update the meter.
        Args:
            val (float): batch_mean_loss (mean loss value for the current batch).
            n (int): Number of samples.
        """
        # If val is of type torch.Tensor, convert it to a Python scalar
        if isinstance(val, torch.Tensor):
            val = val.item()

        # Update current value, sum, and count
        self.val = val  
        self.sum += val * n  # Mean loss value for the current batch * number of samples in the batch
        self.count += n

        # Update the average based on whether EMA is used
        # Exponential Moving Average (EMA) is a weighted moving average method that assigns higher weights
        # to recent data while gradually reducing the weights of older data.
        if self.ema:
            self.avg = self.avg * 0.9 + self.val * 0.1
        else:
            self.avg = self.sum / self.count


class MetricMeter:
    """Store a set of metric values.

    Example::
        >>> # 1. Create a MetricMeter instance
        >>> metric = MetricMeter()
        >>> # 2. Update using a dictionary as input
        >>> input_dict = {'loss_1': value_1, 'loss_2': value_2}
        >>> metric.update(input_dict)
        >>> # 3. Convert to string and print
        >>> print(str(metric))
    """

    def __init__(self, delimiter=" "):
        """
        Args:
            delimiter (str): Separator between metrics.
        """
        self.meters = defaultdict(AverageMeter)  # Dictionary to store average and current values of metrics
        self.delimiter = delimiter  # Separator

    def update(self, input_dict):
        """Update metric values.
        Args:
            input_dict (dict): Dictionary containing metric names and values to update.
        """
        # Return if the input dictionary is empty
        if input_dict is None:
            return

        # Raise a TypeError if the input is not a dictionary
        if not isinstance(input_dict, dict):
            raise TypeError("MetricMeter.update() input must be a dictionary")

        # Update the value of each metric
        for k, v in input_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()  # Convert torch.Tensor to a Python scalar
            self.meters[k].update(v)  # Update the metric value

    def __str__(self):
        """Convert all metrics to a string and join them."""
        output_str = []
        for name, meter in self.meters.items():
            output_str.append(f"{name} {meter.val:.4f} ({meter.avg:.4f})")
        return self.delimiter.join(output_str)
