import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict
import torch
from sklearn.metrics import f1_score, confusion_matrix
from .build import EVALUATOR_REGISTRY
from .EvaluatorBase import EvaluatorBase

@EVALUATOR_REGISTRY.register()
class EvaluatorCls(EvaluatorBase):
    """Evaluator for classification tasks."""

    def __init__(self, cfg, dm):
        """Initialize the classification evaluator.
        Args:
            - cfg (CfgNode): Configuration.
            - dm (Dataset): Dataset manager.
        
        Attributes:
            - lab2cname (dict): Mapping from label to class name.
            - correct (int): Number of correct predictions.
            - total (int): Total number of samples.
            - y_true (list): Ground truth labels.
            - y_pred (list): Predicted labels.
            - per_class (bool): Whether to evaluate results per class.
            - per_class_res (dict): Results per class.
            - calc_cmat (bool): Whether to calculate the confusion matrix.
        """
        super().__init__(cfg, dm)
        self.lab2cname = dm.lab2cname  # Mapping from label to class name {label: classname}

        self.correct = 0  # Number of correct predictions
        self.total = 0  # Total number of samples
        
        self.y_true = []  # Ground truth labels
        self.y_pred = []  # Predicted labels

        self.per_class = cfg.EVALUATOR.per_class  # Whether to evaluate results per class
        self.per_class_res = None  # Results per class
        if self.per_class:  # If evaluating results per class
            assert self.lab2cname is not None
            self.per_class_res = defaultdict(list)  # Dictionary to record results for each class
        
        self.calc_cmat = cfg.EVALUATOR.calc_cmat  # Whether to calculate the confusion matrix

    def reset(self):
        """(Override parent method) Reset the evaluator state."""
        self.correct = 0
        self.total = 0
        self.y_true = []
        self.y_pred = []
        if self.per_class_res is not None:
            self.per_class_res = defaultdict(list)

    def process(self, model_output, gt):
        """(Override parent method) Process model output and ground truth labels.
        Args:
            model_output (torch.Tensor): Model output [batch, num_classes]
            gt (torch.LongTensor): Ground truth labels [batch]
        """
        pred = model_output.max(1)[1]  # Get the predicted class for each sample
        matches = pred.eq(gt).float()  # Compute whether predictions are correct
        self.correct += int(matches.sum().item())  # Accumulate the number of correct predictions
        self.total += gt.shape[0]  # Accumulate the total number of samples

        self.y_true.extend(gt.data.cpu().numpy().tolist())  # Record ground truth labels
        self.y_pred.extend(pred.data.cpu().numpy().tolist())  # Record predicted labels

        if self.per_class_res is not None:  # If results per class need to be recorded
            for i, label in enumerate(gt):  # Iterate over each sample
                label = label.item()  # Get the label value
                matches_i = int(matches[i].item())  # Get the match result for this sample
                self.per_class_res[label].append(matches_i)  # Record the match result for this class

    def evaluate(self):
        """(Override parent method) Compute and return evaluation results."""
        results = OrderedDict()  # Ordered dictionary to store evaluation results
        
        # Overall evaluation results
        acc = 100.0 * self.correct / self.total  # Compute accuracy
        err = 100.0 - acc  # Compute error rate
        macro_f1 = 100.0 * f1_score(  # Compute macro-average F1 score
            self.y_true,
            self.y_pred,
            average="macro",
            labels=np.unique(self.y_true)
        )  
        results["accuracy"] = acc  # Store accuracy
        results["error_rate"] = err  # Store error rate
        results["macro_f1"] = macro_f1  # Store macro-average F1 score
        print(  # Print evaluation results
            "=> result\n"
            f"* total: {self.total:,}\n"
            f"* correct: {self.correct:,}\n"
            f"* accuracy: {acc:.1f}%\n"
            f"* error: {err:.1f}%\n"
            f"* macro_f1: {macro_f1:.1f}%"
        )  
        
        # Results per class
        if self.per_class_res is not None: 
            labels = list(self.per_class_res.keys())  # Get all class labels
            labels.sort()  # Sort the labels
            print("=> per-class result")
            accs = []  # List to store accuracy for each class
            for label in labels:
                classname = self.lab2cname[label]  # Get the class name
                res = self.per_class_res[label]  # Get the match results for this class
                correct = sum(res)  # Compute the number of correct predictions for this class
                total = len(res)  # Compute the total number of samples for this class
                acc = 100.0 * correct / total  # Compute accuracy for this class
                accs.append(acc)  # Add accuracy to the list
                print(  # Print evaluation results for this class
                    f"* class: {label} ({classname})\t"
                    f"total: {total:,}\t"
                    f"correct: {correct:,}\t"
                    f"acc: {acc:.1f}%"
                )  
            mean_acc = np.mean(accs)  # Compute mean accuracy
            print(f"* average: {mean_acc:.1f}%")
            results["perclass_accuracy"] = mean_acc  # Store mean accuracy for all classes

        # Confusion matrix results
        if self.calc_cmat:
            cmat = confusion_matrix(self.y_true, self.y_pred, normalize="true")  # Compute confusion matrix
            save_path = osp.join(self.cfg.OUTPUT_DIR, "cmat.pt")  # Path to save the confusion matrix
            torch.save(cmat, save_path)  # Save the confusion matrix
            print(f"Confusion matrix is saved to {save_path}")  # Print the save path for the confusion matrix

        return results  # Return evaluation results
