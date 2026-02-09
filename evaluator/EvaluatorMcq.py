import torch
from .build import EVALUATOR_REGISTRY
from .EvaluatorBase import EvaluatorBase
from collections import OrderedDict, defaultdict

@ EVALUATOR_REGISTRY.register()
class EvaluatorMcq(EvaluatorBase):
    """
    Interface class for the evaluator.
    
    Subclasses need to implement the following methods:
        - __init__: Initialize the evaluator.
        - reset: Reset the evaluator state.
        - process: Process model outputs and ground truth labels.
        - evaluate: Compute and return evaluation results.
    """

    def __init__(self, cfg, dm):
        """ 
        Initialize the MCQ task evaluator.
        
        Args:
            - cfg (CfgNode): Configuration.
            - dm (Dataset): Dataset manager.
        """
        super().__init__(cfg, dm)

        self.cfg = cfg
        split = cfg.TEST.SPLIT 
        if split == "val" and dm.val_loader is not None: 
            data_loader = dm.val_loader
        elif split == "test" and dm.test_loader is not None:
            data_loader = dm.test_loader
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'val' or 'test'.")

        if "syn" in self.cfg.DATASET.NAME.lower(): # Map wrong answer indices to answer types for synthetic datasets
            print("=> Evaluating on synthetic dataset")
            self.wrong_answer_to_type = {0: 'positive', 1: 'hybrid', 2: 'hybrid', 3: 'negative'}
        else:
            self.wrong_answer_to_type = {1: 'hybrid', 2: 'positive', 3: 'negative'}
        
        # Initialize dictionaries to track correct answers and total questions by type
        self.correct_answers_by_type = {'positive': 0, 'negative': 0, 'hybrid': 0}
        self.total_questions_by_type = {'positive': 0, 'negative': 0, 'hybrid': 0}
        # Initialize wrong answer counts
        self.wrong_answer_counts = {k: 0 for k in self.wrong_answer_to_type.keys()}
        # Initialize prediction counts by type
        self.predictions_by_type = {'positive': 0, 'negative': 0, 'hybrid': 0}
        # Initialize nested dictionary to map question types to wrong answer types
        self.wrong_answers_by_question_type = {
            'positive': {'positive': 0, 'negative': 0, 'hybrid': 0},
            'negative': {'positive': 0, 'negative': 0, 'hybrid': 0},
            'hybrid': {'positive': 0, 'negative': 0, 'hybrid': 0}
        }

        self.total_questions = 0 # Initialize total number of questions in the dataset
        self.correct_answers_sum = 0 # Initialize correct answer count

    def reset(self):
        """Reset the evaluator state."""
        self.correct_answers_sum = 0
        self.total_questions = 0
        self.correct_answers_by_type = {'positive': 0, 'negative': 0, 'hybrid': 0}
        self.total_questions_by_type = {'positive': 0, 'negative': 0, 'hybrid': 0}
        self.wrong_answer_counts = {k: 0 for k in self.wrong_answer_to_type.keys()}
        self.predictions_by_type = {'positive': 0, 'negative': 0, 'hybrid': 0}
        self.wrong_answers_by_question_type = {
            'positive': {'positive': 0, 'negative': 0, 'hybrid': 0},
            'negative': {'positive': 0, 'negative': 0, 'hybrid': 0},
            'hybrid': {'positive': 0, 'negative': 0, 'hybrid': 0}
        }

    def process(self, logits, correct_answer, correct_answer_type):
        """Process model outputs and ground truth labels.
        Args:
            - logits (torch.Tensor): Model outputs [batch, num_classes]
            - labels (torch.LongTensor): Ground truth labels [batch]
            - correct_answer (torch.LongTensor): Correct answer indices [batch]
            - correct_answer_type (list<str>): Correct answer types (batch)
        """
        # Update total number of questions
        self.total_questions += logits.size(0) # [batch_size]=[100]
        
        # Compute the number of correct predictions in the current batch
        preds = logits.argmax(dim=1).cpu().numpy() # [batch_size]=[100]
        labels = torch.tensor(correct_answer, dtype=torch.long).cpu().numpy() # [batch_size]=[100]
        correct_predictions = (preds == labels).sum().item()

        # Accumulate correct answer count
        self.correct_answers_sum += correct_predictions

        # Update counts for each answer type and track predictions
        batch_size = logits.size(0)
        for i in range(batch_size):
            answer_type = correct_answer_type[i]  # Current question's answer type
            self.total_questions_by_type[answer_type] += 1
            
            if preds[i] == correct_answer[i]: # If prediction is correct
                self.correct_answers_by_type[answer_type] += 1
                self.predictions_by_type[answer_type] += 1
            else: # If prediction is incorrect
                wrong_answer_idx = preds[i] # Get the type of the wrong answer
                wrong_answer_type = self.wrong_answer_to_type.get(wrong_answer_idx, "unknown_type")
                self.wrong_answer_counts[wrong_answer_idx] += 1
                self.predictions_by_type[wrong_answer_type] += 1
                self.wrong_answers_by_question_type[answer_type][wrong_answer_type] += 1
    

    def evaluate(self):
        """Compute and return evaluation results."""
        total_accuracy = self.correct_answers_sum / self.total_questions # Compute overall accuracy

        # Compute accuracy for each type
        positive_accuracy = self.correct_answers_by_type['positive'] / self.total_questions_by_type['positive'] if self.total_questions_by_type['positive'] > 0 else float('nan')
        negative_accuracy = self.correct_answers_by_type['negative'] / self.total_questions_by_type['negative'] if self.total_questions_by_type['negative'] > 0 else float('nan')
        hybrid_accuracy = self.correct_answers_by_type['hybrid'] / self.total_questions_by_type['hybrid'] if self.total_questions_by_type['hybrid'] > 0 else float('nan')

        # Identify the most common wrong answer type
        most_common_wrong_answer_type = max(self.wrong_answer_counts, key=self.wrong_answer_counts.get)

        # Compute percentages for each wrong answer type
        total_wrong_answers = sum(self.wrong_answer_counts.values())
        wrong_answer_percentages = {self.wrong_answer_to_type[k]: (v / total_wrong_answers) * 100 for k, v in self.wrong_answer_counts.items()}

        # Print and return an ordered dictionary containing all computed metrics
        results = OrderedDict({
            'total_accuracy': total_accuracy,
            'positive_accuracy': positive_accuracy,
            'negative_accuracy': negative_accuracy,
            'hybrid_accuracy': hybrid_accuracy,
            'most_common_wrong_answer_type': self.wrong_answer_to_type[most_common_wrong_answer_type],
            'wrong_answer_percentages': wrong_answer_percentages,
            'predictions_by_type': self.predictions_by_type,
            'wrong_answers_by_question_type': self.wrong_answers_by_question_type
        })
        if self.cfg.VERBOSE:
            print( # Print evaluation results
                "=> result\n"
                f"Total Accuracy: {total_accuracy:.4f}\n"
                f"Positive Accuracy: {positive_accuracy:.4f}\n"
                f"Negative Accuracy: {negative_accuracy:.4f}\n"
                f"Hybrid Accuracy: {hybrid_accuracy:.4f}\n"
                f"Most Common Wrong Answer Type: {self.wrong_answer_to_type[most_common_wrong_answer_type]}\n"
                f"Wrong Answer Percentages: {wrong_answer_percentages}\n"
                f"Predictions by Type: {self.predictions_by_type}\n"
                f"Wrong Answers by Question Type: {self.wrong_answers_by_question_type}\n"
            )
        return results