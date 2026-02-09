from .build import EVALUATOR_REGISTRY

@EVALUATOR_REGISTRY.register()
class EvaluatorBase:
    """
    Interface class for evaluators.
    
    Subclasses need to implement the following methods:
        - __init__: Initialize the evaluator.
        - reset: Reset the evaluator state.
        - process: Process model outputs and ground truth labels.
        - evaluate: Compute and return evaluation results.
    """

    def __init__(self, cfg, dm):
        """ 
        Initialize the evaluator.
        
        Args:
            - cfg (CfgNode): Configuration.
            - dm (Dataset): Dataset manager.
        """
        self.cfg = cfg
        self.dm = dm

    def reset(self):
        """Reset the evaluator state."""
        raise NotImplementedError

    def process(self, model_output, gt):
        """Process model outputs and ground truth labels.
        
        Args:
            model_output (torch.Tensor): Model outputs [batch, num_classes].
            gt (torch.LongTensor): Ground truth labels [batch].
        """
        raise NotImplementedError

    def evaluate(self):
        """Compute and return evaluation results."""
        raise NotImplementedError