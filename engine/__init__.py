from .build import TRAINER_REGISTRY, build_trainer  # isort:skip

from .base_class.TrainerBase import TrainerBase  # isort:skip
from .base_class.TrainerClsBase import TrainerClsBase
from .base_class.TrainerMcqBase import TrainerMcqBase

from .TrainerClsCoOp import TrainerClsCoOp  # isort:skip
from .TrainerMcqCoOp import TrainerMcqCoOp  # isort:skip
