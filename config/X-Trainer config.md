# X-Trainer Configuration Guide

> by: Junhao Xiao - https://github.com/Codecode-X/X-Trainer.git



## Global Configuration

| Option                | Type | Example Value | Description                                                                 |
| --------------------- | ---- | ------------- | --------------------------------------------------------------------------- |
| **cfg.TASK_TYPE**     | str  | "CLS"         | "CLS" for classification; "MCQ" for multiple choice; "RETRIEVAL" for retrieval |
| **cfg.VERBOSE**       | bool | True          | Enable detailed logging (e.g., training progress, metrics, etc.)            |
| **cfg.SEED**          | int  | 42            | Global random seed to ensure experiment reproducibility                     |
| **cfg.USE_CUDA**      | bool | True          | Automatically detect and use available GPU for acceleration                 |
| **cfg.OUTPUT_DIR**    | str  | "./output"    | Output directory (stores logs/models/evaluation results)                   |
| **cfg.RESUME**        | str  | ""            | Path to resume training (must include `checkpoint.pth` file)               |

---



## Training Engine Configuration

### Basic Configuration
| Option                        | Type | Example       | Description                                                                 |
| ----------------------------- | ---- | ------------- | --------------------------------------------------------------------------- |
| **cfg.TRAINER.NAME**          | str  | "TrainerClsClip" | Trainer class name (e.g., `TrainerClip`)                                   |
| **cfg.TRAINER.PREC**          | str  | "amp"         | Training precision:<br>`fp32`-full precision, `fp16`-half precision, `amp`-automatic mixed precision (memory optimization) |
| **cfg.TRAINER.FROZEN**        | bool | False         | Freeze base network layers (train only classification head)                |

### Training Process
| Option                        | Type | Example | Description                                                                 |
| ----------------------------- | ---- | ------- | --------------------------------------------------------------------------- |
| **cfg.TRAIN.DO_EVAL**         | bool | True    | Validate the model after each epoch (save the best model)                   |
| **cfg.TRAIN.NO_TEST**         | bool | False   | Skip the testing phase after training                                       |
| **cfg.TRAIN.CHECKPOINT_FREQ** | int  | 5       | Model saving frequency (unit: epoch)                                       |
| **cfg.TRAIN.PRINT_FREQ**      | int  | 50      | Training log print interval (unit: batch)                                  |
| **cfg.TRAIN.MAX_EPOCH**       | int  | 100     | Maximum training epochs                                                    |

### Testing Process
| Option                   | Type | Example       | Description                                                                 |
| ------------------------ | ---- | ------------- | --------------------------------------------------------------------------- |
| **cfg.TEST.FINAL_MODEL** | str  | "best_val"    | Model selection for testing:<br>`best_val`-best on validation set, `last_step`-final weights |
| **cfg.TEST.SPLIT**       | str  | "test"        | Dataset split for testing:<br>`val`-validation set, `test`-test set         |

---



## Data Management Configuration

### Dataset

#### Basic Configuration

| Option                      | Type | Example                | Description                                                                 |
| --------------------------- | ---- | ---------------------- | --------------------------------------------------------------------------- |
| **cfg.DATASET.NAME**        | str  | "Caltech101"           | Dataset class name                                                          |
| **cfg.DATASET.DATASET_DIR** | str  | "/root/caltech-101"    | Root directory path of the dataset                                          |
| **cfg.DATASET.SPLIT**       | list | [0.7, 0.1, 0.2]        | Train/validation/test split ratios (sum ≤ 1)                                |
| **cfg.DATASET.NUM_SHOTS**   | int  | -1                     | Number of samples per class:<br>`-1`-full, `0`-zero-shot, `≥1`-few-shot     |

#### CLS Dataset Basic Configuration

| Option                    | Type | Example                                     | Description       |
| ------------------------- | ---- | ------------------------------------------ | ----------------- |
| **cfg.DATASET.IMAGE_DIR** | str  | "/root/caltech-101/101_ObjectCategories"   | Dataset image directory |

#### MCQ Dataset Basic Configuration

| Option                      | Type | Example                                | Description       |
| --------------------------- | ---- | ------------------------------------- | ----------------- |
| **cfg.DATASET.CSV_FILE**    | str  | COCO_val_mcq_llama3.1_rephrased.csv   | Dataset label file |
| **cfg.DATASET.NUM_CHOICES** | int  | 4                                     | Number of choices |

#### Specific Dataset Configuration

| Dataset         | Option | Type | Example | Description       |
| --------------- | ------ | ---- | ------- | ----------------- |
| **Caltech101**  | -      | -    | -       | No additional configuration |

-----


### Data Loading (dataloader)
| Option                              | Type | Example | Description                                                                 |
| ----------------------------------- | ---- | ------- | --------------------------------------------------------------------------- |
| **cfg.DATALOADER.BATCH_SIZE_TRAIN** | int  | 32      | Training batch size                                                        |
| **cfg.DATALOADER.BATCH_SIZE_TEST**  | int  | 100     | Testing batch size                                                         |
| **cfg.DATALOADER.NUM_WORKERS**      | int  | 4       | Number of parallel data loading processes (recommended = number of CPU cores) |
| **cfg.DATALOADER.K_TRANSFORMS**     | int  | 1       | Number of times each augmentation is applied to the original image (horizontally) |
| **cfg.DATALOADER.RETURN_IMG0**      | bool | False   | Whether to return the original unaugmented image (for visualization or contrastive learning) |

---


### Data Sampling (samplers)

| Option                    | Type | Example                | Description               |
| ------------------------- | ---- | ---------------------- | ------------------------- |
| **cfg.SAMPLER.TRAIN_SP**  | str  | "RandomSampler"        | Training data sampler class name |
| **cfg.SAMPLER.TEST_SP**   | str  | "SequentialSampler"    | Testing data sampler class name |

----


### Image Augmentation (transforms)

#### Basic Configuration
| Option                                   | Type | Example                                   | Description                                                                 |
| ---------------------------------------- | ---- | ---------------------------------------- | --------------------------------------------------------------------------- |
| **cfg.INPUT.SIZE**                       | int  | 224                                      | Unified input image size (must match the model)                             |
| **cfg.INPUT.INTERPOLATION**              | str  | "BICUBIC"                                | Image scaling interpolation method:<br>`BILINEAR`, `BICUBIC`, `NEAREST`    |
| **cfg.INPUT.BEFORE_TOTENSOR_TRANSFORMS** | list | `["RandomResizedCrop", "ColorJitter"]`   | List of data augmentation methods before converting to tensor               |
| **cfg.INPUT.AFTER_TOTENSOR_TRANSFORMS**  | list | `["Normalize"]`                          | List of data augmentation methods after converting to tensor                |
| **cfg.INPUT.NORMALIZE**                  | bool | True                                     | Whether to normalize                                                        |
| **cfg.INPUT.PIXEL_MEAN**                 | list | [0.48145466, 0.4578275, 0.40821073]      | Image mean                                                                  |
| **cfg.INPUT.PIXEL_STD**                  | list | [0.26862954, 0.26130258, 0.27577711]     | Image standard deviation                                                    |

#### Specific Image Augmentation Strategy Configuration Table

| Augmentation Strategy           | Option                               | Example        | Description                                                                 |
| ------------------------------- | ------------------------------------ | -------------- | --------------------------------------------------------------------------- |
| **AutoAugment**                 |                                      |                | Randomly selects from 25 best sub-policies.<br />Applicable to different datasets (ImageNet, CIFAR10, SVHN) |
| ├─ **ImageNetPolicy**           | `cfg.INPUT.ImageNetPolicy.fillcolor` | (128,128,128)  | Image fill color (RGB value)                                               |
| ├─ **CIFAR10Policy**            | `cfg.INPUT.CIFAR10Policy.fillcolor`  | (128,128,128)  |                                                                             |
| └─ **SVHNPolicy**               | `cfg.INPUT.SVHNPolicy.fillcolor`     | (128,128,128)  |                                                                             |
| **RandomAugment**               |                                      |                | Randomly combines augmentation operations                                   |
| ├─ **RandomIntensityAugment**   | `cfg.INPUT.RandomIntensityAugment.n` | 2              | Randomly selects n augmentation operations                                 |
|                                 | `cfg.INPUT.RandomIntensityAugment.m` | 10             | Augmentation intensity (0-30, higher values mean stronger effects)         |
| └─ **ProbabilisticAugment**     | `cfg.INPUT.ProbabilisticAugment.n`   | 2              | Randomly selects n augmentation operations                                 |
|                                 | `cfg.INPUT.ProbabilisticAugment.p`   | 0.6            | Probability of applying each operation                                     |
| **Cutout**                      |                                      |                | Randomly masks regions of the image                                        |
|                                 | `cfg.INPUT.Cutout.n_holes`           | 1              | Number of masked regions per image                                         |
|                                 | `cfg.INPUT.Cutout.length`            | 16             | Side length of each square masked region (pixels)                          |
| **GaussianNoise**               |                                      |                | Adds Gaussian noise                                                        |
|                                 | `cfg.INPUT.GaussianNoise.mean`       | 0              | Noise mean (usually kept at 0)                                             |
|                                 | `cfg.INPUT.GaussianNoise.std`        | 0.15           | Noise intensity (higher values mean more noticeable noise)                 |
|                                 | `cfg.INPUT.GaussianNoise.p`          | 0.5            | Application probability (between 0 and 1)                                  |
| **Random2DTranslation**         |                                      |                | Random cropping after scaling                                               |
|                                 | `cfg.INPUT.Random2DTranslation.p`    | 0.5            | Execution probability (0=disabled, 1=always applied)                       |
| **StandardNoAugTransform**      | -                                    | -              | Standardized non-augmentation image transformation pipeline                |
| **RandomResizedCrop**           |                                      |                |                                                                             |
|                                 | `cfg.INPUT.RandomResizedCrop.scale`  | [0.08, 1.0]    | Scale range for random cropping                                            |

---



## Model Configuration

### Basic Configuration

| Option                          | Type | Example                              | Description                 |
| ------------------------------- | ---- | ------------------------------------ | --------------------------- |
| **cfg.MODEL.NAME**              | str  | "Clip"                              | Model class name (e.g., `Clip`) |
| **cfg.MODEL.INIT_WEIGHTS_PATH** | str  | "log/my_model/model-best.pth.tar"   | Path to pre-trained weights |

### Specific Model Configuration

| Specific Model | Option                      | Type | Example            | Description                  |
| -------------- | --------------------------- | ---- | ------------------ | ---------------------------- |
| **Clip**       |                             |      |                    | Classic contrastive learning model |
| ├─             | **cfg.MODEL.pretrained**    | str  | "ViT-B/16"         | Pre-trained model name for Clip |
| └─             | **cfg.MODEL.download_root** | str  | "~/.cache/clip"    | Directory to save downloaded Clip pre-trained weights |
| **CoOpClip**   |                             |      |                    |                                |
| ├─             | **cfg.MODEL.pretrained**    | str  | "ViT-B/16"         | Pre-trained model name for Clip |
| ├─             | **cfg.MODEL.download_root** | str  | "~/.cache/clip"    | Directory to save downloaded Clip pre-trained weights |
| ├─             | **cfg.MODEL.init_ctx**      | str  | "a photo of a"     | Initial context in the prompt |

---



## Learning Rate Strategy Configuration (LR)

### Scheduler (lr_scheduler)

#### Basic Configuration

| Option                    | Type | Example          | Description                                                     |
| ------------------------- | ---- | ---------------- | --------------------------------------------------------------- |
| **cfg.LR_SCHEDULER.NAME** | str  | "MultiStepLR"    | `MultiStepLR` (step decay), `CosineLR` (cosine annealing), `SingleStepLrScheduler` (single step decay) |

#### Specific Scheduler Configuration

| Specific Scheduler         | Option                      | Type      | Example         | Description                     |
| -------------------------- | --------------------------- | --------- | --------------- | ------------------------------- |
| **MultiStepLrScheduler**   |                             |           |                 | Multi-step learning rate scheduler |
| ├─                        | cfg.LR_SCHEDULER.MILESTONES | list[int] | [30, 60, 90]    | List of epochs where the learning rate decreases |
| └─                        | cfg.LR_SCHEDULER.GAMMA      | float     | 0.1             | Learning rate decay factor      |
| **SingleStepLrScheduler**  |                             |           |                 | Single-step learning rate scheduler |
| ├─                        | cfg.LR_SCHEDULER.STEP_SIZE  | int       | 50              | Step size (number of epochs before reducing the learning rate) |
| └─                        | cfg.LR_SCHEDULER.GAMMA      | float     | 0.1             | Learning rate decay factor      |
| **CosineLrScheduler**      | -                           | -         |                 | Cosine learning rate scheduler  |

### Warmup Scheduler (lr_scheduler/warmup)

#### Basic Configuration

| Option                                     | Type | Example                    | Description                     |
| ------------------------------------------ | ---- | -------------------------- | -------------------------------- |
| **cfg.LR_SCHEDULER.WARMUP.NAME**           | str  | "LinearWarmupScheduler"    | Warmup scheduler class name      |
| **cfg.LR_SCHEDULER.WARMUP.WARMUP_RECOUNT** | bool | True                       | Whether to reset the cycle after warmup |
| **cfg.LR_SCHEDULER.WARMUP.EPOCHS**         | int  | 5                          | Warmup duration (epochs)         |

#### Specific Warmup Scheduler Configuration

| Specific Warmup Scheduler   | Option                              | Type  | Example  | Description       |
| --------------------------- | ----------------------------------- | ----- | -------- | ----------------- |
| **ConstantWarmupScheduler** | **cfg.LR_SCHEDULER.WARMUP.CONS_LR** | float | 0.001    | Constant learning rate |
| **LinearWarmupScheduler**   | **cfg.LR_SCHEDULER.WARMUP.MIN_LR**  | float | 1e-6     | Minimum learning rate  |

---



## Evaluator Configuration

### Basic Configuration

| Option                 | Type | Example                      | Description       |
| ---------------------- | ---- | ---------------------------- | ----------------- |
| **cfg.EVALUATOR.NAME** | str  | "EvaluatorClassification"    | Evaluator class name |

### Specific Evaluator Configuration

| Evaluator                  | Option                  | Type | Example | Description                  |
| -------------------------- | ----------------------- | ---- | ------- | ---------------------------- |
| **EvaluatorClassification** |                         |      |         | Evaluator for classification tasks |
| ├─                        | **cfg.EVALUATOR.per_class** | bool | True    | Whether to evaluate results per class |
| └─                        | **cfg.EVALUATOR.calc_cmat** | bool | True    | Whether to calculate the confusion matrix |

----



## Optimizer Configuration

### Basic Configuration

| Option                            | Type      | Example                 | Description                                |
| --------------------------------- | --------- | ----------------------- | ------------------------------------------ |
| **cfg.OPTIMIZER.NAME**            | str       | "AdamW"                | Optimizer class name, e.g., `"Adam"`, `"SGD"` |
| **cfg.OPTIMIZER.LR**              | float     | 0.001                  | Global learning rate                       |
| **cfg.OPTIMIZER.STAGED_LR**       | bool      | True                   | Whether to use staged learning rates       |
| ├─ **cfg.OPTIMIZER.NEW_LAYERS**   | list[str] | ["layer1", "layer2"]   | Newly added network layers (usually for specific tasks) |
| └─ **cfg.OPTIMIZER.BASE_LR_MULT** | float     | 0.1                    | Base layer learning rate scaling factor (usually <1) |


### Specific Optimizer Configuration

| Optimizer      | Option                               | Type                | Example         | Description                                   |
| -------------- | ------------------------------------ | ------------------- | --------------- | -------------------------------------------- |
| **Adam**       |                                      |                     |                 | Suitable for most deep learning tasks         |
| ├─             | **cfg.OPTIMIZER.betas**              | Tuple[float, float] | (0.9, 0.999)    | Beta parameters for Adam                     |
| ├─             | **cfg.OPTIMIZER.eps**                | float               | 1e-8            | Constant added to denominator to avoid division by zero |
| ├─             | **cfg.OPTIMIZER.weight_decay**       | float               | 0.01            | Weight decay                                 |
| └─             | **cfg.OPTIMIZER.amsgrad**            | bool                | False           | Whether to use AMSGrad                       |
| **SGD**        |                                      |                     |                 | Requires fine-tuning but may yield better results |
| ├─             | **cfg.OPTIMIZER.momentum**           | float               | 0.9             | Momentum                                    |
| ├─             | **cfg.OPTIMIZER.weight_decay**       | float               | 0.0005          | Weight decay                                |
| ├─             | **cfg.OPTIMIZER.dampening**          | float               | 0.0             | Dampening                                   |
| └─             | **cfg.OPTIMIZER.nesterov**           | bool                | True            | Whether to use Nesterov momentum            |
| **RMSprop**    |                                      |                     |                 | Suitable for non-stationary target functions |
| ├─             | **cfg.OPTIMIZER.alpha**              | float               | 0.99            | Smoothing constant                          |
| ├─             | **cfg.OPTIMIZER.eps**                | float               | 1e-8            | Constant added to denominator to avoid division by zero |
| ├─             | **cfg.OPTIMIZER.weight_decay**       | float               | 0.0001          | Weight decay                                |
| ├─             | **cfg.OPTIMIZER.momentum**           | float               | 0.9             | Momentum                                    |
| └─             | **cfg.OPTIMIZER.centered**           | bool                | False           | Whether to use centered RMSprop             |
| **RAdam**      |                                      |                     |                 | Robust version of adaptive learning rate    |
| ├─             | **cfg.OPTIMIZER.betas**              | Tuple[float, float] | (0.9, 0.999)    | Beta parameters for RAdam                   |
| ├─             | **cfg.OPTIMIZER.eps**                | float               | 1e-8            | Constant added to denominator to avoid division by zero |
| ├─             | **cfg.OPTIMIZER.weight_decay**       | float               | 0.01            | Weight decay                                |
| └─             | **cfg.OPTIMIZER.degenerated_to_sgd** | bool                | False           | Whether to degrade RAdam to SGD (no adaptivity) |
| **AdamW**      |                                      |                     |                 | Improved Adam with correct weight decay     |
| ├─             | **cfg.OPTIMIZER.betas**              | Tuple[float, float] | (0.9, 0.999)    | Beta parameters for AdamW                   |
| ├─             | **cfg.OPTIMIZER.eps**                | float               | 1e-8            | Constant added to denominator to avoid division by zero |
| ├─             | **cfg.OPTIMIZER.weight_decay**       | float               | 0.01            | Weight decay                                |
| └─             | **cfg.OPTIMIZER.warmup_steps**       | int                 | 1000            | Number of warmup steps (gradual learning rate increase) |

