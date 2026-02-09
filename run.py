""" 
Usage examples:

Validation
    - CLS:
        - python run.py --train --config_path config/CLS/Clip-VitB32-ep10-Caltech101-AdamW.yaml --output-dir output
        - python run.py --train --config_path config/CLS/CLS-CoOp-VitB16-ep50-Caltech101-SGD.yaml --output-dir output
    - MCQ:
        - python run.py --train --config_path config/MCQ/MCQ-CoOp-VitB16-ep200-CocoMcq-SGD.yaml --output-dir output
        - python run.py --train --config_path config/MCQ/MCQ-CoOp-VitB16-ep200-CocoMcq-Aug-SGD.yaml --output-dir output

Testing 
    - CLS:
        - python run.py --eval_only --config_path output/25-03-31-19-40-20/config.yaml --output-dir output --model_dir output/25-03-31-19-40-20 --load_epoch 5
        # Test CoOp official pretrained weights on ImageNet
        - python run.py --eval_only --config_path config/CoOp-VitB16-ep50-Caltech101-SGD.yaml --output-dir output --model_dir output/seed1 --load_epoch 50
    - MCQ:
        # Test model at epoch=50 
        - python run.py --eval_only --config_path config/MCQ/MCQ-CoOp-VitB16-ep200-CocoMcq-Aug-SGD.yaml --output-dir output --model_dir /root/X/output/CoOp_Aug_bep68_ac8767 --load_epoch 50
        # Test the best model 
        - python run.py --eval_only --config_path config/MCQ/MCQ-CoOp-VitB16-ep200-CocoMcq-Aug-SGD.yaml --output-dir output --model_dir /root/X/output/CoOp_Aug_bep68_ac8767
        # Test the initial untrained model 
        - python run.py --eval_only --config_path config/MCQ/MCQ-CoOp-VitB16-ep200-CocoMcq-Aug-SGD.yaml --output-dir output --model_dir /root/X/output/CoOp_Aug_bep68_ac8767 --load_epoch 0
        # Test the initial untrained model in zero-shot mode (i.e., all data as test set)
        - python run.py --eval_only --config_path config/MCQ/MCQ-CoOp-VitB16-eval0shot-CocoMcq.yaml --output-dir output --model_dir /root/X/output/CoOp_Aug_bep68_ac8767 --load_epoch 0

    """

from utils import load_yaml_config, setup_logger, set_random_seed
from engine import build_trainer
import argparse
import torch
import os.path as osp
import datetime

def reset_cfg(cfg, args):
    """ Override configuration (cfg) with arguments (args). """
    cfg.OUTPUT_DIR = args.output_dir
    cfg.RESUME = args.resume
    return cfg

def main(args):
    assert args.train or args.eval_only, "At least one mode (train or eval_only) must be set!"
    assert args.train != args.eval_only, "Train and eval_only modes cannot be set simultaneously!" 

    def modify_fn(cfg):
        """ Modify configuration function """
        cfg = reset_cfg(cfg, args) # Override cfg with args
        cfg.OUTPUT_DIR = osp.join(cfg.OUTPUT_DIR,  # Fix: Output directory = cfg.OUTPUT_DIR + current timestamp 
                                    datetime.datetime.now().strftime(r"%y-%m-%d-%H-%M-%S"))
    # Load and modify configuration    
    cfg = load_yaml_config(args.config_path, save=True, modify_fn=modify_fn) 
    
    # Set up logger
    setup_logger(cfg.OUTPUT_DIR) # Set up logger
    
    print("\n=======Configuration Info=======\n" + str(cfg) + "\n=======Configuration Info=======\n") # Print configuration for verification

    # -----Initialization-----
    # Fix random seed if set
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)

    # Optimize CUDA performance if supported and enabled in configuration
    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    # Build trainer
    trainer = build_trainer(cfg)

    # Test and train
    if args.eval_only: # Test mode
        print("==========Evaluation Only Mode==========")
        assert args.model_dir != '', "Model directory must be provided in evaluation mode!"
        if args.load_epoch > 0: # Load model at specified epoch
            print(f"Testing model at specified epoch: {args.load_epoch}")
            trainer.load_model(args.model_dir, epoch=args.load_epoch)
        elif args.load_epoch == -1: # Test the best model if -1
            print("Testing the best model")
            trainer.load_model(args.model_dir, epoch=None)
        elif args.load_epoch == 0: # Test the untrained model if 0
            print("Testing the untrained model")
            pass
        trainer.test()
        return
    else: # Train mode
        print("==========Training Mode==========")
        trainer.train(start_epoch=0,
                      max_epoch=int(cfg.TRAIN.MAX_EPOCH))


if __name__ == "__main__":
    default_config_path = 'config/defaults.yaml' # Configuration file path
    default_output_dir = 'output' # Output directory
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_path', type=str, default=default_config_path, help='Path to the configuration file')
    parser.add_argument("--output-dir", type=str, default=default_output_dir, help="Output directory")
    parser.add_argument('--seed', type=int, default=-1, help='Random seed')

    parser.add_argument("--resume", type=str, default="", help="Checkpoint directory (resume training from this directory)")
    parser.add_argument('--load_epoch', type=int, default=-1, help='Epoch of the model to load, 0 means testing the untrained model')

    parser.add_argument('--train', action='store_true', help='Set to training mode')
    parser.add_argument('--eval_only', action='store_true', help='Set to evaluation-only mode')
    parser.add_argument('--model_dir', type=str, default='', help='Directory of the model to evaluate')

    args = parser.parse_args()
    main(args)