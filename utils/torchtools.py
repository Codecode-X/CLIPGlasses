"""
Modified from https://github.com/KaiyangZhou/deep-person-reid
"""
import pickle
import shutil
import os.path as osp
import warnings
from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from .tools import mkdir_if_missing

__all__ = [
    "save_checkpoint", # Save checkpoint
    "load_checkpoint", # Load checkpoint
    "resume_from_checkpoint", # Resume training from checkpoint
    "open_all_layers", # Open all layers of the model for training
    "open_specified_layers", # Open specified layers of the model for training
    "count_num_param", # Count the number of parameters in the model
    "load_pretrained_weights", # Load pretrained weights into the model
    "init_network_weights", # Initialize network weights
    "transform_image", # Apply K times tfm augmentation to an image and return the results
    "standard_image_transform", # Image preprocessing transformation pipeline
    "patch_jit_model" # Fix device and dtype information for JIT models
]

def save_checkpoint(state, save_dir, is_best=False,
                    remove_module_from_keys=True, model_name="" ):
    r"""Save checkpoint.

    Args:
        state (dict): Dictionary containing model state.
        save_dir (str): Directory to save the checkpoint.
        is_best (bool, optional): If True, this checkpoint will be copied and named
            ``model-best.pth.tar``. Default is False.
        remove_module_from_keys (bool, optional): Whether to remove "module." from layer names.
            Default is True.
        model_name (str, optional): Name of the saved model.
    """
    mkdir_if_missing(save_dir) # Create save directory

    if remove_module_from_keys:
        # Remove 'module.' from state_dict keys
        state_dict = state["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]
            new_state_dict[k] = v
        state["state_dict"] = new_state_dict

    # Save model
    epoch = state["epoch"]
    if not model_name:
        model_name = "model.pth.tar-" + str(epoch)
    fpath = osp.join(save_dir, model_name)
    torch.save(state, fpath)
    print(f"Checkpoint saved to {fpath}")

    # Save current model name
    checkpoint_file = osp.join(save_dir, "checkpoint")
    checkpoint = open(checkpoint_file, "w+")
    checkpoint.write("{}\n".format(osp.basename(fpath)))
    checkpoint.close()

    if is_best:
        best_fpath = osp.join(osp.dirname(fpath), "model-best.pth.tar")
        shutil.copy(fpath, best_fpath)
        print('Best checkpoint saved to "{}"'.format(best_fpath))

def load_checkpoint(fpath):
    r"""Load checkpoint.

    Handles ``UnicodeDecodeError`` gracefully, meaning files saved in python2
    can be read from python3.

    Args:
        fpath (str): Path to the checkpoint.

    Returns:
        dict

    Example::
        >>> fpath = 'log/my_model/model.pth.tar-10'
        >>> checkpoint = load_checkpoint(fpath)
    """
    if fpath is None:
        raise ValueError("File path is None")

    if not osp.exists(fpath):
        raise FileNotFoundError('File not found "{}"'.format(fpath))

    map_location = None if torch.cuda.is_available() else "cpu"

    try:
        checkpoint = torch.load(fpath, map_location=map_location)

    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        checkpoint = torch.load(
            fpath, pickle_module=pickle, map_location=map_location
        )

    except Exception:
        print('Unable to load checkpoint from "{}"'.format(fpath))
        raise

    return checkpoint

def resume_from_checkpoint(fdir, model, optimizer=None, scheduler=None):
    r"""Resume training from checkpoint.

    This loads (1) model weights and (2) optimizer's ``state_dict`` (if ``optimizer`` is not None).

    Args:
        fdir (str): Directory where the model is saved.
        model (nn.Module): Model.
        optimizer (Optimizer, optional): Optimizer.
        scheduler (Scheduler, optional): Scheduler.

    Returns:
        int: start_epoch.

    Example::
        >>> fdir = 'log/my_model'
        >>> start_epoch = resume_from_checkpoint(fdir, model, optimizer, scheduler)
    """
    with open(osp.join(fdir, "checkpoint"), "r") as checkpoint:
        model_name = checkpoint.readlines()[0].strip("\n")
        fpath = osp.join(fdir, model_name)

    print('Loading checkpoint from "{}"'.format(fpath))
    checkpoint = load_checkpoint(fpath)
    model.load_state_dict(checkpoint["state_dict"])
    print("Model weights loaded")

    if optimizer is not None and "optimizer" in checkpoint.keys():
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("Optimizer loaded")

    if scheduler is not None and "scheduler" in checkpoint.keys():
        scheduler.load_state_dict(checkpoint["scheduler"])
        print("Scheduler loaded")

    start_epoch = checkpoint["epoch"]
    print("Previous epoch: {}".format(start_epoch))

    return start_epoch

def open_all_layers(model):
    r"""Open all layers of the model for training.

    Example::
        >>> open_all_layers(model)
    """
    model.train()
    for p in model.parameters():
        p.requires_grad = True

def open_specified_layers(model, open_layers):
    r"""Open specified layers of the model for training, while keeping other layers frozen.

    Args:
        model (nn.Module): Neural network model.
        open_layers (str or list): Layers to open for training.

    Example::
        >>> # Only model.classifier will be updated.
        >>> open_layers = 'classifier'
        >>> open_specified_layers(model, open_layers)
        >>> # Only model.fc and model.classifier will be updated.
        >>> open_layers = ['fc', 'classifier']
        >>> open_specified_layers(model, open_layers)
    """
    if isinstance(model, nn.DataParallel): # If the model is nn.DataParallel
        model = model.module # Get the model

    if isinstance(open_layers, str):
        open_layers = [open_layers]

    # Check if specified layers exist
    for layer in open_layers:
        assert hasattr(model, layer), f"{layer} is not an attribute"

    # Iterate through all submodules of the model
    for name, module in model.named_children(): 
        if name in open_layers: # Open specified layers for training
            module.train() 
            for p in module.parameters():
                p.requires_grad = True
        else: # Freeze other layers
            module.eval()
            for p in module.parameters():
                p.requires_grad = False

def count_num_param(model=None, params=None):
    r"""Count the number of parameters in the model.

    Args:
        model (nn.Module): Neural network model.
        params: Parameters of the neural network model.

    Example::
        >>> model_size = count_num_param(model)
    """
    if model is not None:
        return sum(p.numel() for p in model.parameters())

    if params is not None:
        s = 0
        for p in params:
            if isinstance(p, dict):
                s += p["params"].numel()
            else:
                s += p.numel()
        return s

    raise ValueError("Either model or params must be provided.")

def load_pretrained_weights(model, weight_path):
    r"""Load pretrained weights into the model.

    Features::
        - Incompatible layers (name or size mismatch) will be ignored.
        - Automatically handles keys containing "module.".

    Args:
        model (nn.Module): Neural network model.
        weight_path (str): Path to the pretrained weights.

    Example::
        >>> weight_path = 'log/my_model/model-best.pth.tar'
        >>> load_pretrained_weights(model, weight_path)
    """
    checkpoint = load_checkpoint(weight_path)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model_dict = model.state_dict() # Get the current state_dict of the model
    new_state_dict = OrderedDict() # Store the state_dict of matched layers with pretrained weights
    matched_layers, discarded_layers = [], [] # Matched layers, discarded layers

    # Iterate through the state_dict of pretrained weights, record matched and unmatched layers
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]  # Discard module.

        if k in model_dict and model_dict[k].size() == v.size(): # If matched (key name and size match)
            new_state_dict[k] = v # Store the pretrained weights of matched layers
            matched_layers.append(k) # Record matched layers
        else: # If unmatched
            discarded_layers.append(k) # Record discarded layers
     
    # Update the state_dict of the model
    model_dict.update(new_state_dict) # Update the state_dict of the model with pretrained weights of matched layers
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0: # If no layers matched
        warnings.warn(
            f"Unable to load {weight_path} (please manually check key names)"
        )
    else: # Print unmatched layers
        print(f"Successfully loaded pretrained weights from {weight_path}")
        if len(discarded_layers) > 0:
            print(
                f"Layers discarded due to key name or size mismatch: {discarded_layers}"
            )

def init_network_weights(model, init_type="normal", gain=0.02):
    """Initialize network weights.
    Args:
        model (nn.Module): Neural network model.
        init_type (str): Initialization type. Options include:
            - normal: Standard normal distribution
            - xavier: Xavier initialization
            - kaiming: Kaiming initialization
            - orthogonal: Orthogonal initialization
        gain (float): Scaling factor.
    """
    def _init_func(m):
        classname = m.__class__.__name__

        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

        elif classname.find("BatchNorm") != -1:
            nn.init.constant_(m.weight.data, 1.0)
            nn.init.constant_(m.bias.data, 0.0)

        elif classname.find("InstanceNorm") != -1:
            if m.weight is not None and m.bias is not None:
                nn.init.constant_(m.weight.data, 1.0)
                nn.init.constant_(m.bias.data, 0.0)

    model.apply(_init_func)


def transform_image(tfm_func, img0, K=1):
    """
    Apply K times tfm augmentation to an image and return the K different augmented results (not stacked K times).

    Args:
    - tfm_func (callable): Transform function.
    - img0 (PIL.Image): Original image.
    - K (int): Number of augmentations, generating K augmented results.

    Returns:
    - Augmented single image (if only one augmentation result) img_list[0] || List of augmented images img_list
    """
    img_list = []  # Initialize image list

    for k in range(K):  # Perform K repeated augmentations
        tfm_img = tfm_func(img0) # Apply transform to the original image
        img_list.append(tfm_img)

    # If multiple augmentations were performed, return the list of augmented images; otherwise, return a single augmented image
    return img_list[0] if len(img_list) == 1 else img_list  


def standard_image_transform(input_size, interp_mode):
    """
    Standard image preprocessing transformation pipeline.
    
    Args:
        - input_size (int): Size of the input image.
        - interp_mode (str): Interpolation mode | Options include: NEAREST, BILINEAR, BICUBIC
    Returns:
        - Compose: Composed image transformation.

    Main steps:
        - Maintain the aspect ratio of the image, resize the image to input_size, and use the specified interpolation mode
        - Center crop the image to input_size*input_size
        - Convert to RGB image
    """
    assert isinstance(input_size, int), "input_size must be an integer"
    
    interp_mode = getattr(InterpolationMode, interp_mode.upper(), InterpolationMode.BILINEAR) # Get interpolation mode
    
    def _image_to_rgb(image):
        return image.convert("RGB")
    
    return Compose([
        Resize(input_size, interpolation=interp_mode),
        CenterCrop(input_size),
        _image_to_rgb
    ])


def patch_jit_model(model, device="cuda"):
    """
    Fix device and dtype information for JIT models.
    This function fixes the device and data type information of a JIT model to ensure the model runs on the specified device.
    
    Main steps:
        1. Generate a node for the target device.
        2. Traverse the computation graph of the model and fix device information.
        3. If the target device is CPU, fix the data type to float32.
        4. Return the fixed model.
    
    Args:
        - model (torch.jit.ScriptModule): Model compiled with torch.jit.trace or torch.jit.script
        - device (str): Target device ("cuda" or "cpu")
    
    Returns:
        - torch.jit.ScriptModule: JIT model with fixed device and dtype
    """
    
    # ----Generate a node for the target device----
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def _node_get(node: torch._C.Node, key: str):
        """Get attribute value from a JIT computation graph node"""
        sel = node.kindOf(key)
        return getattr(node, sel)(key)

    def patch_device(module):
        """Fix device information in the JIT computation graph"""
        graphs = []
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)  # Handle forward1 variant

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(_node_get(node, "value")).startswith("cuda"):
                    node.copyAttributes(device_node)

    # ----Apply device fix to the model and its submodules----
    model.apply(patch_device)
    
    # × Only applicable to clip, replaced with below "Recursively traverse all submodules of the JIT model"
    # patch_device(model.encode_image)
    # patch_device(model.encode_text)
    
    # √ Recursively traverse all submodules of the JIT model
    for name, submodule in model.named_modules():
        patch_device(submodule)

    # ----If CPU, fix dtype to float32----
    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            """Fix dtype in the JIT computation graph (CPU only)"""
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []
                
            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype may be the second or third parameter of aten::to()
                        if _node_get(inputs[i].node(), "value") == 5:  # 5 represents float32
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        # Force the entire model's dtype to float32
        model.float()

    return model
