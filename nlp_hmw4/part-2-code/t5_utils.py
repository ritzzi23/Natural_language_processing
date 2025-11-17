import os

import torch

import transformers
from transformers import T5ForConditionalGeneration, T5Config
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
import wandb

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def setup_wandb(args):
    '''
    Initialize wandb for experiment tracking.
    '''
    wandb.init(
        project="t5-text-to-sql",
        name=args.experiment_name,
        config={
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "max_n_epochs": args.max_n_epochs,
            "finetune": args.finetune,
            "optimizer_type": args.optimizer_type,
            "scheduler_type": args.scheduler_type,
            "weight_decay": args.weight_decay,
        }
    )

def initialize_model(args):
    '''
    Helper function to initialize the model. You should be either finetuning
    the pretrained model associated with the 'google-t5/t5-small' checkpoint
    or training a T5 model initialized with the 'google-t5/t5-small' config
    from scratch.
    
    Args:
        args: Arguments object with 'finetune' flag
        
    Returns:
        model: T5ForConditionalGeneration model
    '''
    # Debug: Check if finetune flag is being received
    print(f"DEBUG: args.finetune = {args.finetune}")
    
    model_name = 'google-t5/t5-small'
    
    if args.finetune:
        # Fine-tune pretrained T5 model
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        print(f"Loaded pretrained model: {model_name}")
    else:
        # Train from scratch (random initialization)
        config = T5Config.from_pretrained(model_name)
        model = T5ForConditionalGeneration(config)
        print(f"Initialized T5 model from scratch with config from: {model_name}")
    
    # Move model to device
    model = model.to(DEVICE)
    
    # Optionally freeze some layers (experiment with this)
    # For now, we fine-tune the entire model
    # Example: To freeze encoder, uncomment:
    # for param in model.encoder.parameters():
    #     param.requires_grad = False
    
    return model

def mkdir(dirpath):
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except FileExistsError:
            pass

def save_model(checkpoint_dir, model, best):
    '''
    Save model checkpoint to be able to load the model later.
    
    Args:
        checkpoint_dir: Directory to save checkpoint
        model: Model to save
        best: If True, save as 'best_model.pt', else save as 'checkpoint.pt'
    '''
    mkdir(checkpoint_dir)
    
    if best:
        checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
    else:
        checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pt')
    
    # Save model state dict
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model.config,
    }, checkpoint_path)
    
    print(f"Saved model checkpoint to: {checkpoint_path}")

def load_model_from_checkpoint(args, best):
    '''
    Load model from a checkpoint.
    
    Args:
        args: Arguments object with 'checkpoint_dir' and 'finetune' flag
        best: If True, load 'best_model.pt', else load 'checkpoint.pt'
        
    Returns:
        model: Loaded T5ForConditionalGeneration model
    '''
    checkpoint_dir = args.checkpoint_dir
    
    if best:
        checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
    else:
        checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pt')
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    # weights_only=False is safe here since we're loading our own checkpoints
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    
    # Initialize model (same way as during training)
    model_name = 'google-t5/t5-small'
    if args.finetune:
        model = T5ForConditionalGeneration.from_pretrained(model_name)
    else:
        config = T5Config.from_pretrained(model_name)
        model = T5ForConditionalGeneration(config)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    
    print(f"Loaded model checkpoint from: {checkpoint_path}")
    return model

def initialize_optimizer_and_scheduler(args, model, epoch_length):
    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer, epoch_length)
    return optimizer, scheduler

def initialize_optimizer(args, model):
    decay_parameters = get_parameter_names(model, transformers.pytorch_utils.ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    if args.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8, betas=(0.9, 0.999)
        )
    else:
        pass

    return optimizer
        
def initialize_scheduler(args, optimizer, epoch_length):
    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs

    if args.scheduler_type == "none":
        return None
    elif args.scheduler_type == "cosine":
        return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif args.scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    else:
        raise NotImplementedError

def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

