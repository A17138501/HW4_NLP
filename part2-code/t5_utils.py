import os
import torch
import transformers
from transformers import T5ForConditionalGeneration, T5Config

# 更稳的导入方式：避免部分版本没有 transformers.pytorch_utils 时报错
try:
    from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
except Exception:
    # 兜底，如果不行就直接用 tuple()，不做 weight decay 划分（不会报错，只是效果略有区别）
    ALL_LAYERNORM_LAYERS = ()

# scheduler 单独从 optimization 导入，更通用
from transformers.optimization import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

# wandb 可选导入，没装也不会报错
try:
    import wandb
except ImportError:
    wandb = None

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def setup_wandb(args):
    # 你如果想用 wandb，可以在这里写 wandb.init(...)
    # 现在先留空，保证不会出错
    if wandb is not None and args.use_wandb:
        # 示例（按需要自己改项目名等）
        # wandb.init(project="nlp_hw4_t5", name=args.experiment_name)
        pass


def initialize_model(args):
    """
    Initialize the T5 model.
    If --finetune is passed, load pretrained 't5-small';
    otherwise initialize from scratch using its config.
    """
    if args.finetune:
        # Load pretrained checkpoint
        model = T5ForConditionalGeneration.from_pretrained("t5-small")
    else:
        # Initialize from scratch
        config = T5Config.from_pretrained("t5-small")
        model = T5ForConditionalGeneration(config)

    model.to(DEVICE)
    return model


def mkdir(dirpath):
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except FileExistsError:
            pass


def save_model(checkpoint_dir, model, best):
    """
    Save model checkpoint.
    - best=True : save as best.pt
    - best=False: save as last.pt
    """
    mkdir(checkpoint_dir)
    fname = "best.pt" if best else "last.pt"
    path = os.path.join(checkpoint_dir, fname)

    # 保存 state_dict（最标准、最轻量）
    torch.save(model.state_dict(), path)


def load_model_from_checkpoint(args, best):
    """
    Load model from checkpoint.
    - best=True  -> best.pt
    - best=False -> last.pt
    This function must match save_model().
    """
    fname = "best.pt" if best else "last.pt"
    path = os.path.join(args.checkpoint_dir, fname)

    # 初始化一个新模型（需要与保存时一致）
    model = initialize_model(args)

    # 加载权重
    state_dict = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()
    return model


def initialize_optimizer_and_scheduler(args, model, epoch_length):
    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer, epoch_length)
    return optimizer, scheduler


def initialize_optimizer(args, model):
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
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
        raise NotImplementedError(f"Optimizer {args.optimizer_type} not supported")

    return optimizer


def initialize_scheduler(args, optimizer, epoch_length):
    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs

    if args.scheduler_type == "none":
        return None
    elif args.scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif args.scheduler_type == "linear":
        return get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
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


