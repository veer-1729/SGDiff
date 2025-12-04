import os
from datetime import datetime
from torch import optim
from torch.cuda.amp import GradScaler
import torch.utils.tensorboard as tensorboard
from sgEncoderTraining.sgEncoder.create_sg_encoder import create_model_and_transforms
from sgEncoderTraining.training.logger import setup_logging
from configs.configs_laion import parse_args
from sgEncoderTraining.training.scheduler import cosine_lr
from sgEncoderTraining.training.train_and_val_one_iter import train_by_iters
from sgEncoderTraining.datasets.laion_dataset import build_laion_loaders
import logging
import os
import torch
import torch.utils.checkpoint
from transformers import AutoTokenizer, PretrainedConfig

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.utils.import_utils import is_xformers_available
from accelerate import Accelerator, DistributedDataParallelKwargs

accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str,  subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")



def trainer():
    args = parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


    if args.name is None:
        args.name = '-'.join([
            datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"j_{args.workers}",
            f"p_{args.precision}",
        ])


    args.log_path = None
    if accelerator.is_main_process:
        log_base_path = os.path.join(args.logs, args.name)
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path):
            print(
                "Error. Experiment already exists. Use --name {} to specify a new experiment."
            )
            return -1

    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    if accelerator.is_main_process:
        args.tensorboard_path = os.path.join(args.logs, args.name, "tensorboard") if args.tensorboard else ''
        args.checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''
        args.checkpoint_path = ''

    assert args.precision in ['amp', 'amp_bfloat16', 'fp16', 'fp32']
    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')


    torch.manual_seed(args.seed)

    train_dataloader, val_dataloader = build_laion_loaders(args)


    tokenizer_one = AutoTokenizer.from_pretrained(
        args.stable_diffusion_checkpoint,
        subfolder="tokenizer",
        use_fast=False,
        cache_dir = args.cache_dir
    )

    tokenizer_two = AutoTokenizer.from_pretrained(
        args.stable_diffusion_checkpoint,
        subfolder="tokenizer_2",
        use_fast=False,
        cache_dir=args.cache_dir
    )

    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.stable_diffusion_checkpoint
    )

    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.stable_diffusion_checkpoint, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.stable_diffusion_checkpoint, subfolder="scheduler",cache_dir = args.cache_dir)

    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.stable_diffusion_checkpoint, subfolder="text_encoder", variant="fp16",cache_dir = args.cache_dir).to(accelerator.device)
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.stable_diffusion_checkpoint, subfolder="text_encoder_2", variant="fp16",cache_dir = args.cache_dir).to(accelerator.device)

    vae = AutoencoderKL.from_pretrained(
        args.stable_diffusion_checkpoint,
        subfolder="vae",
        variant="fp16",
        cache_dir=args.cache_dir
    ).to(accelerator.device)

    unet = UNet2DConditionModel.from_pretrained(
        args.stable_diffusion_checkpoint, subfolder="unet", variant="fp16",cache_dir = args.cache_dir
    ).to(accelerator.device)

    # enable xformers + gradient checkpointing to reduce VRAM
    if is_xformers_available():
        try:
            unet.enable_xformers_memory_efficient_attention()
            logging.info("Enabled xFormers memory efficient attention.")
        except Exception as e:
            logging.warning("Failed to enable xFormers attention: %s", e)
    else:
        logging.info("xFormers not available; install xformers for lower VRAM use.")

    precision_lower = args.precision.lower()
    if precision_lower == "fp16":
        # FP16 behaves fine on all CUDA archs
        autocast_dtype = torch.float16
    else:
        # default to bfloat16 only on A100+; fall back to fp16 on older GPUs
        capability = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
        if capability >= (8, 0):
            autocast_dtype = torch.bfloat16 if precision_lower in {"amp", "amp_bfloat16", "bf16"} else torch.float16
        else:
            if precision_lower in {"amp", "amp_bfloat16", "bf16"}:
                logging.warning(
                    "Requested %s but GPU capability %s lacks native bf16; falling back to fp16.",
                    args.precision,
                    capability,
                )
            autocast_dtype = torch.float16

    args.autocast_dtype = autocast_dtype
    try:
        unet.enable_gradient_checkpointing()
        logging.info("Enabled UNet gradient checkpointing.")
    except Exception as e:
        logging.warning("Failed to enable gradient checkpointing: %s", e)

    # We only train the additional adapter SGencoder layers
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)

    model = create_model_and_transforms(
        args,
        text_encoders=[text_encoder_one, text_encoder_two],
        tokenizers =[tokenizer_one,tokenizer_two],
        model_config_json=args.model_config_json,
        precision=args.precision,
        device=accelerator.device,
        force_quick_gelu=args.force_quick_gelu,
        pretrained_image=args.pretrained_image,
    ).to(accelerator.device)

    del text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two

    if args.pretrained_sgencoder_path:
        if os.path.isfile(args.pretrained_sgencoder_path):
            logging.info(f"Loading sgEncoder weights from {args.pretrained_sgencoder_path}")
            checkpoint = torch.load(args.pretrained_sgencoder_path, map_location=accelerator.device)
            state_dict = checkpoint.get("state_dict", checkpoint)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                logging.warning("Missing keys: %s", missing)
                logging.warning("Unexpected keys: %s", unexpected)
        else:
            logging.warning("Pretrained sgEncoder path %s not found.", args.pretrained_sgencoder_path)

    torch.manual_seed(args.seed)
    if accelerator.is_main_process:
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")


    optimizer = None
    scaler = None
    if args.image_dir:

        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
        include = lambda n, p: not exclude(n, p)

        named_parameters = list(model.named_parameters())
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

        optimizer = optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": args.wd},
            ],
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )


        scaler = GradScaler() if (args.precision == "amp" or args.precision == "amp_bfloat16")else None


    #start_epoch = checkpoint.get('epoch', 0)
    start_epoch = 0

    total_steps = len(train_dataloader) * args.epochs
    scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)

    args.save_logs = args.logs and args.logs.lower() != 'none' and accelerator.is_main_process
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

        logging.debug('Finished loading wandb.')


    model, train_dataloader, val_dataloader,optimizer, scaler, scheduler, vae, unet, noise_scheduler = accelerator.prepare(
        model, train_dataloader, val_dataloader, optimizer, scaler, scheduler, vae, unet, noise_scheduler
    )

    for epoch in range(start_epoch, args.epochs):
        if accelerator.is_main_process:
            logging.info(f'Start epoch {epoch}')

        train_by_iters(model, train_dataloader, val_dataloader, epoch, optimizer, scaler, scheduler, args,
                       vae,
                       unet,
                       noise_scheduler,
                       accelerator,
                       writer, val_count=args.val_times_per_epoch)

if __name__ == "__main__":
    trainer()
