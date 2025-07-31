import os
os.environ["WANDB_SILENT"] = "true"
import time
import wandb
import torch
import logging
import warnings
import numpy as np
import random
from collections import OrderedDict
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from pkgs.openai.clip import load as load_model
from pkgs.openai.clip import load_processor
from pkgs.LaCLIP import models
from src.train import train_one_epoch
from src.evaluate import evaluate
from src.data import load as load_data
from src.parser import parse_args
from src.scheduler import cosine_scheduler
from src.logger import get_logger, set_logger

mp.set_start_method("spawn", force=True)
warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", message="torch.distributed.reduce_op is deprecated")
def seed_everything(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 这两行设置可以提高训练速度，但可能会导致结果略有不同
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

def get_prompt_parameters(model, options):
    model = model.module if (options.distributed) else model
    """获取需要在 prompt 训练阶段更新的参数"""
    return [p for name, p in model.named_parameters() if "prompt" in name]


def get_clip_parameters(model, options):
    model = model.module if (options.distributed) else model
    for name, param in model.named_parameters():
        if "prompt" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    weight_decay_parameters = []
    no_weight_decay_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if all(key not in name for key in ["bn", "ln", "bias", "scale"]):
            weight_decay_parameters.append(parameter)
        else:
            no_weight_decay_parameters.append(parameter)

    return weight_decay_parameters, no_weight_decay_parameters

def worker(rank, options, logger):
    options.the_rank = rank
    options.master = rank == 0

    set_logger(the_rank=rank, logger=logger, distributed=options.distributed)

    if (options.device == "cuda"):
        options.device += ":" + str(options.device_ids[options.the_rank] if options.distributed else options.device_id)

    logging.info(f"Using {options.device_ids} device") if (options.master) else None

    if (options.master):
        logging.info("Params:")
        with open(os.path.join(options.log_dir_path, "params.txt"), "w") as file:
            for key in sorted(vars(options)):
                value = getattr(options, key)
                logging.info(f"{key}: {value}")
                file.write(f"{key}: {value}\n")

    if (options.distributed):
        dist.init_process_group(backend=options.distributed_backend, init_method=options.distributed_init_method,
                                world_size=options.num_devices, rank=options.the_rank)

    options.batch_size = options.batch_size // options.num_devices

    #########################################################

    model = getattr(models, options.model_name)(rand_embed=False)
    processor = load_processor(model)

    #########################################################

    torch.cuda.set_device(options.device_ids[options.the_rank] if options.distributed else options.device_id)
    model.to(options.device)


    data = load_data(options, processor)

    start_epoch = 0
    if (dist.get_rank() == 0 and options.checkpoint is not None):
        if (os.path.isfile(options.checkpoint)):

            ckpt_path = options.checkpoint
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                state_dict[k.replace('module.', '')] = v
            model.cuda()
            model.load_state_dict(state_dict, strict=False)

            # checkpoint = torch.load(options.checkpoint, map_location=options.device)
            # # start_epoch = checkpoint["epoch"]
            # state_dict = checkpoint["state_dict"]
            # #if (not options.distributed and next(iter(state_dict.items()))[0].startswith("module")):
            # state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
            # model.load_state_dict(state_dict, strict=False)
            # if(optimizer is not None): optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info(f"logit scale '{model.logit_scale.item()}'")
            logging.info(f"Loaded checkpoint '{ckpt_path}'")
        else:
            raise Exception(f"Required checkpoint not found at {options.checkpoint}")

    if (options.distributed):
        model = DDP(model, device_ids=[options.device_ids[options.the_rank]])


    cudnn.benchmark = True
    cudnn.deterministic = False

    optimizer_prompt = None
    optimizer_clip = None
    scheduler_prompt = None
    scheduler_clip = None
    if (data["train"] is not None):
        prompt_params = get_prompt_parameters(model, options)
        weight_decay_clip_params, no_weight_decay_clip_params = get_clip_parameters(model, options)


        optimizer_clip = optim.AdamW([
            {"params": no_weight_decay_clip_params, "weight_decay": 0},
            {"params": weight_decay_clip_params, "weight_decay": options.weight_decay}
        ], lr=options.lr_UL, betas=(options.beta1, options.beta2), eps=options.eps)
        scheduler_clip = cosine_scheduler(optimizer_clip, options.lr_UL, options.num_warmup_steps, data["train"].num_batches * options.epochs)


        optimizer_prompt = optim.AdamW([
            {"params": no_weight_decay_clip_params, "lr": options.lr_pre, "weight_decay": 0},
            {"params": weight_decay_clip_params, "lr": options.lr_pre, "weight_decay": options.weight_decay},
            {"params": prompt_params, "lr": options.lr_HN,  "weight_decay": options.weight_decay_prompt}
        ], betas=(options.beta1, options.beta2), eps=options.eps)
        scheduler_prompt = cosine_scheduler(optimizer_prompt, options.lr_HN, options.num_warmup_steps, data["train"].num_batches * options.epochs)

    if (options.wandb and options.master):
        logging.debug("Starting wandb")
        wandb.init(project="mrl", notes=options.notes, tags=[], config=vars(options))
        wandb.run.name = options.name
        wandb.save(os.path.join(options.log_dir_path, "params.txt"))

    # before unlearning, evaluate the pre-trained model performance
    #evaluate(start_epoch, model, processor, data, options)

    if (data["train"] is not None):
        options.checkpoints_dir_path = os.path.join(options.log_dir_path, "checkpoints")
        os.makedirs(options.checkpoints_dir_path, exist_ok=True)
        scaler_prompt = GradScaler()
        scaler_clip = GradScaler()

        best_loss = np.inf
        for epoch in range(start_epoch + 1, options.epochs + 1):
            if (options.master):
                logging.info(f"Starting Epoch {epoch}")

            start = time.time()
            train_one_epoch(epoch, model, processor, data, optimizer_prompt, optimizer_clip,
                            scheduler_prompt, scheduler_clip, scaler_prompt, scaler_clip, options)
            end = time.time()

            if (options.master):
                logging.info(f"Finished Epoch {epoch}, Time Taken: {end - start:.3f}")

            metrics = evaluate(epoch, model, processor, data, options)


            if (options.master):
                checkpoint = {"epoch": epoch, "name": options.name, "state_dict": model.state_dict(),
                              "optimizer": optimizer_clip.state_dict()}
                torch.save(checkpoint, os.path.join(options.checkpoints_dir_path, f"epoch_{epoch}.pt"))
                if ("loss" in metrics):
                    if (metrics["loss"] < best_loss):
                        best_loss = metrics["loss"]
                        torch.save(checkpoint, os.path.join(options.checkpoints_dir_path, f"epoch.best.pt"))

    if (options.distributed):
        dist.destroy_process_group()

    if (options.wandb and options.master):
        wandb.finish()


if (__name__ == "__main__"):
    options = parse_args()
    options.log_dir_path = os.path.join(options.logs, options.name)
    options.log_file_path = os.path.join(options.log_dir_path, "output.log")

    os.makedirs(options.log_dir_path, exist_ok=True)
    logger, listener = get_logger(options.log_file_path)

    listener.start()

    ngpus = torch.cuda.device_count()
    if (ngpus == 0 or options.device == "cpu"):
        options.device = "cpu"
        options.num_devices = 1
        options.distributed = False
        worker(0, options, logger)
    else:
        if (ngpus == 1 or not options.distributed):
            options.device = "cuda"
            options.num_devices = 1
            options.distributed = False
            worker(0, options, logger)
        else:
            options.device = "cuda"
            if (options.device_ids is None):
                options.device_ids = list(range(ngpus))
                options.num_devices = ngpus
            else:
                options.device_ids = list(map(int, options.device_ids))
                options.num_devices = len(options.device_ids)
            options.distributed = True
            os.environ["NCCL_P2P_DISABLE"] = "1"
            mp.spawn(worker, nprocs=options.num_devices, args=(options, logger))

    listener.stop()
