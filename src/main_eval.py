import os

os.environ["WANDB_SILENT"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ[
    "TORCH_DISTRIBUTED_DEBUG"
] = "DETAIL"

import sys
import time
import wandb
import torch

import logging
import warnings
import numpy as np
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from pkgs.openai.clip import load as load_model

from src.train import train_prompt, train_CLIP
from src.evaluate import evaluate
from src.data import load as load_data
from src.parser import parse_args
from src.scheduler import cosine_scheduler
from src.logger import get_logger, set_logger

mp.set_start_method("spawn", force=True)
warnings.filterwarnings("ignore")


def worker(rank, options, logger):
    options.rank = rank
    options.master = rank == 0

    set_logger(rank=rank, logger=logger, distributed=options.distributed)

    if (options.device == "cuda"):
        options.device += ":" + str(options.device_ids[options.rank] if options.distributed else options.device_id)

    logging.info(f"Using {options.device} device")

    if (options.master):
        logging.info("Params:")
        with open(os.path.join(options.log_dir_path, "params.txt"), "w") as file:
            for key in sorted(vars(options)):
                value = getattr(options, key)
                logging.info(f"{key}: {value}")
                file.write(f"{key}: {value}\n")

    if (options.distributed):
        dist.init_process_group(backend=options.distributed_backend, init_method=options.distributed_init_method,
                                world_size=options.num_devices, rank=options.rank)

    options.batch_size = options.batch_size // options.num_devices

    model, processor = load_model(name=options.model_name, pretrained=options.pretrained)

    if (options.device == "cpu"):
        model.float()
    else:
        torch.cuda.set_device(options.device_ids[options.rank] if options.distributed else options.device_id)
        model.to(options.device)
        if (options.distributed):
            model = DDP(model, device_ids=[options.device_ids[options.rank]])

    data = load_data(options, processor)


    start_epoch = 0
    if (options.checkpoint is not None):
        if (os.path.isfile(options.checkpoint)):
            checkpoint = torch.load(options.checkpoint, map_location=options.device)
            # start_epoch = checkpoint["epoch"]
            state_dict = checkpoint["state_dict"]
            if (not options.distributed and next(iter(state_dict.items()))[0].startswith("module")):
                state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            # if(optimizer is not None): optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info(f"Loaded checkpoint '{options.checkpoint}' (start epoch {checkpoint['epoch']})")
        else:
            logging.info(f"No checkpoint found at {options.checkpoint}")

    cudnn.benchmark = True
    cudnn.deterministic = False

    if (options.wandb and options.master):
        logging.debug("Starting wandb")
        wandb.init(project="mrl", notes=options.notes, tags=[], config=vars(options))
        wandb.run.name = options.name
        wandb.save(os.path.join(options.log_dir_path, "params.txt"))

    # debug 注释 正式训练请启用
    evaluate(start_epoch, model, processor, data, options)



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