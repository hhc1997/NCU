import time
import wandb
import torch
import numpy as np
import torch.distributed as dist
import logging
from torch.cuda.amp import autocast
from src.ul_clip import Invert_Loss, ContrastiveLoss
from torch.nn.parallel import DistributedDataParallel as DDP


def switch_to_prompt_mode(model, options):
    """switch to train promt"""
    model = model.module
    for param in model.parameters():
        param.requires_grad = True
    for param in model.visual.parameters():
        param.requires_grad = False
    return DDP(model, device_ids=[options.device_ids[options.the_rank]])


def switch_to_clip_mode(model, options):
    """switch to train CLIP"""
    model = model.module
    # only train part of the model
    for name, param in model.named_parameters():
        if "prompt" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    model.token_embedding.requires_grad = False
    model.positional_embedding.requires_grad = False
    for name, param in model.transformer.named_parameters():
        if "resblocks." in name and int(name.split('.')[1]) <= 4:
            param.requires_grad = False  # freeze for accurating
        else:
            param.requires_grad = True
    return DDP(model, device_ids=[options.device_ids[options.the_rank]])


def train_one_epoch(epoch, model, processor, data, opt_p, opt_c, scheduler_p, scheduler_c, scaler_p, scaler_c, options):
    dataloader = data["train"]
    # if (options.distributed): dataloader.sampler.set_epoch(epoch)
    model.train()
    Invert_criterion = Invert_Loss().to(options.device)
    Contrast_criterion = ContrastiveLoss().to(options.device)

    #modulo = max(1, int(dataloader.samples_per_rank / options.batch_size / 10))
    modulo = max(1, int(dataloader.num_samples / options.batch_size / 10))

    start = time.time()
    logging.info(f"Num samples: {dataloader.num_samples}, Num_batches: {dataloader.num_batches}") if (
        options.master) else None
    if epoch <= options.epochs_HN:
        ######### set prompt to be train #####################
        model = switch_to_prompt_mode(model, options)
        umodel = model.module if (options.distributed) else model
        for index, batch in enumerate(dataloader):
            step = dataloader.num_batches * epoch + index
            raw_captions, pixel_values = batch["raw_caps"], batch["pixel_values"].to(options.device, non_blocking=True)
            ########## train prompt by invert loss################
            scheduler_p(step)
            opt_p.zero_grad()
            with autocast():
                outputs = model(image=pixel_values, raw_caps=raw_captions, processor=processor, device=options.device)
                batched_NC_ids, batched_CC_ids, cyclic_loss, inmodal_opposite_loss, cross_sig_y, cross_sig_n = Invert_criterion(
                    outputs, options, umodel.logit_scale)
                loss_p = cyclic_loss + inmodal_opposite_loss + cross_sig_y + cross_sig_n
            scaler_p.scale(loss_p).backward()
            scaler_p.step(opt_p)
            scaler_p.update()
            loss_c = torch.tensor(0.0)

            end = time.time()
            if (options.master and (((index + 1) % modulo == 0) or (index == dataloader.num_batches - 1))):
                num_samples = (index + 1) * len(raw_captions) * options.num_devices
                #dataloader_num_samples = dataloader.samples_per_rank
                dataloader_num_samples = dataloader.num_samples
                logging.info(
                    f"Train Epoch: {epoch:02d} "
                    f"[{num_samples}/{dataloader_num_samples} "
                    f"({100.0 * (index + 1) / dataloader.num_batches:.0f}%)] "
                    f"\tloss_cy: {cyclic_loss.item():.4f} "
                    f"\tloss_io: {inmodal_opposite_loss.item():.4f} "
                    f"\tloss_coy: {cross_sig_y.item():.4f} "
                    f"\tloss_con: {cross_sig_n.item():.4f} "
                    f"\tloss_c: {loss_c.item():.4f} "
                    f"\tTime taken {end - start:.3f} "
                ) if options.master else None

                metrics = {"loss_p": loss_p.item(),
                           "loss_c": loss_c.item(),
                           "time": end - start,
                           "lr_p": opt_p.param_groups[0]["lr"],
                           "lr_c": opt_c.param_groups[0]["lr"]}
                if (options.wandb):
                    for key, value in metrics.items():
                        wandb.log({f"train/{key}": value, "step": step})

                start = time.time()
            # torch.distributed.barrier()
    else:
        ######### set CLIP to be train #####################
        model = switch_to_clip_mode(model, options)
        umodel = model.module if (options.distributed) else model
        for index, batch in enumerate(dataloader):
            step = dataloader.num_batches * epoch + index
            raw_captions, pixel_values = batch["raw_caps"], batch["pixel_values"].to(options.device, non_blocking=True)
            ########## train clip by contrast loss################
            scheduler_c(step)
            opt_c.zero_grad()
            with autocast():
                outputs = model(image=pixel_values, raw_caps=raw_captions, processor=processor, device=options.device)
                loss_c, inmodal_opposite_loss = Contrast_criterion(outputs, options, umodel.logit_scale)
                loss = loss_c + inmodal_opposite_loss
            scaler_c.scale(loss).backward()
            scaler_c.step(opt_c)
            scaler_c.update()
            umodel.logit_scale.data = torch.clamp(umodel.logit_scale.data, 0, 4.6052)
            ############# end for clip ###########################

            cyclic_loss = torch.tensor(0.0)
            cross_sig_y = torch.tensor(0.0)
            cross_sig_n = torch.tensor(0.0)
            loss_p = torch.tensor(0.0)

            end = time.time()
            if (options.master and (((index + 1) % modulo == 0) or (index == dataloader.num_batches - 1))):
                num_samples = (index + 1) * len(raw_captions) * options.num_devices
                #dataloader_num_samples = dataloader.samples_per_rank
                dataloader_num_samples = dataloader.num_samples
                logging.info(
                    f"Train Epoch: {epoch:02d} "
                    f"[{num_samples}/{dataloader_num_samples} "
                    f"({100.0 * (index + 1) / dataloader.num_batches:.0f}%)] "
                    f"\tloss_cy: {cyclic_loss.item():.4f} "
                    f"\tloss_io: {inmodal_opposite_loss.item():.4f} "
                    f"\tloss_coy: {cross_sig_y.item():.4f} "
                    f"\tloss_con: {cross_sig_n.item():.4f} "
                    f"\tloss_c: {loss_c.item():.4f} "
                    f"\tTime taken {end - start:.3f} "
                ) if options.master else None

                metrics = {"loss_p": loss_p.item(),
                           "loss_c": loss_c.item(),
                           "time": end - start,
                           "lr_p": opt_p.param_groups[0]["lr"],
                           "lr_c": opt_c.param_groups[0]["lr"]}
                if (options.wandb):
                    for key, value in metrics.items():
                        wandb.log({f"train/{key}": value, "step": step})

                start = time.time()

