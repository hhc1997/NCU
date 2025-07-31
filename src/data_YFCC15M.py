import os
import torch
import logging
import torchvision
import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler

import re
import argparse
from datasets import load_dataset
from dataclasses import dataclass
from pkgs.DeCLIP.tokenizer import SimpleTokenizer
from pkgs.LaCLIP.tokenizer import SimpleTokenizer as SimpleTokenizer_origi

ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None

    def set_epoch(self, epoch):
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


class ImageRawCaptionDataset(Dataset):
    def __init__(self, path, image_key, caption_key, delimiter, processor):
        logging.debug(f"Loading aligned data from {path}")

        df = pd.read_csv(path, sep=delimiter)

        self.root = os.path.dirname(path)
        self.images = df[image_key].tolist()
        self.captions = df[caption_key].tolist()
        self.processor = processor
        # self.all_NC_idxs = None
        self.all_NC_idxs = torch.ones(len(self.images))  # init as all corrected pairs 1 for correct and 0 for NC
        logging.debug("Loaded data")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = {}
        # if self.all_NC_idxs is not None:
        item["noisy_pairs"] = self.all_NC_idxs[idx]
        item["sample_idxs"] = idx
        item["raw_caps"] = self.captions[idx]
        item["pixel_values"] = self.processor.process_image(Image.open(os.path.join(self.root, self.images[idx])),
                                                            train_mode=True)
        return item


class YFCC15MDataset(IterableDataset):
    def __init__(self, transforms, tokenizer=None):

        dataset = load_dataset(
            "/mnt/peijun/YFCC15M",
            trust_remote_code=True,
            streaming=True
        )
        self.dataset = dataset['train']
        self.transforms = transforms
        self.tokenizer_original = SimpleTokenizer_origi()
        self.tokenizer = tokenizer
        # 添加分布式训练所需的属性
        self.rank = 0
        self.world_size = 1
        if torch.distributed.is_initialized():
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        # 计算每个rank的样本数
        self.total_samples = 15060992
        self.samples_per_rank = self.total_samples // self.world_size

        self.dataset = self.dataset.shard(
            num_shards=self.world_size,
            index = self.rank
        )

        self.dataset = self.dataset.take(self.samples_per_rank)
        self.pattern = re.compile(r"<\|startoftext\|>(.*?)<\|endoftext\|>")

    def __len__(self):
        return self.samples_per_rank

    def __iter__(self):
        # 对数据集进行分片
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # 单worker情况
            for item in self.dataset:
                data_item = {}
                data_item["raw_caps"] = self._process_text(item['texts'])
                data_item["pixel_values"] = self.transforms(item['images'])
                yield data_item
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            per_worker = self.dataset.shard(
                num_shards=num_workers,
                index=worker_id
            )
            for item in per_worker:
                data_item = {}
                data_item["raw_caps"] = self._process_text(item['texts'])
                data_item["pixel_values"] = self.transforms(item['images'])
                yield data_item
    def _process_text(self, texts):
        """文本处理函数保持不变"""
        texts = self.tokenizer_original.decode(texts)
        match = self.pattern.search(texts)
        texts = match.group(1).strip()
        return texts




def get_YFCC15M_dataset(args, preprocess_fn, is_train, tokenizer=None):
    dataset = YFCC15MDataset(preprocess_fn, tokenizer)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers = True,
        drop_last=is_train
    )

    # num_samples应该是整个数据集的大小
    dataloader.num_samples = len(dataset)
    # num_batches是每个rank处理的batch数量
    samples_per_rank = len(dataset) // dataset.world_size
    dataloader.samples_per_rank = samples_per_rank
    dataloader.num_batches = samples_per_rank // args.batch_size

    return DataInfo(dataloader, None)


def get_YFCC15M_dataloader(args, preprocess_fns, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {
        "train": get_YFCC15M_dataset(
            args, preprocess_train, is_train=True, tokenizer=tokenizer
        )
    }
    return data


def get_train_dataloader(options, processor):
    path = options.train_data
    if (path is None): return None

    batch_size = options.batch_size

    dataset = ImageRawCaptionDataset(path, image_key=options.image_key, caption_key=options.caption_key,
                                     delimiter=options.delimiter, processor=processor)

    sampler = DistributedSampler(dataset) if (options.distributed) else None

    dataloade_CLIP = DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None),
                                num_workers=options.num_workers, pin_memory=True, sampler=sampler, drop_last=True,
                                prefetch_factor=2)

    dataloade_CLIP.num_samples = len(dataloade_CLIP) * batch_size
    dataloade_CLIP.num_batches = len(dataloade_CLIP)

    return dataloade_CLIP


def get_validation_dataloader(options, processor):
    path = options.validation_data
    if (path is None): return

    dataset = ImageRawCaptionDataset(path, image_key=options.image_key, caption_key=options.caption_key,
                                     delimiter=options.delimiter, processor=processor)
    dataloader = DataLoader(dataset, batch_size=options.batch_size, shuffle=False, num_workers=options.num_workers,
                            pin_memory=True, sampler=None, drop_last=True)
    dataloader.num_samples = len(dataset)
    dataloader.num_batches = len(dataloader)

    return dataloader


class ImageLabelDataset(Dataset):
    def __init__(self, root, transform):
        self.root = root
        df = pd.read_csv(os.path.join(root, "labels.csv"))
        self.images = df["image"]
        self.labels = df["label"]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.transform(Image.open(os.path.join(self.root, self.images[idx])))
        label = self.labels[idx]
        return image, label


def get_eval_test_dataloader(options, processor):
    if (options.eval_test_data_dir is None): return

    if (options.eval_data_type == "Caltech101"):
        dataset = ImageLabelDataset(root=options.eval_test_data_dir, transform=processor.process_image)
        # dataset = torchvision.datasets.Caltech101(root=os.path.dirname(options.eval_test_data_dir), download=True, transform=processor.process_image)
    elif (options.eval_data_type == "CIFAR10"):
        dataset = torchvision.datasets.CIFAR10(root=os.path.dirname(options.eval_test_data_dir), download=True,
                                               train=False, transform=processor.process_image)
    elif (options.eval_data_type == "CIFAR100"):
        dataset = torchvision.datasets.CIFAR100(root=os.path.dirname(options.eval_test_data_dir), download=True,
                                                train=False, transform=processor.process_image)
    elif (options.eval_data_type == "DTD"):
        dataset = torchvision.datasets.DTD(root=os.path.dirname(options.eval_test_data_dir), download=True,
                                           split="test", transform=processor.process_image)
    elif (options.eval_data_type == "FGVCAircraft"):
        dataset = torchvision.datasets.FGVCAircraft(root=os.path.dirname(options.eval_test_data_dir), download=True,
                                                    split="test", transform=processor.process_image)
    elif (options.eval_data_type == "Flowers102"):
        dataset = ImageLabelDataset(root=options.eval_test_data_dir, transform=processor.process_image)
    elif (options.eval_data_type == "Food101"):
        dataset = torchvision.datasets.Food101(root=os.path.dirname(options.eval_test_data_dir), download=True,
                                               split="test", transform=processor.process_image)
    elif (options.eval_data_type == "GTSRB"):
        dataset = torchvision.datasets.GTSRB(root=os.path.dirname(options.eval_test_data_dir), download=True,
                                             split="test", transform=processor.process_image)
    elif (options.eval_data_type == "ImageNet1K"):
        dataset = ImageLabelDataset(root=options.eval_test_data_dir, transform=processor.process_image)
    elif (options.eval_data_type == "OxfordIIITPet"):
        dataset = torchvision.datasets.OxfordIIITPet(root=os.path.dirname(options.eval_test_data_dir), download=True,
                                                     split="test", transform=processor.process_image)
    elif (options.eval_data_type == "RenderedSST2"):
        dataset = torchvision.datasets.RenderedSST2(root=os.path.dirname(options.eval_test_data_dir), download=True,
                                                    split="test", transform=processor.process_image)
    elif (options.eval_data_type == "StanfordCars"):
        dataset = torchvision.datasets.StanfordCars(root=os.path.dirname(options.eval_test_data_dir), download=False,
                                                    split="test", transform=processor.process_image)
    elif (options.eval_data_type == "STL10"):
        dataset = torchvision.datasets.STL10(root=os.path.dirname(options.eval_test_data_dir), download=True,
                                             split="test", transform=processor.process_image)
    elif (options.eval_data_type == "SVHN"):
        dataset = torchvision.datasets.SVHN(root=os.path.dirname(options.eval_test_data_dir), download=True,
                                            split="test", transform=processor.process_image)
    elif (options.eval_data_type == "EuroSAT"):
        dataset = ImageLabelDataset(root=options.eval_test_data_dir, transform=processor.process_image)
    elif (options.eval_data_type == "RESISC45"):
        dataset = ImageLabelDataset(root=options.eval_test_data_dir, transform=processor.process_image)
    elif (options.eval_data_type == "SUN397"):
        dataset = ImageLabelDataset(root=options.eval_test_data_dir, transform=processor.process_image)
    elif (options.eval_data_type in ["ImageNetSketch", "ImageNetV2", "ImageNet-A", "ImageNet-R"]):
        dataset = ImageLabelDataset(root=options.eval_test_data_dir, transform=processor.process_image)
    else:
        raise Exception(f"Eval test dataset type {options.eval_data_type} is not supported")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=options.batch_size, num_workers=options.num_workers,
                                             sampler=None, pin_memory=True,
                                             prefetch_factor=4)
    dataloader.num_samples = len(dataset)
    dataloader.num_batches = len(dataloader)

    return dataloader


def get_eval_train_dataloader(options, processor):
    if (not options.linear_probe or options.eval_train_data_dir is None): return

    if (options.eval_data_type == "Caltech101"):
        dataset = ImageLabelDataset(root=options.eval_train_data_dir, transform=processor.process_image)
    elif (options.eval_data_type == "CIFAR10"):
        dataset = torchvision.datasets.CIFAR10(root=os.path.dirname(options.eval_train_data_dir), download=True,
                                               train=True, transform=processor.process_image)
    elif (options.eval_data_type == "CIFAR100"):
        dataset = torchvision.datasets.CIFAR100(root=os.path.dirname(options.eval_train_data_dir), download=True,
                                                train=True, transform=processor.process_image)
    elif (options.eval_data_type == "DTD"):
        dataset = torch.utils.data.ConcatDataset([torchvision.datasets.DTD(
            root=os.path.dirname(options.eval_train_data_dir), download=True, split="train",
            transform=processor.process_image), torchvision.datasets.DTD(
            root=os.path.dirname(options.eval_train_data_dir), download=True, split="val",
            transform=processor.process_image)])
    elif (options.eval_data_type == "FGVCAircraft"):
        dataset = torchvision.datasets.FGVCAircraft(root=os.path.dirname(options.eval_train_data_dir), download=True,
                                                    split="trainval", transform=processor.process_image)
    elif (options.eval_data_type == "Flowers102"):
        dataset = ImageLabelDataset(root=options.eval_train_data_dir, transform=processor.process_image)
    elif (options.eval_data_type == "Food101"):
        dataset = torchvision.datasets.Food101(root=os.path.dirname(options.eval_train_data_dir), download=True,
                                               split="train", transform=processor.process_image)
    elif (options.eval_data_type == "GTSRB"):
        dataset = torchvision.datasets.GTSRB(root=os.path.dirname(options.eval_train_data_dir), download=True,
                                             split="train", transform=processor.process_image)
    elif (options.eval_data_type == "ImageNet1K"):
        dataset = ImageLabelDataset(root=options.eval_train_data_dir, transform=processor.process_image)
    elif (options.eval_data_type == "OxfordIIITPet"):
        dataset = torchvision.datasets.OxfordIIITPet(root=os.path.dirname(options.eval_train_data_dir), download=True,
                                                     split="trainval", transform=processor.process_image)
    elif (options.eval_data_type == "RenderedSST2"):
        dataset = torchvision.datasets.RenderedSST2(root=os.path.dirname(options.eval_train_data_dir), download=True,
                                                    split="train", transform=processor.process_image)
    elif (options.eval_data_type == "StanfordCars"):
        dataset = torchvision.datasets.StanfordCars(root=os.path.dirname(options.eval_train_data_dir), download=True,
                                                    split="train", transform=processor.process_image)
    elif (options.eval_data_type == "STL10"):
        dataset = torchvision.datasets.STL10(root=os.path.dirname(options.eval_train_data_dir), download=True,
                                             split="train", transform=processor.process_image)
    elif (options.eval_data_type == "SVHN"):
        dataset = torchvision.datasets.SVHN(root=os.path.dirname(options.eval_train_data_dir), download=True,
                                            split="train", transform=processor.process_image)
    elif (options.eval_data_type == "SUN397"):
        dataset = ImageLabelDataset(root=options.eval_train_data_dir, transform=processor.process_image)
    else:
        raise Exception(f"Eval train dataset type {options.eval_data_type} is not supported")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=options.linear_probe_batch_size,
                                             num_workers=options.num_workers, sampler=None)
    dataloader.num_samples = len(dataset)
    dataloader.num_batches = len(dataloader)

    return dataloader


def load(options, processor):
    data = {}
    ########### For YFCC15M #################
    from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop, Lambda
    transform_tain = Compose([
        RandomResizedCrop(224, scale=(0.5, 1.0)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
        # Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
        #           std=[0.26862954, 0.26130258, 0.27577711])
    ])
    args_YFCC = argparse.Namespace(batch_size=options.batch_size, workers=options.num_workers)
    #data["train"] = get_YFCC15M_dataloader(args_YFCC, [transform_tain, None], SimpleTokenizer())['train'].dataloader
    data["train"] = None
    #########################################
    # data["train"] = get_train_dataloader(options, processor)
    #data["validation"] = get_validation_dataloader(options, processor)
    data["eval_test"] = get_eval_test_dataloader(options, processor)
    data["eval_train"] = get_eval_train_dataloader(options, processor)

    return data