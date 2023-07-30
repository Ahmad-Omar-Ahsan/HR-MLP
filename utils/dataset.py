import logging
import math
import os
import random

import numpy as np
import torch
import torch.utils.data as data
from timm.data.auto_augment import (
    augment_and_mix_transform,
    auto_augment_transform,
    rand_augment_transform,
)
from timm.data.constants import (
    DEFAULT_CROP_PCT,
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
)
from timm.data.random_erasing import RandomErasing
from timm.data.transforms import (
    RandomResizedCropAndInterpolation,
    ToNumpy,
    str_to_interp_mode,
    str_to_pil_interp,
)
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, datasets
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader, sampler, random_split

_logger = logging.getLogger(__name__)
_ERROR_RETRY = 50

CIFAR10_DEFAULT_MEAN = [0.485, 0.456, 0.406]
CIFAR10_DEFAULT_STD = [0.2023, 0.1994, 0.2010]

CIFAR100_DEFAULT_MEAN = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
CIFAR100_DEFAULT_STD = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]


def transforms_noaug_train(
    img_size=32,
    interpolation="bilinear",
    use_prefetcher=False,
    mean=CIFAR10_DEFAULT_MEAN,
    std=CIFAR10_DEFAULT_STD,
):
    if interpolation == "random":
        # random interpolation not supported with no-aug
        interpolation = "bilinear"
    tfl = [
        transforms.Resize(img_size, interpolation=str_to_interp_mode(interpolation)),
        transforms.CenterCrop(img_size),
    ]
    if use_prefetcher:
        # prefetcher and collate will handle tensor conversion and norm
        tfl += [ToNumpy()]
    else:
        tfl += [
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        ]
    return transforms.Compose(tfl)


def transforms_cifar10_train(
    img_size=32,
    scale=None,
    ratio=None,
    hflip=0.5,
    vflip=0.0,
    color_jitter=0.4,
    auto_augment=None,
    interpolation="random",
    use_prefetcher=False,
    mean=CIFAR10_DEFAULT_MEAN,
    std=CIFAR10_DEFAULT_STD,
    re_prob=0.0,
    re_mode="const",
    re_count=1,
    re_num_splits=0,
    separate=False,
):
    """
    If separate==True, the transforms are returned as a tuple of 3 separate transforms
    for use in a mixing dataset that passes
     * all data through the first (primary) transform, called the 'clean' data
     * a portion of the data through the secondary transform
     * normalizes and converts the branches above with the third, final transform
    """
    scale = tuple(scale or (0.08, 1.0))  # default imagenet scale range
    ratio = tuple(ratio or (3.0 / 4.0, 4.0 / 3.0))  # default imagenet ratio range
    primary_tfl = [
        RandomResizedCropAndInterpolation(
            img_size, scale=scale, ratio=ratio, interpolation=interpolation
        )
    ]
    if hflip > 0.0:
        primary_tfl += [transforms.RandomHorizontalFlip(p=hflip)]
    if vflip > 0.0:
        primary_tfl += [transforms.RandomVerticalFlip(p=vflip)]

    secondary_tfl = []
    if auto_augment:
        assert isinstance(auto_augment, str)
        if isinstance(img_size, (tuple, list)):
            img_size_min = min(img_size)
        else:
            img_size_min = img_size
        aa_params = dict(
            translate_const=int(img_size_min * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        if interpolation and interpolation != "random":
            aa_params["interpolation"] = str_to_pil_interp(interpolation)
        if auto_augment.startswith("rand"):
            secondary_tfl += [rand_augment_transform(auto_augment, aa_params)]
        elif auto_augment.startswith("augmix"):
            aa_params["translate_pct"] = 0.3
            secondary_tfl += [augment_and_mix_transform(auto_augment, aa_params)]
        else:
            secondary_tfl += [auto_augment_transform(auto_augment, aa_params)]
    elif color_jitter is not None:
        # color jitter is enabled when not using AA
        if isinstance(color_jitter, (list, tuple)):
            # color jitter should be a 3-tuple/list if spec brightness/contrast/saturation
            # or 4 if also augmenting hue
            assert len(color_jitter) in (3, 4)
        else:
            # if it's a scalar, duplicate for brightness, contrast, and saturation, no hue
            color_jitter = (float(color_jitter),) * 3
        secondary_tfl += [transforms.ColorJitter(*color_jitter)]

    final_tfl = []
    if use_prefetcher:
        # prefetcher and collate will handle tensor conversion and norm
        final_tfl += [ToNumpy()]
    else:
        final_tfl += [
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        ]
        if re_prob > 0.0:
            final_tfl.append(
                RandomErasing(
                    re_prob,
                    mode=re_mode,
                    max_count=re_count,
                    num_splits=re_num_splits,
                    device="cpu",
                )
            )

    if separate:
        return (
            transforms.Compose(primary_tfl),
            transforms.Compose(secondary_tfl),
            transforms.Compose(final_tfl),
        )
    else:
        return transforms.Compose(primary_tfl + secondary_tfl + final_tfl)


def transforms_cifar10_eval(
    img_size=32,
    crop_pct=None,
    interpolation="bilinear",
    use_prefetcher=False,
    mean=CIFAR10_DEFAULT_MEAN,
    std=CIFAR10_DEFAULT_STD,
):
    crop_pct = crop_pct or DEFAULT_CROP_PCT

    if isinstance(img_size, (tuple, list)):
        assert len(img_size) == 2
        if img_size[-1] == img_size[-2]:
            # fall-back to older behaviour so Resize scales to shortest edge if target is square
            scale_size = int(math.floor(img_size[0] / crop_pct))
        else:
            scale_size = tuple([int(x / crop_pct) for x in img_size])
    else:
        scale_size = int(math.floor(img_size / crop_pct))

    tfl = [
        transforms.Resize(scale_size, interpolation=str_to_interp_mode(interpolation)),
        transforms.CenterCrop(img_size),
    ]
    if use_prefetcher:
        # prefetcher and collate will handle tensor conversion and norm
        tfl += [ToNumpy()]
    else:
        tfl += [
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        ]

    return transforms.Compose(tfl)


def create_transform(
    input_size,
    is_training=False,
    use_prefetcher=False,
    no_aug=False,
    scale=None,
    ratio=None,
    hflip=0.5,
    vflip=0.0,
    color_jitter=0.4,
    auto_augment=None,
    interpolation="bilinear",
    mean=CIFAR10_DEFAULT_MEAN,
    std=CIFAR10_DEFAULT_STD,
    re_prob=0.0,
    re_mode="const",
    re_count=1,
    re_num_splits=0,
    crop_pct=None,
    separate=False,
):

    if isinstance(input_size, (tuple, list)):
        img_size = input_size[-2:]
    else:
        img_size = input_size

    if is_training and no_aug:
        assert not separate, "Cannot perform split augmentation with no_aug"
        transform = transforms_noaug_train(
            img_size,
            interpolation=interpolation,
            use_prefetcher=use_prefetcher,
            mean=mean,
            std=std,
        )
    elif is_training:
        transform = transforms_cifar10_train(
            img_size,
            scale=scale,
            ratio=ratio,
            hflip=hflip,
            vflip=vflip,
            color_jitter=color_jitter,
            auto_augment=auto_augment,
            interpolation=interpolation,
            use_prefetcher=use_prefetcher,
            mean=mean,
            std=std,
            re_prob=re_prob,
            re_mode=re_mode,
            re_count=re_count,
            re_num_splits=re_num_splits,
            separate=separate,
        )
    else:
        assert (
            not separate
        ), "Separate transforms not supported for validation preprocessing"
        transform = transforms_cifar10_eval(
            img_size,
            interpolation=interpolation,
            use_prefetcher=use_prefetcher,
            mean=mean,
            std=std,
            crop_pct=crop_pct,
        )

    return transform


def get_train_valid_loader_CIFAR10(
    data_dir,
    batch_size,
    train_transform_config,
    valid_transform_config,
    random_seed,
    valid_size=0.1,
    shuffle=True,
    num_workers=4,
    pin_memory=False,
    show_sample=True,
    augment=True,
):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert (valid_size >= 0) and (valid_size <= 1), error_msg

    if augment == True:
        train_transform = create_transform(
            input_size=train_transform_config["input_size"],
            is_training=train_transform_config["is_training"],
        )
        valid_transform = create_transform(
            input_size=valid_transform_config["input_size"],
            is_training=valid_transform_config["is_training"],
        )
    else:
        normalize = transforms.Normalize(
            mean=CIFAR10_DEFAULT_MEAN,
            std=CIFAR10_DEFAULT_STD,
        )
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )
        valid_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )

    # load the dataset
    train_dataset = CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )

    valid_dataset = CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=valid_transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if show_sample:
        sample_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=9,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        data_iter = iter(sample_loader)
        images, labels = next(data_iter)
        X = images.numpy().transpose([0, 2, 3, 1])

    return (train_loader, valid_loader)


def get_test_loader(
    data_dir, batch_size, shuffle=True, num_workers=2, pin_memory=False
):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """
    normalize = transforms.Normalize(
        mean=CIFAR10_DEFAULT_MEAN,
        std=CIFAR10_DEFAULT_STD,
    )

    # define transform
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )

    dataset = CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return data_loader


def get_train_valid_loader_CIFAR100(
    data_dir,
    batch_size,
    train_transform_config,
    valid_transform_config,
    random_seed,
    valid_size=0.1,
    shuffle=True,
    num_workers=4,
    pin_memory=False,
    show_sample=True,
    augment=True,
):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert (valid_size >= 0) and (valid_size <= 1), error_msg

    if augment == True:
        train_transform = create_transform(
            input_size=train_transform_config["input_size"],
            is_training=train_transform_config["is_training"],
        )
        valid_transform = create_transform(
            input_size=valid_transform_config["input_size"],
            is_training=valid_transform_config["is_training"],
        )
    else:
        normalize = transforms.Normalize(
            mean=CIFAR100_DEFAULT_MEAN,
            std=CIFAR100_DEFAULT_STD,
        )
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )
        valid_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )
    # load the dataset
    train_dataset = CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )

    valid_dataset = CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=valid_transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if show_sample:
        sample_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=9,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        data_iter = iter(sample_loader)
        images, labels = next(data_iter)
        X = images.numpy().transpose([0, 2, 3, 1])

    return (train_loader, valid_loader)


def get_test_loader_CIFAR100(
    data_dir, batch_size, shuffle=True, num_workers=2, pin_memory=False
):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """
    normalize = transforms.Normalize(
        mean=CIFAR100_DEFAULT_MEAN,
        std=CIFAR100_DEFAULT_STD,
    )

    # define transform
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )

    dataset = CIFAR100(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return data_loader


def get_CUB_data_loaders(data_dir, batch_size, train=False, num_workers=2):
    if train:
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomApply(
                    torch.nn.ModuleList([transforms.ColorJitter()]), p=0.1
                ),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                #             transforms.RandomErasing(p=0.25, value='random')
            ]
        )
        all_data = datasets.ImageFolder(data_dir, transform=transform)
        train_data_len = int(len(all_data) * 0.75)
        valid_data_len = int((len(all_data) - train_data_len) / 2)
        test_data_len = int(len(all_data) - train_data_len - valid_data_len)
        train_data, val_data, test_data = random_split(
            all_data, [train_data_len, valid_data_len, test_data_len]
        )
        train_loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        return train_loader

    else:
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        all_data = datasets.ImageFolder(data_dir, transform=transform)
        train_data_len = int(len(all_data) * 0.70)
        valid_data_len = int((len(all_data) - train_data_len) / 2)
        test_data_len = int(len(all_data) - train_data_len - valid_data_len)
        train_data, val_data, test_data = random_split(
            all_data, [train_data_len, valid_data_len, test_data_len]
        )
        val_loader = DataLoader(
            val_data, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        test_loader = DataLoader(
            test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        return (val_loader, test_loader)


def split_dataset(dataset, train_ratio, val_ratio):
    total_samples = len(dataset)
    train_size = int(train_ratio * total_samples)
    val_size = int(val_ratio * total_samples)
    test_size = total_samples - train_size - val_size

    # Get a list of unique class labels
    classes = dataset.class_to_idx
    num_classes = len(classes)
    class_samples = {v: [] for _, v in classes.items()}

    # Group samples by class
    for i in range(total_samples):
        class_samples[dataset[i][1]].append(i)

    # Shuffle the samples in each class
    for _, v in classes.items():
        random.shuffle(class_samples[v])

    # Split indices for each set
    train_indices = []
    val_indices = []
    test_indices = []

    for _, v in classes.items():
        class_total_samples = len(class_samples[v])
        train_count = int(train_ratio * class_total_samples)
        val_count = int(val_ratio * class_total_samples)

        train_indices.extend(class_samples[v][:train_count])
        val_indices.extend(class_samples[v][train_count : train_count + val_count])
        test_indices.extend(class_samples[v][train_count + val_count :])

    # Create data loaders for each set
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset


def get_miniImageNet_dataloaders(data_dir: str, batch_size: int, augment: bool = True):
    if augment:
        transform = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
                ),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
                ),
            ]
        )

    test_transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )
    dataset = datasets.ImageFolder(root=data_dir)
    train_dataset, val_dataset, test_dataset = split_dataset(
        dataset=dataset, train_ratio=0.8, val_ratio=0.1
    )

    train_dataset.dataset.transform = transform
    val_dataset.dataset.transform = test_transform
    test_dataset.dataset.transform = test_transform

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    # For the validation dataset, use the val_test_transform (without augmentation)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    # For the test dataset, also use the val_test_transform (without augmentation)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader
