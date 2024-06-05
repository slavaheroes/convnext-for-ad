# -*- coding: utf-8 -*-
"""
Created on Wed Feb 8 2023

@author: qasymjomart
"""

from __future__ import print_function, division
import pandas as pd
from sklearn.model_selection import train_test_split

import monai
from monai import data
from monai import transforms

import glob
import os
from natsort import natsorted

def make_mae_pretraining_dataloaders(cfg, args):
    """Build datalaoders for MAE pretraining

    Args:
        cfg : config (read from yaml)
        args : argument parser from command line
    
    Returns:
    ------------
    pretraining_dataloader : type (torch.utils.data.DataLoader)

    """
    dataset_list = []
    datapath_list = []
    
 
    if "IXI" in args.datasets_to_use:
        datapath_list = datapath_list + glob.glob(cfg['IXI']['dataroot'])
        print('Used IXI')
    
    if "HCP" in args.datasets_to_use:
        datapath_list = datapath_list + glob.glob(cfg['HCP']['dataroot'])
        print('Used HCP')
    
    if "ADNI1" in args.datasets_to_use:
        datapath_list = datapath_list + glob.glob(cfg['ADNI1']['dataroot'])
        print('Used ADNI1')
    
    if "ADNI2" in args.datasets_to_use:
        datapath_list = datapath_list + glob.glob(cfg['ADNI2']['dataroot'])
        print('Used ADNI2')
    
    if "OASIS3" in args.datasets_to_use:
        # have to include code for sorting out healthy subjects
        datapath_temp_list = glob.glob(cfg["OASIS3"]["dataroot"])
        datapath_list = datapath_list + datapath_temp_list
        print("Used OASIS3")

    for data_path in datapath_list:
        dataset_list.append({"image": data_path})
    
    if args.use_aug:
        # MAE training transforms with Geometric augmentations
        train_transforms = monai.transforms.Compose([
            monai.transforms.LoadImaged(keys=["image"]),
            monai.transforms.EnsureChannelFirstd(keys=["image",]),
            monai.transforms.Orientationd(keys=["image"], axcodes=cfg["TRANSFORMS"]["orientation"]),
            monai.transforms.Spacingd(keys=["image"], pixdim=tuple(cfg["TRANSFORMS"]["spacing"])),
            monai.transforms.NormalizeIntensityd(keys=["image"], nonzero=cfg["TRANSFORMS"]["normalize_non_zero"]),
            monai.transforms.CropForegroundd(keys=["image"], source_key="image"), 
            monai.transforms.Resized(keys=["image"], spatial_size=tuple(cfg["TRANSFORMS"]["resize"])),
            monai.transforms.RandFlipd(keys=["image"], prob=0.1, spatial_axis=0),
            monai.transforms.RandFlipd(keys=["image"], prob=0.1, spatial_axis=1),
            monai.transforms.RandFlipd(keys=["image"], prob=0.1, spatial_axis=2),
            monai.transforms.RandRotate90d(keys=["image"], prob=0.1, max_k=3),
            monai.transforms.ToTensord(keys=["image"])
            ])
        print("Geometric Augmentations used")
    else:
        train_transforms = monai.transforms.Compose([
            monai.transforms.LoadImaged(keys=["image"]),
            monai.transforms.EnsureChannelFirstd(keys=["image",]),
            monai.transforms.Orientationd(keys=["image"], axcodes=cfg["TRANSFORMS"]["orientation"]),
            monai.transforms.Spacingd(keys=["image"], pixdim=tuple(cfg["TRANSFORMS"]["spacing"])),
            monai.transforms.NormalizeIntensityd(keys=["image"], nonzero=cfg["TRANSFORMS"]["normalize_non_zero"]),
            monai.transforms.CropForegroundd(keys=["image"], source_key="image"), 
            monai.transforms.Resized(keys=["image"], spatial_size=tuple(cfg["TRANSFORMS"]["resize"])),
            monai.transforms.ToTensord(keys=["image"])
            ])
        
        print("No augmentations used")
        

        
    pretrain_dataset = data.PersistentDataset(
            data=dataset_list,
            transform=train_transforms,
            cache_dir=cfg['TRANSFORMS']['cache_dir_train']
        )
    
    pretraining_dataloader = data.DataLoader(pretrain_dataset, 
                                    batch_size=cfg['DATALOADER']['BATCH_SIZE'],
                                    shuffle=True, 
                                    num_workers=cfg['DATALOADER']['NUM_WORKERS'],
                                    pin_memory=True,
                                    # drop_last=True
                                    )
    
    return pretraining_dataloader, pretrain_dataset


def make_kfold_dataloaders(cfg, args, train_df, test_df):

    train_transforms = monai.transforms.Compose([
        monai.transforms.LoadImaged(keys=["image"]),
        monai.transforms.EnsureChannelFirstd(keys=["image"]),
        monai.transforms.Orientationd(keys=["image"], axcodes=cfg["TRANSFORMS"]["orientation"]),
        monai.transforms.Spacingd(keys=["image"], pixdim=tuple(cfg["TRANSFORMS"]["spacing"])),
        monai.transforms.CropForegroundd(keys=["image"], source_key="image"), 
        monai.transforms.NormalizeIntensityd(keys=["image"], nonzero=cfg["TRANSFORMS"]["normalize_non_zero"]),
        monai.transforms.Resized(keys=["image"], spatial_size=tuple(cfg["TRANSFORMS"]["resize"])),
        monai.transforms.RandFlipd(keys=["image"], prob=0.2, spatial_axis=0),
        monai.transforms.RandFlipd(keys=["image"], prob=0.2, spatial_axis=1),
        monai.transforms.RandFlipd(keys=["image"], prob=0.2, spatial_axis=2),
        monai.transforms.RandRotate90d(keys=["image"], prob=0.2, max_k=3),
        monai.transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=0.2),
        monai.transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.2),
        monai.transforms.RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1), 
        monai.transforms.ToTensord(keys=["image", "label"])
    ])
    
    test_transforms = monai.transforms.Compose([
        monai.transforms.LoadImaged(keys=["image"]),
        monai.transforms.EnsureChannelFirstd(keys=["image",]),
        monai.transforms.Orientationd(keys=["image"], axcodes=cfg["TRANSFORMS"]["orientation"]),
        monai.transforms.Spacingd(keys=["image"], pixdim=tuple(cfg["TRANSFORMS"]["spacing"])),
        monai.transforms.CropForegroundd(keys=["image"], source_key="image"),
        monai.transforms.NormalizeIntensityd(keys=["image"], nonzero=cfg["TRANSFORMS"]["normalize_non_zero"]),
        monai.transforms.Resized(keys=["image"], spatial_size=tuple(cfg["TRANSFORMS"]["resize"])),
        monai.transforms.ToTensord(keys=["image", "label"])
    ])


    nii_list = natsorted(glob.glob(cfg[args.dataset]['dataroot'] + '*/hdbet_*[!mask].nii.gz'))
    print(f'{len(nii_list)} nii files found.')

    # if need to train with few samples, split in a stratified fashion
    if cfg["DATALOADER"]["train_size"] in [0.1, 0.2, 0.4, 0.6, 0.8]:
        train_df, _, _, _ = train_test_split(train_df, train_df["Group"], 
                                             train_size=int(cfg["DATALOADER"]["train_size"]*len(train_df)), random_state=args.seed,
                                             shuffle=True, stratify=train_df["Group"])
        print(f'Few sample training of {100*cfg["DATALOADER"]["train_size"]} % samples: {len(train_df)}')

    train_datalist = []
    for _, row in train_df.iterrows():
        label = args.classes_to_use.index(row["Group"])
        path_to_file = [x for x in nii_list if row['Subject'] in x and row['Image Data ID'] in x]
        assert len(path_to_file) == 1, f'More than one file found for {row["Subject"]} and {row["Image Data ID"]}'
        
        train_datalist.append({
            "image": path_to_file[0],
            "label": label
        })

    ratios_train = {}
    for label in args.classes_to_use:
        label_id = args.classes_to_use.index(label)
        ratios_train[label] = sum([1 for x in train_datalist if x['label'] == label_id])
    
    train_dataset = data.PersistentDataset(data=train_datalist, 
                                           transform=train_transforms, 
                                           cache_dir=cfg['TRANSFORMS']['cache_dir_train'])
    print(f'Train dataset len: {len(train_dataset)}')
    
    test_datalist = []

    for _, row in test_df.iterrows():
        label = args.classes_to_use.index(row["Group"])
        path_to_file = [x for x in nii_list if row['Subject'] in x and row['Image Data ID'] in x]
        assert len(path_to_file) == 1, f'More than one file found for {row["Subject"]} and {row["Image Data ID"]}'

        test_datalist.append({
            "image": path_to_file[0],
            "label": label
        })

    test_dataset = data.PersistentDataset(data=test_datalist, 
                                          transform=test_transforms, 
                                          cache_dir=cfg['TRANSFORMS']['cache_dir_test'])
    print(f'Test dataset len: {len(test_dataset)}')

    train_dataloader = data.DataLoader(train_dataset, 
                        batch_size=cfg['DATALOADER']['BATCH_SIZE'],
                        shuffle=True, 
                        num_workers=cfg["DATALOADER"]["NUM_WORKERS"],
                        drop_last=True,
                        pin_memory=True
                        )

    test_dataloader = data.DataLoader(test_dataset, 
                                    batch_size=1,
                                    shuffle=False, 
                                    num_workers=0
                                    )

    ratios_test = {}
    for label in args.classes_to_use:
        label_id = args.classes_to_use.index(label)
        ratios_test[label] = sum([1 for x in test_datalist if x['label'] == label_id])

    return train_dataloader, test_dataloader, train_dataset, test_dataset, ratios_train, ratios_test
