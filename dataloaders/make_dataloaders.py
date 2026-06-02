# -*- coding: utf-8 -*-
"""
Created on Wed Feb 8 2023

@author: qasymjomart
"""

from sklearn.model_selection import train_test_split
import glob
import os
import pandas as pd
from natsort import natsorted
import monai
from monai import data

def replace_data_path(datapath):
    servername = os.uname()[1]
    if os.path.exists('/SSD/qasymjomart'):
        print(f'Configured path for {servername} server')
        return datapath
    elif os.path.exists('/SSD/guest/qasymjomart'):
        print(f'Configured path for {servername} server')
        return datapath.replace('/SSD/qasymjomart/', '/SSD/guest/qasymjomart/')
    elif os.path.exists('/DATA3/guest/qasymjomart'):
        print(f'Configured path for {servername} server')
        return datapath.replace('/SSD/qasymjomart/', '/DATA3/guest/qasymjomart/')
    else:
        print(f'Path configuration for {servername} server not found. Using original path.')
    return datapath

def prepare_pt_data(cfg, args):
    """Build datalaoders for pretraining

    Args:
        cfg : config (read from yaml)
        args : argument parser from command line
    
    Returns:
    ------------
    pretraining_dataloader : type (torch.utils.data.DataLoader)

    """
    dataset_list = []
    datapath_list = []
    
 
    if "IXI" in args.datasets:
        datapath_list = datapath_list + glob.glob(cfg['IXI']['dataroot'])
        print('Used IXI')
    
    if "HCP" in args.datasets:
        datapath_list = datapath_list + glob.glob(cfg['HCP']['dataroot'])
        print('Used HCP')
    
    if "BRATS2023" in args.datasets:
        datapath_list = datapath_list + glob.glob(cfg["BRATS2023"]["dataroot"])
        print('Used BRATS2023')    
        
    if "ADNI1" in args.datasets:
        datapath_list = datapath_list + glob.glob(cfg['ADNI1']['dataroot'])
        print('Used ADNI1')
    
    if "ADNI2" in args.datasets:
        datapath_list = datapath_list + glob.glob(cfg['ADNI2']['dataroot'])
        print('Used ADNI2')
    
    if "OASIS3" in args.datasets: # have to include code for sorting out healthy subjects
        datapath_temp_list = glob.glob(cfg["OASIS3"]["dataroot"])
        # Step 1: Read the cfg["OASIS3"]["labelsroot"] file into a DataFrame and select the "filename" column
        df_healthy_oasis3 = pd.read_csv(cfg["OASIS3"]["labelsroot"])
        filenames = df_healthy_oasis3['filename'].tolist()
        # Step 2: Remove the .json ending from each entry in the list, replace it with .nii.gz, and add 'hdbet_' as a prefix
        filenames = ['hdbet_' + filename.replace('.json', '.nii.gz') for filename in filenames]
        # Step 3: Select the subset of datapath_list where the filename matches with the entries of the earlier created list
        datapath_list = datapath_list + [path for path in datapath_temp_list if os.path.basename(path) in filenames]
        print('Used OASIS3')

    datapath_list = natsorted(datapath_list)
    dataset_list = [{"image": x} for x in datapath_list]
    
    if args.use_aug:
        print('Using data augmentations')
        # train_transforms = monai.transforms.Compose([
        #     monai.transforms.LoadImaged(keys=["image"]),
        #     monai.transforms.EnsureChannelFirstd(keys=["image",]),
        #     monai.transforms.Orientationd(keys=["image"], axcodes=cfg.transforms.orientation),
        #     monai.transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=-1, b_max=1, clip=True),
        #     monai.transforms.Spacingd(keys=["image"], pixdim=tuple(cfg.transforms.spacing)),
        #     monai.transforms.CropForegroundd(keys=["image"], source_key="image"), 
        #     # monai.transforms.RandSpatialCropd(keys=["image"], roi_size=(80,80,80), max_roi_size=tuple(cfg.transforms.resize)),
        #     monai.transforms.Resized(keys=["image"], spatial_size=tuple(cfg.transforms.resize)),
        #     monai.transforms.RandFlipd(keys=["image"], prob=0.2, spatial_axis=0),
        #     monai.transforms.RandFlipd(keys=["image"], prob=0.2, spatial_axis=1),
        #     monai.transforms.RandFlipd(keys=["image"], prob=0.2, spatial_axis=2),
        #     monai.transforms.RandRotate90d(keys=["image"], prob=0.2, max_k=3),
        #     monai.transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=0.2),
        #     monai.transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.2),
        #     monai.transforms.ToTensord(keys=["image"])
        #     ])
        train_transforms = monai.transforms.Compose([
            monai.transforms.LoadImaged(keys=["image"]),
            monai.transforms.EnsureChannelFirstd(keys=["image",]),
            monai.transforms.Orientationd(keys=["image"], axcodes=cfg.transforms.orientation),
            monai.transforms.Spacingd(keys=["image"], pixdim=tuple(cfg.transforms.spacing)),
            monai.transforms.NormalizeIntensityd(keys=["image"], nonzero=cfg.transforms.normalize_non_zero),
            monai.transforms.CropForegroundd(keys=["image"], source_key="image"), 
            monai.transforms.Resized(keys=["image"], spatial_size=tuple(cfg.transforms.resize)),
            monai.transforms.RandFlipd(keys=["image"], prob=0.1, spatial_axis=0),
            monai.transforms.RandFlipd(keys=["image"], prob=0.1, spatial_axis=1),
            monai.transforms.RandFlipd(keys=["image"], prob=0.1, spatial_axis=2),
            monai.transforms.RandRotate90d(keys=["image"], prob=0.1, max_k=3),
            monai.transforms.ToTensord(keys=["image"])
        ])
    else:
        print('No data augmentations used')
        train_transforms = monai.transforms.Compose([
            monai.transforms.LoadImaged(keys=["image"]),
            monai.transforms.EnsureChannelFirstd(keys=["image",]),
            monai.transforms.Orientationd(keys=["image"], axcodes=cfg.transforms.orientation),
            monai.transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=-1, b_max=1, clip=True),
            monai.transforms.Spacingd(keys=["image"], pixdim=tuple(cfg.transforms.spacing)),
            monai.transforms.CropForegroundd(keys=["image"], source_key="image", allow_smaller=True), 
            monai.transforms.Resized(keys=["image"], spatial_size=tuple(cfg.transforms.resize)),
            monai.transforms.ToTensord(keys=["image"])
        ])
        
    
    # train_transforms.set_random_state(args.seed)
    dataset = data.PersistentDataset(data=dataset_list, transform=train_transforms, cache_dir=cfg.transforms.cache_dir_train)
    
    data_loader = data.DataLoader(dataset, 
                                    batch_size=cfg.training.batch_size,
                                    shuffle=True, 
                                    num_workers=cfg.training.num_workers,
                                    pin_memory=True,
                                    persistent_workers=True if cfg.training.num_workers > 0 else False,
                                    )
    
    return data_loader, dataset

def make_aibl_test_dataloader(cfg, args, verbose=True):
    
    classes_to_use = []
    if 'CN' in args.classes_to_use:
        classes_to_use.append(1)
    if 'AD' in args.classes_to_use:
        classes_to_use.append(3)
    
    dataset = 'AIBL'
    test_transforms = monai.transforms.Compose([
        monai.transforms.LoadImaged(keys=["image"]),
        monai.transforms.EnsureChannelFirstd(keys=["image",]),
        monai.transforms.Orientationd(keys=["image"], axcodes=cfg["transforms"]["orientation"]),
        monai.transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=-1, b_max=1, clip=True),
        monai.transforms.Spacingd(keys=["image"], pixdim=tuple(cfg["transforms"]["spacing"])),
        monai.transforms.CropForegroundd(keys=["image"], source_key="image"),
        monai.transforms.Resized(keys=["image"], spatial_size=tuple(cfg["transforms"]["resize"])),
        monai.transforms.ToTensord(keys=["image", "label"])
    ])
    # test_transforms.set_random_state(args.seed)

    nii_list = natsorted(glob.glob(cfg[dataset]['dataroot']) + '*/hdbet_*[!mask].nii.gz')
    if verbose:
        print(f'{len(nii_list)} nii files found.')
    
    test_datalist = []

    test_df = pd.read_csv(cfg[dataset]['labelsroot'])
    test_df = test_df[test_df['DXCURREN'].isin(classes_to_use)]
    for _, row in test_df.iterrows():
        label = classes_to_use.index(row["DXCURREN"])
        path_to_file = [x for x in nii_list if f"_{row['RID']}_" in x and 'ADNI_confirmed' in x]
        assert len(path_to_file) == 1, f"Expected one file for RID {row['RID']}, found {len(path_to_file)}"

        test_datalist.append({
            "image": path_to_file[0],
            "label": label
        })

    test_dataset = data.PersistentDataset(data=test_datalist, 
                                          transform=test_transforms, 
                                          cache_dir=cfg['transforms']['cache_dir_test'])
    
    test_dataloader = data.DataLoader(test_dataset, 
                                    batch_size=4,
                                    shuffle=False, 
                                    num_workers=0
                                    )

    ratios_test = {}
    for label in classes_to_use:
        label_id = classes_to_use.index(label)
        ratios_test[label] = sum([1 for x in test_datalist if x['label'] == label_id])
    
    print(f'{dataset} test dataset and dataloader built. Len: {len(test_dataset)}')
    
    return test_dataloader, test_dataset, ratios_test


def make_adni2_test_dataloader(cfg, args, verbose=True):
    dataset = 'ADNI2'
    test_transforms = monai.transforms.Compose([
        monai.transforms.LoadImaged(keys=["image"]),
        monai.transforms.EnsureChannelFirstd(keys=["image",]),
        monai.transforms.Orientationd(keys=["image"], axcodes=cfg["transforms"]["orientation"]),
        monai.transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=-1, b_max=1, clip=True),
        monai.transforms.Spacingd(keys=["image"], pixdim=tuple(cfg["transforms"]["spacing"])),
        monai.transforms.CropForegroundd(keys=["image"], source_key="image"),
        monai.transforms.Resized(keys=["image"], spatial_size=tuple(cfg["transforms"]["resize"])),
        monai.transforms.ToTensord(keys=["image", "label"])
    ])
    # test_transforms.set_random_state(args.seed)

    nii_list = natsorted(glob.glob(cfg[dataset]['dataroot']) + '*/hdbet_*[!mask].nii.gz')
    if verbose:
        print(f'{len(nii_list)} nii files found.')
    
    test_datalist = []

    test_df = pd.read_csv(cfg[dataset]['labelsroot'])
    test_df = test_df[test_df['Group'].isin(args.classes_to_use)]
    for _, row in test_df.iterrows():
        label = args.classes_to_use.index(row["Group"])
        path_to_file = [x for x in nii_list if row['Subject'] in x and row['Image Data ID'] in x]
        assert len(path_to_file) == 1, f'More than one file found for {row["Subject"]} and {row["Image Data ID"]}. Length: {len(path_to_file)}'

        test_datalist.append({
            "image": path_to_file[0],
            "label": label
        })

    test_dataset = data.PersistentDataset(data=test_datalist, 
                                          transform=test_transforms, 
                                          cache_dir=cfg['transforms']['cache_dir_test'])
    
    test_dataloader = data.DataLoader(test_dataset, 
                                    batch_size=4,
                                    shuffle=False, 
                                    num_workers=0
                                    )

    ratios_test = {}
    for label in args.classes_to_use:
        label_id = args.classes_to_use.index(label)
        ratios_test[label] = sum([1 for x in test_datalist if x['label'] == label_id])
    
    print(f'{dataset} test dataset and dataloader built. Len: {len(test_dataset)}')
    
    return test_dataloader, test_dataset, ratios_test
    
def make_kfold_dataloaders(cfg, args, train_df, test_df, verbose=True):

    if args.use_aug:
        # old augmentation
        # train_transforms = monai.transforms.Compose([
        #     monai.transforms.LoadImaged(keys=["image"]),
        #     monai.transforms.EnsureChannelFirstd(keys=["image"]),
        #     monai.transforms.Orientationd(keys=["image"], axcodes=cfg["transforms"]["orientation"]),
        #     monai.transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=-1, b_max=1, clip=True),
        #     monai.transforms.Spacingd(keys=["image"], pixdim=tuple(cfg["transforms"]["spacing"])),
        #     monai.transforms.CropForegroundd(keys=["image"], source_key="image"), 
        #     # monai.transforms.NormalizeIntensityd(keys=["image"], nonzero=cfg["TRANSFORMS"]["normalize_non_zero"]),
        #     monai.transforms.Resized(keys=["image"], spatial_size=tuple(cfg["transforms"]["resize"])),
        #     monai.transforms.RandFlipd(keys=["image"], prob=0.2, spatial_axis=0),
        #     monai.transforms.RandFlipd(keys=["image"], prob=0.2, spatial_axis=1),
        #     monai.transforms.RandFlipd(keys=["image"], prob=0.2, spatial_axis=2),
        #     monai.transforms.RandRotate90d(keys=["image"], prob=0.2, max_k=3),
        #     monai.transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=0.2), # must be disabled
        #     monai.transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.2), # must be disabled
        #     # monai.transforms.RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1), # must be disabled
        #     monai.transforms.ToTensord(keys=["image", "label"])
        # ])
        
        train_transforms = monai.transforms.Compose([
            monai.transforms.LoadImaged(keys=["image"]),
            monai.transforms.EnsureChannelFirstd(keys=["image"]),
            monai.transforms.Orientationd(keys=["image"], axcodes=cfg.transforms.orientation),
            monai.transforms.Spacingd(keys=["image"], pixdim=tuple(cfg.transforms.spacing)),
            monai.transforms.CropForegroundd(keys=["image"], source_key="image"), 
            monai.transforms.NormalizeIntensityd(keys=["image"], nonzero=cfg.transforms.normalize_non_zero),
            monai.transforms.Resized(keys=["image"], spatial_size=tuple(cfg.transforms.resize)),
            monai.transforms.RandFlipd(keys=["image"], prob=0.2, spatial_axis=0),
            monai.transforms.RandFlipd(keys=["image"], prob=0.2, spatial_axis=1),
            monai.transforms.RandFlipd(keys=["image"], prob=0.2, spatial_axis=2),
            monai.transforms.RandRotate90d(keys=["image"], prob=0.2, max_k=3),
            monai.transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=0.2),
            monai.transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.2),
            monai.transforms.RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1), 
            monai.transforms.ToTensord(keys=["image", "label"])
        ])
            
        
    else:
        # train_transforms = monai.transforms.Compose([
        #     monai.transforms.LoadImaged(keys=["image"]),
        #     monai.transforms.EnsureChannelFirstd(keys=["image"]),
        #     monai.transforms.Orientationd(keys=["image"], axcodes=cfg["transforms"]["orientation"]),
        #     monai.transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=-1, b_max=1, clip=True),
        #     monai.transforms.Spacingd(keys=["image"], pixdim=tuple(cfg["transforms"]["spacing"])),
        #     monai.transforms.CropForegroundd(keys=["image"], source_key="image"), 
        #     monai.transforms.Resized(keys=["image"], spatial_size=tuple(cfg["transforms"]["resize"])),
        #     monai.transforms.ToTensord(keys=["image", "label"])
        # ])
        train_transforms = monai.transforms.Compose([
            monai.transforms.LoadImaged(keys=["image"]),
            monai.transforms.EnsureChannelFirstd(keys=["image"]),
            monai.transforms.Orientationd(keys=["image"], axcodes=cfg["transforms"]["orientation"]),
            monai.transforms.Spacingd(keys=["image"], pixdim=tuple(cfg["transforms"]["spacing"])),
            monai.transforms.CropForegroundd(keys=["image"], source_key="image"),
            monai.transforms.NormalizeIntensityd(keys=["image"], nonzero=cfg.transforms.normalize_non_zero),
            monai.transforms.Resized(keys=["image"], spatial_size=tuple(cfg["transforms"]["resize"])),
            monai.transforms.ToTensord(keys=["image", "label"])
        ])


    # test_transforms = monai.transforms.Compose([
    #     monai.transforms.LoadImaged(keys=["image"]),
    #     monai.transforms.EnsureChannelFirstd(keys=["image",]),
    #     monai.transforms.Orientationd(keys=["image"], axcodes=cfg["transforms"]["orientation"]),
    #     monai.transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=-1, b_max=1, clip=True),
    #     monai.transforms.Spacingd(keys=["image"], pixdim=tuple(cfg["transforms"]["spacing"])),
    #     monai.transforms.CropForegroundd(keys=["image"], source_key="image"),
    #     monai.transforms.Resized(keys=["image"], spatial_size=tuple(cfg["transforms"]["resize"])),
    #     monai.transforms.ToTensord(keys=["image", "label"])
    # ])
    test_transforms = monai.transforms.Compose([
        monai.transforms.LoadImaged(keys=["image"]),
        monai.transforms.EnsureChannelFirstd(keys=["image",]),
        monai.transforms.Orientationd(keys=["image"], axcodes=cfg.transforms.orientation),
        monai.transforms.Spacingd(keys=["image"], pixdim=tuple(cfg.transforms.spacing)),
        monai.transforms.CropForegroundd(keys=["image"], source_key="image"),
        monai.transforms.NormalizeIntensityd(keys=["image"], nonzero=cfg.transforms.normalize_non_zero),
        monai.transforms.Resized(keys=["image"], spatial_size=tuple(cfg.transforms.resize)),
        monai.transforms.ToTensord(keys=["image", "label"])
    ])

    
    # train_transforms.set_random_state(args.seed)
    # test_transforms.set_random_state(args.seed)
    
    nii_list = natsorted(glob.glob(cfg[args.dataset]['dataroot'] + '*/hdbet_*[!mask].nii.gz'))
    if verbose:
        print(f'{len(nii_list)} nii files found.')

    # if need to train with few samples, split in a stratified fashion
    if cfg["training"]["train_size"] in [0.05, 0.1, 0.2, 0.25, 0.4, 0.5, 0.6, 0.8]:
        train_df, _, _, _ = train_test_split(train_df, train_df["Group"], 
                                             train_size=int(cfg["training"]["train_size"]*len(train_df)), random_state=args.seed,
                                             shuffle=True, stratify=train_df["Group"])
        if verbose:
            print(f'Few sample training of {100*cfg["training"]["train_size"]} % samples: {len(train_df)}')

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
                                           cache_dir=cfg['transforms']['cache_dir_train']
                                           )
    if verbose:
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
                                          cache_dir=cfg['transforms']['cache_dir_test'])
    if verbose:
        print(f'Test dataset len: {len(test_dataset)}')

    train_dataloader = data.DataLoader(train_dataset, 
                        batch_size=cfg['training']['batch_size'],
                        shuffle=True, 
                        num_workers=cfg["training"]["num_workers"],
                        drop_last=True, pin_memory=True,
                        persistent_workers=True if cfg["training"]["num_workers"] > 0 else False
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