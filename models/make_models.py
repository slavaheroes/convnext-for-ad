# -*- coding: utf-8 -*-
"""
Created on Wed Feb 8 2023

@author: qasymjomart
"""
# from make_dataloaders import make_dataloaders
from .convnets import make_densenet1213d, make_resnet103d, make_resnet183d, make_resnet343d, make_resnet1013d, make_resnet1523d
from .convnextv2_3d import ConvNeXtV2_3D
from .fcmae_3d import FCMAE_3D
from .mednext_forclassification import MedNeXtEncoderOnly
from .vit3d import Vision_Transformer3D
from .maskedautoencoder3d import MaskedAutoencoderViT3D

_models_factory = {
    'DenseNet121': make_densenet1213d,
    'ResNet10': make_resnet103d,
    'ResNet18': make_resnet183d,
    'ResNet34': make_resnet343d,
    'ResNet101': make_resnet1013d,
    'ResNet152': make_resnet1523d,
    'ConvNeXtV2_3D': ConvNeXtV2_3D,
    'ViT3D': Vision_Transformer3D,
    'MaskedAutoencoderViT3D': MaskedAutoencoderViT3D,
    'FCMAE_3D': FCMAE_3D,
}

       
def make_vanilla_model(cfg, args):
    """
    Make models for vanilla classifcation training

    """
    if cfg.model.arch in ['ViT3D']:
        assert cfg.model.arch in _models_factory.keys(), cfg.model.arch + ' not in the model factory list'

        model = _models_factory[cfg.model.arch](
            img_size        = cfg['model']['img_size'],
            patch_size      = cfg['model']['patch_size'],
            in_chans        = cfg['model']['in_chans'],
            n_classes       = cfg['model']['n_classes'],
            embed_dim       = cfg['model']['embed_dim'],
            depth           = cfg['model']['depth'],
            n_heads         = cfg['model']['n_heads'],
            mlp_ratio       = cfg['model']['mlp_ratio'],
            qkv_bias        = cfg['model']['qkv_bias'],
            drop_path_rate  = cfg['model']['drop_path_rate'],
            p               = cfg['model']['p'],
            attn_p          = cfg['model']['attn_p'],
            global_avg_pool = cfg['model']['global_avg_pool'],
            patch_embed_fun = cfg['model']['patch_embed_fun'],
            pos_embed_type  = cfg['model']['pos_embed_type']
        )
        
        print('ViT3D model built with parameters ')
        print('patch_size: ', cfg['model']['patch_size'])
        print('embed_dim: ', cfg['model']['embed_dim'])
        print('depth: ', cfg['model']['depth'])
        print('n_heads: ', cfg['model']['n_heads'])
        print('mlp_ratio: ', cfg['model']['mlp_ratio'])
        
        
        
    elif cfg.model.arch in ['DenseNet121', 'ResNet10', 'ResNet18', 'ResNet34', 'ResNet101', 'ResNet152']:
        assert cfg.model.arch in _models_factory.keys(), cfg.model.arch + '_' + ' not in the model factory list'

        model = _models_factory[cfg.model.arch](
            spatial_dims     = 3,
            n_input_channels = 1,
            num_classes      = cfg['model']['n_classes']
        )
        
        print(f'Traditional Convolution {cfg["model"]["arch"]} model built.')
    
    elif cfg.model.arch in ['ConvNeXtV2_3D']:
        assert cfg.model.arch in _models_factory.keys(), cfg.model.arch + ' not in the model factory list'
        
        if args.model_size in ['small']:
            cfg['model']['depths'] = [3, 3, 9, 3]
            cfg['model']['dims'] = [32, 64, 128, 256]
        elif args.model_size in ['base']:
            cfg['model']['depths'] = [3, 3, 27, 3]
            cfg['model']['dims'] = [64, 128, 256, 512]
        elif args.model_size in ['large']:
            cfg['model']['depths'] = [3, 3, 27, 3]
            cfg['model']['dims'] = [128, 256, 512, 512]
        elif args.model_size in ['tiny']:
            cfg['model']['depths'] = [2, 2, 6, 2]
            cfg['model']['dims'] = [16, 32, 64, 128]

        model = _models_factory[cfg.model.arch](
            in_chans        = cfg['model']['in_chans'],
            num_classes     = cfg['model']['n_classes'],
            drop_path_rate  = cfg['model']['drop_path_rate'],
            depths          = cfg['model']['depths'],
            dims           = cfg['model']['dims'],
            kernel_size    = cfg['model']['kernel_size'],
            padding        = cfg['model']['kernel_size']//2,
            downsampling   = cfg['model']['downsampling']
        )
        
        print(f'ConvNeXtV2_3D model built with kernel_size={cfg["model"]["kernel_size"]},\
            padding={cfg["model"]["padding"]}, downsampling={cfg["model"]["downsampling"]}')
        
        print(f'Depths: {cfg["model"]["depths"]}, Dims: {cfg["model"]["dims"]}')
        print(f'Drop path rate: {cfg["model"]["drop_path_rate"]}')
        
    elif cfg.model.arch in ['MedNeXt', 'MedNext']:
        model = MedNeXtEncoderOnly(
            in_channels=cfg['model']['in_chans'],
            n_classes=cfg['model']['n_classes'],
            n_channels=cfg['model']['n_channels'],
            exp_r=cfg['model']['exp_r'],
            kernel_size=cfg['model']['kernel_size'],
            deep_supervision=cfg['model']['deep_supervision'],
            do_res=cfg['model']['do_res'],
            do_res_up_down=cfg['model']['do_res_up_down'],
            block_counts=cfg['model']['block_counts'],
        )
        print('kernel_size: ', cfg['model']['kernel_size'])
        print('MedNeXt for Classification model built.')

    return model

def make_pt_model(cfg, args):
    """Build a 3D MAE
    to be used for pre-training
    """
    if cfg.model.arch in ['FCMAE_3D']:
        assert cfg.model.arch  in _models_factory.keys(), \
            f"{cfg.model.arch} not in the model factory list"
        
        if args.size in ['small']:
            cfg.model.depths = [3, 3, 9, 3]
            cfg.model.dims = [32, 64, 128, 256]
        elif args.size in ['base']:
            cfg.model.depths = [3, 3, 27, 3]
            cfg.model.dims = [64, 128, 256, 512]
        elif args.size in ['large']:
            cfg.model.depths = [3, 3, 27, 3]
            cfg.model.dims = [128, 256, 512, 512]
        elif args.size in ['tiny']:
            cfg.model.depths = [2, 2, 6, 2]
            cfg.model.dims = [16, 32, 64, 128]
        
        model = _models_factory[cfg.model.arch](
            # remove 'arch' key and pass all unpacking
            **{key: value for key, value in cfg.model.items() if key != 'arch'}
        )
        
        print('FCMAE_3D model built with parameters ', cfg.model)
    
    elif cfg.model.arch in ['MaskedAutoencoderViT3D']:
        assert cfg.model.arch in _models_factory.keys(), cfg.model.arch + ' not in the model factory list'
        model = _models_factory[cfg.model.arch](
            **{key: value for key, value in cfg.model.items() if key != 'arch'}
        )
        print('MaskedAutoencoderViT3D model built with parameters ', cfg.model)

    return model
