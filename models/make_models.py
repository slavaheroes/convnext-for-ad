# -*- coding: utf-8 -*-
"""
Created on Wed Feb 8 2023

@author: qasymjomart
"""

from __future__ import print_function, division

# from make_dataloaders import make_dataloaders
from .convnets import make_densenet1213d, make_resnet103d, make_resnet183d, make_resnet343d, make_resnet1013d, make_resnet1523d
from .convnextv2_3d import ConvNeXtV2_3D
from .fcmae_3d import FCMAE_3D
from .mednext import MedNeXt
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
    if cfg['MODEL']['TYPE'] in ['ViT3D']:
        assert cfg['MODEL']['TYPE'] in _models_factory.keys(), cfg['MODEL']['TYPE'] + ' not in the model factory list'

        model = _models_factory[cfg['MODEL']['TYPE']](
            img_size        = cfg['MODEL']['img_size'],
            patch_size      = cfg['MODEL']['patch_size'],
            in_chans        = cfg['MODEL']['in_chans'],
            n_classes       = cfg['MODEL']['n_classes'],
            embed_dim       = cfg['MODEL']['embed_dim'],
            depth           = cfg['MODEL']['depth'],
            n_heads         = cfg['MODEL']['n_heads'],
            mlp_ratio       = cfg['MODEL']['mlp_ratio'],
            qkv_bias        = cfg['MODEL']['qkv_bias'],
            drop_path_rate  = cfg['MODEL']['drop_path_rate'],
            p               = cfg['MODEL']['p'],
            attn_p          = cfg['MODEL']['attn_p'],
            global_avg_pool = cfg['MODEL']['global_avg_pool'],
            patch_embed_fun = cfg['MODEL']['patch_embed_fun'],
            pos_embed_type  = cfg['MODEL']['pos_embed_type']
        )
        
        print('ViT3D model built with parameters ')
        print('patch_size: ', cfg['MODEL']['patch_size'])
        print('embed_dim: ', cfg['MODEL']['embed_dim'])
        print('depth: ', cfg['MODEL']['depth'])
        print('n_heads: ', cfg['MODEL']['n_heads'])
        print('mlp_ratio: ', cfg['MODEL']['mlp_ratio'])
        
        
        
    elif cfg['MODEL']['TYPE'] in ['DenseNet121', 'ResNet10', 'ResNet18', 'ResNet34', 'ResNet101', 'ResNet152']:
        assert cfg['MODEL']['TYPE'] in _models_factory.keys(), cfg['MODEL']['TYPE'] + '_' + ' not in the model factory list'

        model = _models_factory[cfg['MODEL']['TYPE']](
            spatial_dims     = 3,
            n_input_channels = 1,
            num_classes      = cfg['MODEL']['n_classes']
        )
        
        print(f'Traditional Convolution {cfg["MODEL"]["TYPE"]} model built.')
    
    elif cfg['MODEL']['TYPE'] in ['ConvNeXtV2_3D']:
        assert cfg['MODEL']['TYPE'] in _models_factory.keys(), cfg['MODEL']['TYPE'] + ' not in the model factory list'
        
        if args.size in ['small']:
            cfg['MODEL']['depths'] = [3, 3, 9, 3]
            cfg['MODEL']['dims'] = [32, 64, 128, 256]
        elif args.size in ['base']:
            cfg['MODEL']['depths'] = [3, 3, 27, 3]
            cfg['MODEL']['dims'] = [64, 128, 256, 512]
        elif args.size in ['large']:
            cfg['MODEL']['depths'] = [3, 3, 27, 3]
            cfg['MODEL']['dims'] = [128, 256, 512, 512]
        elif args.size in ['tiny']:
            cfg['MODEL']['depths'] = [2, 2, 6, 2]
            cfg['MODEL']['dims'] = [16, 32, 64, 128]
            
        model = _models_factory[cfg['MODEL']['TYPE']](
            in_chans        = cfg['MODEL']['in_chans'],
            num_classes     = cfg['MODEL']['n_classes'],
            drop_path_rate  = cfg['MODEL']['drop_path_rate'],
            depths          = cfg['MODEL']['depths'],
            dims           = cfg['MODEL']['dims'],
            kernel_size    = cfg['MODEL']['kernel_size'],
            padding        = cfg['MODEL']['kernel_size']//2,
            downsampling   = cfg['MODEL']['downsampling']
        )
        
        print(f'ConvNeXtV2_3D model built with kernel_size={cfg["MODEL"]["kernel_size"]},\
            padding={cfg["MODEL"]["padding"]}, downsampling={cfg["MODEL"]["downsampling"]}')
        
        print(f'Depths: {cfg["MODEL"]["depths"]}, Dims: {cfg["MODEL"]["dims"]}')
        print(f'Drop path rate: {cfg["MODEL"]["drop_path_rate"]}')
        
    elif cfg['MODEL']['TYPE'] in ['MedNeXt', 'MedNext']:
        model = MedNeXtEncoderOnly(
            in_channels=cfg['MODEL']['in_chans'],
            n_classes=cfg['MODEL']['n_classes'],
            n_channels=cfg['MODEL']['n_channels'],
            exp_r=cfg['MODEL']['exp_r'],
            kernel_size=cfg['MODEL']['kernel_size'],
            deep_supervision=cfg['MODEL']['deep_supervision'],
            do_res=cfg['MODEL']['do_res'],
            do_res_up_down=cfg['MODEL']['do_res_up_down'],
            block_counts=cfg['MODEL']['block_counts'],
        )
        print('kernel_size: ', cfg['MODEL']['kernel_size'])
        print('MedNeXt for Classification model built.')

    return model

def make_mae_model(cfg, args):
    """Build a 3D MAE
    to be used for pre-training
    """
    if cfg['MODEL']['TYPE'] in ['FCMAE_3D']:
        assert cfg['MODEL']['TYPE'] in _models_factory.keys(), cfg['MODEL']['TYPE'] + ' not in the model factory list'
        
        if args.size in ['small']:
            cfg['MODEL']['depths'] = [3, 3, 9, 3]
            cfg['MODEL']['dims'] = [32, 64, 128, 256]
        elif args.size in ['base']:
            cfg['MODEL']['depths'] = [3, 3, 27, 3]
            cfg['MODEL']['dims'] = [64, 128, 256, 512]
        elif args.size in ['large']:
            cfg['MODEL']['depths'] = [3, 3, 27, 3]
            cfg['MODEL']['dims'] = [128, 256, 512, 512]
        elif args.size in ['tiny']:
            cfg['MODEL']['depths'] = [2, 2, 6, 2]
            cfg['MODEL']['dims'] = [16, 32, 64, 128]
        
        model_mae = _models_factory[cfg['MODEL']['TYPE']](
            size                = cfg['MODEL']['img_size'][0],
            in_chans          = cfg['MODEL']['in_chans'],
            depths              = cfg['MODEL']['depths'],
            dims                = cfg['MODEL']['dims'],
            patch_size          = cfg['MODEL']['patch_size'],
            mask_ratio          = cfg['MODEL']['mask_ratio'],
            decoder_embed_dim   = cfg['MODEL']['decoder_embed_dim'],
            decoder_depth       = cfg['MODEL']['decoder_depth'],
            norm_pix_loss       = cfg['MODEL']['norm_pix_loss'],
            encoder_type        = cfg['MODEL']['encoder_type'],
            kernel_size         = cfg['MODEL']['kernel_size'],
            padding             = cfg['MODEL']['kernel_size']//2,
            downsampling        = cfg['MODEL']['downsampling']
        )

        print('MAE ', cfg['MODEL']['TYPE'], ' model built with parameters ')
        print('Size: ', cfg['MODEL']['img_size'])
        print('In channels: ', cfg['MODEL']['in_chans'])
        print('Depths: ', cfg['MODEL']['depths'])
        print('Dims: ', cfg['MODEL']['dims'])
        print('Patch size: ', cfg['MODEL']['patch_size'])
        print('Mask ratio: ', cfg['MODEL']['mask_ratio'])
        print('Decoder embed dim: ', cfg['MODEL']['decoder_embed_dim'])
        print('Decoder depth: ', cfg['MODEL']['decoder_depth'])
        print('Norm pix loss: ', cfg['MODEL']['norm_pix_loss'])
        print('Encoder type: ', cfg['MODEL']['encoder_type'])
        print('Kernel size: ', cfg['MODEL']['kernel_size'])
        print('Padding: ', cfg['MODEL']['padding'])
        print('Downsampling: ', cfg['MODEL']['downsampling'])
    elif cfg['MODEL']['TYPE'] in ['MaskedAutoencoderViT3D']:
        assert cfg['MODEL']['TYPE'] in _models_factory.keys(), cfg['MODEL']['TYPE'] + ' not in the model factory list'

        model_mae = _models_factory[cfg['MODEL']['TYPE']](
            img_size          = cfg['MODEL']['img_size'],
            patch_size        = cfg['MODEL']['patch_size'], 
            in_chans          = cfg['MODEL']['in_chans'],
            embed_dim         = cfg['MODEL']['embed_dim'], 
            depth             = cfg['MODEL']['depth'], 
            num_heads         = cfg['MODEL']['n_heads'],
            qkv_bias          = cfg['MODEL']['qkv_bias'],
            drop_path_rate    = cfg['MODEL']['drop_path_rate'],
            decoder_embed_dim = cfg['MODEL']['decoder_embed_dim'], 
            decoder_depth     = cfg['MODEL']['decoder_depth'], 
            decoder_num_heads = cfg['MODEL']['decoder_num_heads'],
            mlp_ratio         = cfg['MODEL']['mlp_ratio'], 
            norm_pix_loss     = cfg['MODEL']['norm_pix_loss'],
            patch_embed_fun   = 'conv3d'
        )

        print('MAE ', cfg['MODEL']['TYPE'], ' model built.')

    return model_mae
