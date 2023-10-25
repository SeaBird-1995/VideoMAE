'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-03-02 22:13:22
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''
import sys
sys.path.append("./")
sys.path.append("../")
import torch

from easydict import EasyDict
from collections import OrderedDict

import utils

import modeling_pretrain
from timm.models import create_model
from masking_generator import TubeMaskingGenerator, FrameMaskingGenerator

masking_generator = TubeMaskingGenerator((8, 14, 14), 0.5)
mask = masking_generator()
mask = torch.from_numpy(mask)[None]
mask = mask.flatten(1).to(torch.bool)
print(mask.shape, mask.dtype)

def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        tubelet_size=1
    )
    return model


def test_pretrain_vipc_mae_base_patch16_224():
    args = EasyDict(model="pretrain_vipc_mae_base_patch16_224",
                    drop_path=0.0,
                    decoder_depth=4)

    model = get_model(args)

    B = 5
    x = torch.randn(B, 3, 8, 224, 224) # (B, C, T, H, W)

    window_size = (8, 14, 14)
    masked_position_generator = FrameMaskingGenerator(window_size, 0.125)
    mask = torch.from_numpy(masked_position_generator()[None])
    print(mask.shape)
    mask = mask.expand(B, -1)
    mask = mask.flatten(1).to(torch.bool)

    cam_view_pos = torch.randn(B, 8, 3)


    output = model(x, mask, cam_view_pos)
    print("output: ", output.shape)


def test_vipc_mae_base_patch16_224():
    import modeling_vipc_pretrain

    args = EasyDict(model="pretrain_vipc_self_distillation",
                    drop_path=0.0)

    model = get_model(args)

    B = 5
    x = torch.randn(B, 3, 8, 224, 224) # (B, C, T, H, W)

    window_size = (8, 14, 14)
    masked_position_generator = FrameMaskingGenerator(window_size, 0.125)
    mask = torch.from_numpy(masked_position_generator()[None])
    print(mask.shape)
    mask = mask.expand(B, -1)
    mask = mask.flatten(1).to(torch.bool)

    cam_view_pos = torch.randn(B, 8, 3)


    output = model(x, mask, cam_view_pos)
    print("output: ", output.shape)


def test_finetune_model():
    import modeling_finetune

    args = EasyDict(model="vit_base_patch16_224", nb_classes=400, num_frames=16, num_segments=1,
                    tubelet_size=1, drop=0.0, drop_path=0.1, attn_drop_rate=0.0, 
                    use_mean_pooling=True, init_scale=0.001,
                    finetune="experiments/normalize_vipc_mae_pretrain_base_patch16_224_mask_one_view_total_epoch/checkpoint-74.pth",
                    model_key="model|module", model_prefix="")

    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        all_frames=args.num_frames * args.num_segments,
        tubelet_size=args.tubelet_size,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        drop_block_rate=None,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
    )

    if args.finetune.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            args.finetune, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(args.finetune, map_location='cpu')

    print("Load ckpt from %s" % args.finetune)
    checkpoint_model = None
    for model_key in args.model_key.split('|'):
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print("Load state_dict by model_key = %s" % model_key)
            break
    if checkpoint_model is None:
        checkpoint_model = checkpoint
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    all_keys = list(checkpoint_model.keys())
    new_dict = OrderedDict()
    for key in all_keys:
        if key.startswith('backbone.'):
            new_dict[key[9:]] = checkpoint_model[key]
        elif key.startswith('encoder.'):
            new_dict[key[8:]] = checkpoint_model[key]
        else:
            new_dict[key] = checkpoint_model[key]
    checkpoint_model = new_dict

    # interpolate position embedding
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1] # channel dim
        num_patches = model.patch_embed.num_patches # 
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches # 0/1

        # height (== width) for the checkpoint position embedding 
        orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens)//(args.num_frames // model.patch_embed.tubelet_size)) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int((num_patches // (args.num_frames // model.patch_embed.tubelet_size) )** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            # B, L, C -> BT, H, W, C -> BT, C, H, W
            pos_tokens = pos_tokens.reshape(-1, args.num_frames // model.patch_embed.tubelet_size, orig_size, orig_size, embedding_size)
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, args.num_frames // model.patch_embed.tubelet_size, new_size, new_size, embedding_size) 
            pos_tokens = pos_tokens.flatten(1, 3) # B, L, C
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

    utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)


if __name__ == "__main__":
    test_vipc_mae_base_patch16_224()