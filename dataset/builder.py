'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-03-03 12:09:39
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

from .shapenet_vipc_dataset import ViPCMultiViewDataset

def build_pretraining_dataset(args):
    if args.dataset_type == 'shapenet_vipc':
        dataset = ViPCMultiViewDataset(
            filepath=args.data_path,
            data_path=args.video_root,
            num_views=args.num_frames,
            category='car',
            status='train',
            mask_ratio=args.mask_ratio)
    else:
        raise NotImplementedError
    return dataset

