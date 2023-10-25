'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-03-03 09:02:14
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''


import sys
sys.path.append("./")
sys.path.append("../")

import torch

def test_():
    from masking_generator import TubeMaskingGenerator

    masking_generator = TubeMaskingGenerator((8, 14, 14), 0.5)
    mask = masking_generator()
    print(mask.shape, mask.dtype)
    mask = torch.from_numpy(mask)[None]
    mask = mask.flatten(1).to(torch.bool)

    vis_mask = ~mask
    print(vis_mask.shape, vis_mask.dtype)
    print(vis_mask[:, :10])


def test_FrameMaskingGenerator():
    from masking_generator import FrameMaskingGenerator

    masking_generator = FrameMaskingGenerator((8, 14, 14), 0.25)
    mask = masking_generator()
    print(mask.shape)


def test_vis_dataset():
    from dataset.shapenet_vipc_dataset import ViPCMultiViewDataset

    dataset = ViPCMultiViewDataset(
            filepath="datasets/test_list2.txt",
            data_path="datasets/ShapeNetViPC-Dataset",
            num_views=8,
            category='all',
            status='test',
            mask_ratio=0.125)
    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=16)

    for i, batch in enumerate(test_dataloader):
        process_data = batch['process_data']
        views, mask = process_data
        print(views.shape, mask.shape)

        view_cam_params = batch['view_cam_params']
        print(view_cam_params.shape)
        break



if __name__ == "__main__":
    test_vis_dataset()