'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-03-03 11:30:51
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

import sys
sys.path.append("./")
sys.path.append("../")

import os
from torchvision import transforms
import os.path
from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import pickle
import random
import math 
from tqdm import tqdm
from masking_generator import FrameMaskingGenerator


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def rotation_z(pts, theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.array([[cos_theta, -sin_theta, 0.0],
                                [sin_theta, cos_theta, 0.0],
                                [0.0, 0.0, 1.0]])
    return pts @ rotation_matrix.T


def rotation_y(pts, theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.array([[cos_theta, 0.0, -sin_theta],
                                [0.0, 1.0, 0.0],
                                [sin_theta, 0.0, cos_theta]])
    return pts @ rotation_matrix.T


def rotation_x(pts, theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.array([[1.0, 0.0, 0.0],
                                [0.0, cos_theta, -sin_theta],
                                [0.0, sin_theta, cos_theta]])
    return pts @ rotation_matrix.T

class ViPCDataLoader(Dataset):
    def __init__(self,filepath,data_path,status,
                 pc_input_num=3500, view_align=False, category='all',
                 transform=None):
        super(ViPCDataLoader,self).__init__()
        self.pc_input_num = pc_input_num
        self.status = status
        self.filelist = []
        self.cat = []
        self.key = []
        self.category = category
        self.view_align = view_align
        self.cat_map = {
            'airplane':'02691156',
            'bench': '02828884', 
            'cabinet':'02933112', 
            'car':'02958343',
            'chair':'03001627',
            'monitor': '03211117',
            'lamp':'03636649',
            'speaker': '03691459', 
            'firearm': '04090263', 
            'couch':'04256520',
            'table':'04379243',
            'cellphone': '04401088', 
            'watercraft':'04530566'
        }
        with open(filepath,'r') as f:
            line = f.readline()
            while (line):
                self.filelist.append(line)
                line = f.readline()
        
        self.imcomplete_path = os.path.join(data_path,'ShapeNetViPC-Partial')
        self.gt_path = os.path.join(data_path,'ShapeNetViPC-GT')
        self.rendering_path = os.path.join(data_path,'ShapeNetViPC-View')

        for key in self.filelist:
            if category !='all':
                if key.split(';')[0]!= self.cat_map[category]:
                    continue
            self.cat.append(key.split(';')[0])
            self.key.append(key)

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor()
        ])

        print(f'{status} data num: {len(self.key)}')


    def __getitem__(self, idx):
        
        key = self.key[idx]
       
        pc_part_path = os.path.join(self.imcomplete_path,key.split(';')[0]+'/'+ key.split(';')[1]+'/'+key.split(';')[-1].replace('\n', '')+'.dat')
        # view_align = True, means the view of image equal to the view of partial points
        # view_align = False, means the view of image is different from the view of partial points
        if self.view_align:
            ran_key = key        
        else:
            ran_key = key[:-3]+str(random.randint(0,23)).rjust(2,'0')
       
        pc_path = os.path.join(self.gt_path, ran_key.split(';')[0]+'/'+ ran_key.split(';')[1]+'/'+ran_key.split(';')[-1].replace('\n', '')+'.dat')
        view_path = os.path.join(self.rendering_path,ran_key.split(';')[0]+'/'+ran_key.split(';')[1]+'/rendering/'+ran_key.split(';')[-1].replace('\n','')+'.png')

        if not os.path.exists(pc_path):
            # print(f"[WARNING] {pc_path} is not exist....")
            return None
        
        #Inserted to correct a bug in the splitting for some lines 
        if(len(ran_key.split(';')[-1])>3):
            print("bug")
            print(ran_key.split(';')[-1])
            fin = ran_key.split(';')[-1][-2:]
            interm = ran_key.split(';')[-1][:-2]
            
            pc_path = os.path.join(self.gt_path, ran_key.split(';')[0]+'/'+ interm +'/'+ fin.replace('\n', '')+'.dat')
            view_path = os.path.join(self.rendering_path,ran_key.split(';')[0]+ '/' + interm + '/rendering/' + fin.replace('\n','')+'.png')

        views = self.transform(Image.open(view_path))
        views = views[:3,:,:]
        # load partial points
        with open(pc_path,'rb') as f:
            pc = pickle.load(f).astype(np.float32)
        # load gt
        with open(pc_part_path,'rb') as f:
            pc_part = pickle.load(f).astype(np.float32)
        # incase some item point number less than 3500 
        if pc_part.shape[0]<self.pc_input_num:
            pc_part = np.repeat(pc_part,(self.pc_input_num//pc_part.shape[0])+1,axis=0)[0:self.pc_input_num]


        # load the view metadata
        # image_view_id = view_path.split('.')[0].split('/')[-1]
        # part_view_id = pc_part_path.split('.')[0].split('/')[-1]
        ## zhm add:
        image_view_id = os.path.basename(view_path).split('.')[0]
        part_view_id = os.path.basename(pc_part_path).split('.')[0]
        
        view_metadata = np.loadtxt(view_path[:-6]+'rendering_metadata.txt')

        theta_part = math.radians(view_metadata[int(part_view_id),0])
        phi_part = math.radians(view_metadata[int(part_view_id),1])

        theta_img = math.radians(view_metadata[int(image_view_id),0])
        phi_img = math.radians(view_metadata[int(image_view_id),1])

        pc_part = rotation_y(rotation_x(pc_part, - phi_part),np.pi + theta_part)
        pc_part = rotation_x(rotation_y(pc_part, np.pi - theta_img), phi_img)

        # normalize partial point cloud and GT to the same scale
        gt_mean = pc.mean(axis=0) 
        pc = pc -gt_mean
        pc_L_max = np.max(np.sqrt(np.sum(abs(pc ** 2), axis=-1)))
        pc = pc/pc_L_max

        pc_part = pc_part-gt_mean
        pc_part = pc_part/pc_L_max

        return views.float(), torch.from_numpy(pc).float(), torch.from_numpy(pc_part).float()

    def __len__(self):
        return len(self.key)


class ViPCMultiViewDataset(ViPCDataLoader):
    
    def __init__(self, filepath, data_path, status, 
                 pc_input_num=3500, view_align=False, category='all', num_views=8,
                 mask_ratio=0.125):
        super(ViPCMultiViewDataset, self).__init__(
            filepath, data_path, status, pc_input_num, view_align, category)

        ## only containing the unique objects
        self.key = sorted(list(set([k.split(';')[0]+';'+k.split(';')[1] for k in self.key])))
        print(f'{status} unique objects num: {len(self.key)}')
        
        self.num_views = num_views
        
        window_size = (num_views, 14, 14)
        self.masked_position_generator = FrameMaskingGenerator(window_size, mask_ratio)
    
    def __getitem__(self, idx):
        
        key = self.key[idx]
        category_id, object_id = key.strip().split(';')
       
        ### Randomly select several view images
        image_view_ids = random.sample(range(24), self.num_views)

        view_dir = os.path.join(self.rendering_path, category_id, object_id, 'rendering')
        camera_meta_info = np.loadtxt(os.path.join(view_dir, 'rendering_metadata.txt')).astype(np.float32)
        camera_meta_info = np.radians(camera_meta_info[:, [0,1,3]])

        view_imgs = []
        for image_id in image_view_ids:
            image_path = os.path.join(view_dir, f'{image_id:02d}.png')
            view_img = self.transform(Image.open(image_path))[:3, ...]
            view_imgs.append(view_img)
        views = torch.stack(view_imgs, dim=0).to(torch.float32)  # (T, 3, H, W)
        views = views.permute(1, 0, 2, 3)  # to (3, T, H, W)
        view_camera_params = camera_meta_info[image_view_ids]  # (T, 3)

        mask = self.masked_position_generator()
        return {'process_data': (views, mask),
                'view_cam_params': torch.from_numpy(view_camera_params).to(torch.float32)
                } 


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import numpy as np

    multi_view_dataset = ViPCMultiViewDataset(
        'datasets/train_list2.txt', data_path="datasets/ShapeNetViPC-Dataset", 
        status="train", category="all", num_views=5)
    print(len(multi_view_dataset))
    # views, pc_gt, pc_partial = multi_view_dataset[0]
    # print(views.shape, pc_gt.shape, pc_partial.shape)
    # np.savetxt("pc_gt.xyz", pc_gt.numpy())
    # np.savetxt("pc_partial.xyz", pc_partial.numpy())

    train_loader = DataLoader(multi_view_dataset,
                              batch_size=4,
                              num_workers=1,
                              shuffle=True,
                              drop_last=True)
    for batch in tqdm(train_loader):
        process_data = batch['process_data']
        views, mask = process_data
        print(views.shape, mask.shape)

        view_cam_params = batch['view_cam_params']
        print(view_cam_params.shape)
        break
    exit(0)
