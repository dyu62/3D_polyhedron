    
# data['face'].x=data.face_norm
# data['edge'].x=torch.zeros(data.edge_index.shape[1],1) #edge没有属性，属性其实是node但是node index是会变的，不能这里指定

# #只要记住第二个数是所在face数就行，会自动变，第一个数其实是第几个edge就是第几个
# data['edge', 'on', 'face'].edge_index=torch.vstack([torch.arange(len(data.edge_face)),data.edge_face]) 
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import pandas as pd
import sys
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
from torchvision.datasets import VisionDataset
from torch.utils.data import Dataset
import random
import pickle as pkl
from tqdm import tqdm
from util import *


def get_dataset(data_dir='?'):
    train_path=data_dir+"_train.pkl"
    val_path=data_dir+"_val.pkl"
    test_path=data_dir+"_test.pkl"
    with open(train_path, 'rb') as f:
        train_ds = pkl.load(f)
        np.random.shuffle(train_ds)
    with open(val_path, 'rb') as f:
        val_ds = pkl.load(f)
        np.random.shuffle(val_ds)
    with open(test_path, 'rb') as f:
        test_ds = pkl.load(f)
        np.random.shuffle(test_ds)        
    # val_test_split = int(np.around( test_ratio * len(dataset) ))
    # train_val_split = int(len(dataset)-2*val_test_split)
    # train_ds = dataset[:train_val_split]
    # val_ds = dataset[train_val_split:train_val_split+val_test_split]
    # test_ds = dataset[train_val_split+val_test_split:]  

    print(data_dir)
    print('Train: ' +str(len(train_ds)))
    print('Val  : ' +str(len(val_ds)))
    print('Test : ' +str(len(test_ds)))  
    return train_ds,val_ds,test_ds

def affine_transform_to_range(ds, target_range=(-1, 1)):
    # Find the extent (min and max) of coordinates in both x and y directions
    for item in ds:
        min_x  = torch.min(item.pos[:,0])
        min_y  = torch.min(item.pos[:,1])
        min_z  = torch.min(item.pos[:,2])
        
        max_x  = torch.max(item.pos[:,0])
        max_y  = torch.max(item.pos[:,1])
        max_z  = torch.max(item.pos[:,2])
        
        scale_x = (target_range[1] - target_range[0]) / (max_x - min_x)
        scale_y = (target_range[1] - target_range[0]) / (max_y - min_y)
        scale_z = (target_range[1] - target_range[0]) / (max_z - min_z)
        translate_x = target_range[0] - min_x * scale_x
        translate_y = target_range[0] - min_y * scale_y
        translate_z = target_range[0] - min_z * scale_z

        # Apply the affine transformation to 
        item.pos[:,0] = item.pos[:,0] * scale_x + translate_x
        item.pos[:,1] = item.pos[:,1] * scale_y + translate_y
        item.pos[:,2] = item.pos[:,2] * scale_z + translate_z
    return ds

def rotate_points(points, face_norm, angles):
    """
    Rotate points around their centroid by given angles (in radians) for x, y, and z axes using PyTorch.
    :param points: (N, 3) tensor of 3D points
    :param angles: tuple of angles (angle_x, angle_y, angle_z)
    :return: rotated points as (N, 3) tensor
    """
    def rotation_matrix_x(angle):
        """ Create a rotation matrix for rotating around the x-axis """
        return torch.tensor([
            [1, 0, 0],
            [0, torch.cos(angle), -torch.sin(angle)],
            [0, torch.sin(angle), torch.cos(angle)]
        ], dtype=points.dtype, device=points.device)

    def rotation_matrix_y(angle):
        return torch.tensor([
            [torch.cos(angle), 0, torch.sin(angle)],
            [0, 1, 0],
            [-torch.sin(angle), 0, torch.cos(angle)]
        ], dtype=points.dtype, device=points.device)

    def rotation_matrix_z(angle):
        return torch.tensor([
            [torch.cos(angle), -torch.sin(angle), 0],
            [torch.sin(angle), torch.cos(angle), 0],
            [0, 0, 1]
        ], dtype=points.dtype, device=points.device)
    
    # Calculate the centroid of the points
    centroid = torch.mean(points, dim=0)
    
    # Subtract centroid to move the center of the object to the origin
    centered_points = points - centroid
    
    # Create rotation matrices
    Rx = rotation_matrix_x(angles[0])
    Ry = rotation_matrix_y(angles[1])
    Rz = rotation_matrix_z(angles[2])
    
    # Combine rotations: First rotate around Z, then Y, then X
    R = Rx @ Ry @ Rz
    
    # Apply the rotation to the points
    rotated_points = centered_points @ R.T
    rotated_normals=face_norm@ R.T
    # Add the centroid back to move the object back to its original position
    rotated_points += centroid
    return rotated_points,rotated_normals

def rotate_ds(ds):
    angles = torch.tensor(np.random.rand(len(ds),3) * torch.pi,dtype=torch.float)
    for i,item in enumerate(ds):
        item.pos,item.face_norm = rotate_points(item.pos,item.face_norm, angles[i//len(angles)])
    return ds



if __name__ == '__main__':
    # get rotated mnist
    # data_dir=os.path.join("data","mnist_color_graph")
    # train_ds,val_ds,test_ds=get_dataset(data_dir)
    # train_ds,val_ds,test_ds=rotate_ds(train_ds),rotate_ds(val_ds),rotate_ds(test_ds)
    # def save_dataset(dataset, filename):
    #     with open(filename, 'wb') as f:
    #         pkl.dump(dataset, f)
    # save_dataset(train_ds, os.path.join("data",'mnist_color_r_train.pkl') )
    # save_dataset(val_ds, os.path.join("data",'mnist_color_r_val.pkl') )
    # save_dataset(test_ds, os.path.join("data",'mnist_color_r_test.pkl') )
    
    # with open(os.path.join("data",'mnist_color_r_val.pkl'), 'rb') as f:
    #     train_ds = pkl.load(f)  
    #     1 
    
    # generate 50000 3d angles for augmentation
    np.random.seed(42) 
    angles = torch.tensor(np.random.rand(50000,3) * torch.pi,dtype=torch.float) #[1.1767, 2.9868, 2.2996]...
    data_dir=os.path.join("data","modelnet_graph")#modelnet_graph mnist_color_graph
    train_ds,val_ds,test_ds=get_dataset(data_dir)
    1
                        
    # Test connectivity
    # a,b,c=get_mnist_dataset(data_dir='data/mnist_colorgraph.pkl')
    # dsname="bnet"
    # a,b,c=get_bnet_dataset(data_dir="data/bnet_sgraph.pkl")
    # dsname="shapenet"
    # a,b,c=get_bnet_dataset(data_dir="data/shapenet_sgraph.pkl")
    # if dsname in ['bnet']:
    #     category_count = { 'RELIGIOUS': 0, 'RESIDENTIAL': 0}
    # else:
    #     shapenet_class_names_must = [
    #         "chair", "display",  
    #         "loudspeaker", "sofa", "table" 
    #     ]
    #     shapenet_class_names = ["bathtub",  "bench", "bookshelf", "bottle", "cabinet", "cellular telephone","file","knife","lamp","laptop","pot","vessel"]
    #     shapenet_class_names = [ "chair",   
    # "loudspeaker", "table" ,"bathtub",  "bench", "bottle","cellular telephone","file","knife","lamp","laptop","pot","vessel"]

    #     category_count =shapenet_class_names_must+shapenet_class_names
    #     category_count = {category: 0 for category in category_count}
    # category_count_total=copy.deepcopy(category_count)
    # count=0
    # names=[]
    # for sample in tqdm(a):
    #     edge_index = sample['vertices','to','vertices']['edge_index']
    #     hsample = Data(edge_index=edge_index)
    #     parts = sample['file_path'].split('/')
    #     category = parts[2]
    #     if category in category_count:
    #         category_count_total[category] += 1
    #     if is_connected(hsample):
    #         continue
    #     else:
    #         count+=1
    #         names.append(sample['file_path'])
    
    # for filename in names:
    #     # Split the filename to extract the category
    #     parts = filename.split('/')
    #     category = parts[2]  # Assuming the category is always in the same position based on the example provided

    #     # Update count in the dictionary
    #     if category in category_count:
    #         category_count[category] += 1

    # # Print the results
    # for category, count in category_count.items():
    #     print(f"{category}: {count/category_count_total[category]}")
    # print("")
    
