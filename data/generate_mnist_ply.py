# generate blank 3d mnist ply files from 2d graph
import os
import os.path
import torch
import pickle as pkl
import copy
import two2three
with open('data/single_mnist.pkl', 'rb') as f:
    dataset = pkl.load(f)
def unique_integers(list_of_lists):
    for sublist in list_of_lists:
        if len(sublist) != len(set(sublist))+1:
            return False
    return True

def valid_check(list_of_lists):
    for sublist in list_of_lists:
        if sublist[0] != sublist[-1]:
            return False
    return True
path="data/mnist"
processed_ds=[]
if not os.path.exists(path):
    os.makedirs(path,exist_ok=True)
written_count=0
corrected=0
for i in range(len(dataset)):
    data=two2three.extrude_shape(copy.deepcopy(dataset[i]))
    if unique_integers(data["faces"]) and valid_check(data["faces"]):
        data=two2three.add_attributes(data)
        data.y=dataset[i].y
        data['vertices'].x=data.pos
        data['vertices', 'to', 'vertices'].edge_index = data.edge_index
        data['face'].x=data.face_norm
        data['edge'].x=torch.zeros(data.edge_index.shape[1],1) #edge do not have attribute
        data['edge', 'on', 'face'].edge_index=torch.vstack([torch.arange(len(data.edge_face)),data.edge_face])  
        two2three.write_ply(data,os.path.join(path,f"dataset_{i}_{written_count}_{int(dataset[i].y)}.ply"))
        written_count+=1
        del data.edge_index
        processed_ds.append(data)
    else: #try to correct it
        data=(copy.deepcopy(dataset[i]))
        data[('vertices', 'inside', 'vertices')].edge_index= data[('vertices', 'inside', 'vertices')].edge_index.flip(0)
        data=two2three.extrude_shape(data)
        if unique_integers(data["faces"]) and valid_check(data["faces"]):
            data=two2three.add_attributes(data)
            data.y=dataset[i].y
            data['vertices'].x=data.pos
            data['vertices', 'to', 'vertices'].edge_index = data.edge_index
            data['face'].x=data.face_norm
            data['edge'].x=torch.zeros(data.edge_index.shape[1],1)
            data['edge', 'on', 'face'].edge_index=torch.vstack([torch.arange(len(data.edge_face)),data.edge_face])          
        
            two2three.write_ply(data,os.path.join(path,f"dataset_{i}_{written_count}_{int(dataset[i].y)}.ply"))
            written_count+=1
            corrected+=1
            del data.edge_index
            processed_ds.append(data)
        else:
            # print(f"fail_{i}")
            pass
print(written_count,corrected)
print("done")