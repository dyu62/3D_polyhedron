# get final graph pkl files from train/val/test ply files
import os
import numpy as np
import two2three
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm
import pickle
from CONST import *


def build_graph(vertices_array, faces,face_attributes):
    # input: single graph attributes
    # output: a heterograph
    def calculate_normal(face, pos):
        # Retrieve positions for the first three vertices A, B, and C
        A = pos[face[0]]
        B = pos[face[1]]
        C = pos[face[2]]

        # Compute vectors AB and BC
        AB = B - A
        BC = C - B

        # Compute the normal vector using the cross product of AB and AC
        normal = torch.cross(AB, BC,dim=-1)
        if torch.norm(normal) == 0:
            # print("same line")
            1
        else:
            normal = normal / torch.norm(normal)
        assert (torch.sum(torch.isnan(normal)) ==0)
        return normal
    def extract_edge_index(faces):
        # Initialize a set to keep edges unique
        # edge_set = set()
        edge_set=[]
        for face_id in range(len(faces)):
            face=faces[face_id]
            n = len(face)
            # Generate edges for a polygon where each vertex connects to the next
            # and the last vertex connects back to the first
            for i in range(n):
                next_index = (i + 1) % n  # Wrap around to the start for the last vertex
                edge = (face[i], face[next_index],face_id)
                # edge_set.add(edge)
                edge_set.append(edge)

        # edge_index = torch.tensor(list(edge_set), dtype=torch.long).t()
        edge_index = torch.tensor((edge_set), dtype=torch.long).t()
        return edge_index
    
    pos = vertices_array
    edge_index = extract_edge_index(faces) 
    faces = faces
    edge_face = torch.zeros((edge_index.size(1)), dtype=torch.long)
    # `edge_face` attribute, can be searched using the following code, but we already recorded in edge_index[2,:] 
    # for face_index, face in enumerate(faces):
    #     for i in range(len(face)):
    #         # Identify edge in edge_index
    #         start_node = face[i]
    #         # Ensures circular connection in the face
    #         end_node = face[(i + 1) % len(face)]
    #         edge_position = (edge_index[0] == start_node) & (
    #             edge_index[1] == end_node)
    #         edge_face[edge_position] = face_index
    edge_face=edge_index[2,:]
    edge_index=edge_index[0:2,:]
    # `face_norm` attribute
    face_norm = [calculate_normal(face, pos) for face in faces]
    face_norm = torch.stack(face_norm)
    # New data structure without `vertices` and `(vertices, inside, vertices)`
    new_data = HeteroData()
    new_data.pos = pos
    new_data['vertices'].x=pos
    new_data['vertices', 'to', 'vertices'].edge_index = edge_index
    new_data.face_list = faces
    new_data.edge_face = edge_face
    new_data.face_norm = face_norm
    if face_attributes is not None:
        new_data['face'].x=torch.cat([face_norm, face_attributes],1)
    else: 
        new_data['face'].x=face_norm

    new_data['edge'].x=torch.zeros(edge_index.shape[1],1) 
    new_data['edge', 'on', 'face'].edge_index=torch.vstack([torch.arange(len(edge_face)),edge_face])
    return new_data

def read_ply_shapenet(file_path):
    # Initialize containers for vertices and faces
    vertices = []
    faces = []
    material_indices = []
    materials = []

    # Reading state flags
    in_vertex_section = False
    in_face_section = False
    in_material_section = False
    num_vertices,num_faces,num_materials=100,100,100
    # Read the file
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            elif line.startswith('element face'):
                num_faces = int(line.split()[-1])
            elif line.startswith('element material'):
                num_materials = int(line.split()[-1])
            elif line == 'end_header':
                in_vertex_section = True
            elif in_vertex_section:
                num_vertices-=1
                vertices.append(list(map(float, line.split())))
                if num_vertices==0:
                    in_vertex_section=False
                    in_face_section=True
            elif in_face_section:
                num_faces-=1
                # Extract face vertex indices and material index
                parts = line.split()
                num_vertices_in_face = int(parts[0])
                if num_vertices_in_face>=3:
                    face_vertex_indices = list(map(int, parts[1:num_vertices_in_face+1]))
                    material_index = int(parts[num_vertices_in_face+1])
                    faces.append(face_vertex_indices)
                    material_indices.append(material_index)
                if num_faces==0:
                    in_face_section=False  
                    in_material_section=True              
            elif in_material_section:
                num_materials-=1
                materials.append(list(map(float, line.split())))

    # Convert lists to appropriate numpy arrays
    vertices_array = torch.tensor(vertices,dtype=torch.float32)
    material_indices_array = torch.tensor(material_indices)
    materials=torch.tensor(materials,dtype=torch.float32)
    return vertices_array, faces, material_indices_array, materials

def read_ply_mnist_color(file_path):
    # Initialize containers for vertices and faces
    vertices = []
    faces = []
    face_attributes=[]

    # Reading state flags
    in_vertex_section = False
    in_face_section = False
    num_vertices,num_faces=100,100
    # Read the file
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            elif line.startswith('element face'):
                num_faces = int(line.split()[-1])
            elif line == 'end_header':
                in_vertex_section = True
            elif in_vertex_section:
                num_vertices-=1
                vertices.append(list(map(float, line.split())))
                if num_vertices==0:
                    in_vertex_section=False
                    in_face_section=True
            elif in_face_section:
                num_faces-=1
                parts = line.split()
                num_vertices_in_face = int(parts[0])
                face_vertex_indices = list(map(int, parts[1:num_vertices_in_face+1]))
                faces.append(face_vertex_indices)
                face_attributes.append(list(map(float, parts[num_vertices_in_face+1:])))

    # Convert lists to appropriate numpy arrays
    vertices_array = torch.tensor(vertices,dtype=torch.float32)
    face_attributes = torch.tensor(face_attributes,dtype=torch.float32)
    
    return vertices_array, faces, face_attributes

def read_ply_building(file_path):
    # Initialize containers for vertices and faces
    vertices = []
    faces = []

    # Reading state flags
    in_vertex_section = False
    in_face_section = False
    num_vertices,num_faces=100,100
    # Read the file
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            elif line.startswith('element face'):
                num_faces = int(line.split()[-1])
            elif line == 'end_header':
                in_vertex_section = True
            elif in_vertex_section:
                num_vertices-=1
                vertices.append(list(map(float, line.split())))
                if num_vertices==0:
                    in_vertex_section=False
                    in_face_section=True
            elif in_face_section:
                num_faces-=1
                parts = line.split()
                num_vertices_in_face = int(parts[0])
                face_vertex_indices = list(map(int, parts[1:num_vertices_in_face+1]))
                faces.append(face_vertex_indices)

    # Convert lists to appropriate numpy arrays
    vertices_array = torch.tensor(vertices,dtype=torch.float32)
    return vertices_array, faces

def read_dir_modelnet(base_dir,dsname="a"):
    class_files = []
    label_mapping = modelnet_class_to_int
    # Walk through the directory
    file_names = os.listdir(base_dir)
    for file_name in tqdm(file_names, desc=f"Processing {dsname}"):
        if file_name.endswith('.ply'):
            file_path = os.path.join(base_dir, file_name)
            last_part = file_name.split('_')[-1].split(".")[0]
            label = label_mapping[last_part]
            vertices_array, faces=read_ply_building(file_path)
            graphdata=build_graph(vertices_array, faces,None)
            graphdata.y=label
            graphdata.file_path=file_path
            class_files.append(graphdata)
    return class_files

def read_dir_shapenet(base_dir,dsname="a"):
    class_files = []
    label_mapping = shapenet_class_to_int
    # Walk through the directory
    file_names = os.listdir(base_dir)
    for file_name in tqdm(file_names, desc=f"Processing {dsname}"):
        if file_name.endswith('.ply'):
            file_path = os.path.join(base_dir, file_name)
            last_part = file_name.split('_')[-1].split(".")[0]
            label = label_mapping[last_part]
            vertices_array, faces, material_indices_array, materials=read_ply_shapenet(file_path)
            graphdata=build_graph(vertices_array, faces,materials[material_indices_array])
            graphdata.y=label
            graphdata.file_path=file_path
            class_files.append(graphdata)
    return class_files

def read_dir_mnist_color(base_dir,dsname="a"):
    class_files = []
    # Walk through the directory
    file_names = os.listdir(base_dir)
    for file_name in tqdm(file_names, desc=f"Processing {dsname}"):
        if file_name.endswith('.ply'):
            file_path = os.path.join(base_dir, file_name)
            last_part = file_name.split('_')[-1]
            label = int(last_part.split('.')[0])
            vertices_array, faces, face_attributes=read_ply_mnist_color(file_path)
            graphdata=build_graph(vertices_array, faces,face_attributes)
            graphdata.y=label
            graphdata.file_path=file_path
            class_files.append(graphdata)
    return class_files

def read_dir_building(base_dir,dsname="a"):
    class_files = []
    # Walk through the directory
    file_names = os.listdir(base_dir)
    for file_name in tqdm(file_names, desc=f"Processing {dsname}"):
        if file_name.endswith('.ply'):
            file_path = os.path.join(base_dir, file_name)
            last_part = file_name.split('_')[-1]
            label = single_label_mapping[last_part.split('.')[0]] 
            vertices_array, faces=read_ply_building(file_path)
            graphdata=build_graph(vertices_array, faces,None)
            graphdata.y=label
            graphdata.file_path=file_path
            class_files.append(graphdata)
    return class_files
            
# Example usage
if __name__ == '__main__':
    ds="shapenet" # mnist_color shapenet shapenet modelnet
    if ds=="shapenet": 
        base_directory = os.path.join('data',"shapenet")
        for folder in ["train","test","val"]:
            directory=os.path.join(base_directory,folder)
            all_classes_files = read_dir_shapenet(directory,dsname=ds)
            with open(os.path.join('data',ds+"_graph_"+folder+".pkl"), 'wb') as handle: 
                pickle.dump(all_classes_files, handle)           
    elif ds =="modelnet":
        base_directory = os.path.join('data',"modelnet")
        for folder in ["train","test","val"]:
            directory=os.path.join(base_directory,folder)
            all_classes_files = read_dir_modelnet(directory,dsname=ds)
            with open(os.path.join('data',ds+"_graph_"+folder+".pkl"), 'wb') as handle: 
                pickle.dump(all_classes_files, handle)           
    elif ds == "mnist_color":
        base_directory = os.path.join('data',ds) 
        for folder in ["train","test","val"]:
            directory=os.path.join(base_directory,folder)
            all_classes_files = read_dir_mnist_color(directory,dsname=ds)
            with open(os.path.join('data',ds+"_graph_"+folder+".pkl"), 'wb') as handle: 
                pickle.dump(all_classes_files, handle)
    elif ds == "building":
        base_directory = os.path.join('data',ds) 
        for folder in ["train","test","val"]:
            directory=os.path.join(base_directory,folder)
            all_classes_files = read_dir_building(directory,dsname=ds)
            with open(os.path.join('data',ds+"_graph_"+folder+".pkl"), 'wb') as handle: 
                pickle.dump(all_classes_files, handle)                    
    print("done")
