# simplify mesh ply to get polyhedron format
import os
import numpy as np
from collections import defaultdict
import concurrent.futures
from tqdm import tqdm
initial_max_faces=1600
max_face=400

def read_mtl(filename):
    materials = {}
    current_material = None
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                if line.startswith('newmtl'):
                    current_material = line.split()[1]
                    materials[current_material] = {}
                elif line.startswith(('Ka', 'Kd', 'Ks', 'Ke', 'Ns', 'd', 'illum')):
                    key, *values = line.split()
                    materials[current_material][key] = [float(v) for v in values]
    except FileNotFoundError:
        print(f"Warning: MTL file not found: {filename}")
        return None
    return materials


def read_obj(file_path, initial_max_faces):
    vertices = []
    faces = defaultdict(list)
    normals = []
    vertex_normals = defaultdict(list)
    face_materials = []  
    current_material = None
    mtllib = None
    total_faces = 0

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            if not parts:
                continue
            if line.startswith("v "):
                vertices.append(list(map(float, parts[1:])))
            elif line.startswith("vn "):
                normals.append(list(map(float, parts[1:])))
            elif line.startswith("usemtl"):
                current_material = parts[1]
            elif line.startswith("f "):
                if len(parts) < 2:
                    continue
                if "//" in parts[1]:  # v//vn
                    face = [int(p.split("//")[0]) - 1 for p in parts[1:]]  # vertex index
                    vn_indices = [int(p.split("//")[1]) - 1 for p in parts[1:]]  # normal index
                else:  # v/vt/vn
                    face = [int(p.split("/")[0]) - 1 for p in parts[1:]]  # vertex index
                    vn_indices = [int(p.split("/")[-1]) - 1 for p in parts[1:]]  # normal index
                faces[current_material].append(face)
                vertex_normals[current_material].append(vn_indices)
                face_materials.append(current_material)  # save all face material
                total_faces += 1
                if total_faces > initial_max_faces:
                    return None, None, None, None, None, None  
            elif line.startswith("mtllib"):
                mtllib = line.strip()

    return np.array(vertices), faces, normals, vertex_normals, face_materials, mtllib


def calculate_face_center(vertices, face):
    face_vertices = np.array([vertices[idx] for idx in face])
    centroid = np.mean(face_vertices, axis=0)
    return centroid


def ray_intersects_polygon(ray_origin, ray_direction, polygon_vertices):
    if len(polygon_vertices) < 3:
        return False

    v0, v1, v2 = polygon_vertices[:3]
    normal = np.cross(v1 - v0, v2 - v0)
    normal_length = np.linalg.norm(normal)
    if normal_length == 0:
        return False
    normal = normal / normal_length

    denom = np.dot(normal, ray_direction)
    if abs(denom) < 1e-6:
        return False

    d = np.dot(normal, v0)
    t = (d - np.dot(normal, ray_origin)) / denom
    if t < 0:
        return False

    intersect_point = ray_origin + t * ray_direction

    def is_point_in_polygon(point, polygon):
        winding_number = 0
        for i in range(len(polygon)):
            v1 = polygon[i] - point
            v2 = polygon[(i + 1) % len(polygon)] - point
            angle = np.arctan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v1, v2))
            if np.dot(np.cross(v1, v2), normal) > 0:
                winding_number += angle
            else:
                winding_number -= angle
        return abs(winding_number) > 1e-6

    return is_point_in_polygon(intersect_point, polygon_vertices)


def is_external_face(vertices, faces, face_index):
    face = faces[face_index]
    if len(face) < 3:
        return False

    face_normal = np.cross(vertices[face[1]] - vertices[face[0]], vertices[face[2]] - vertices[face[0]])
    normal_length = np.linalg.norm(face_normal)
    if normal_length == 0:
        return False
    face_normal = face_normal / normal_length
    face_center = calculate_face_center(vertices, face)

    ray_origin = face_center + face_normal * 1e-3
    ray_direction = face_normal

    num_intersections = 0
    for i, other_face in enumerate(faces):
        if i != face_index:
            if ray_intersects_polygon(ray_origin, ray_direction, [vertices[idx] for idx in other_face]):
                num_intersections += 1

    # print(f"Face {face_index}, Direction {face_normal}, Intersections {num_intersections}")

    return num_intersections // 2 == 0 or num_intersections // 2 % 2 == 0


def filter_outside_faces(vertices, faces):
    outside_faces = []
    outside_faces_indices = []
    if len(faces)>max_face:
        return None,None
    for i in range(len(faces)):
        if is_external_face(vertices, faces, i):
            outside_faces.append(faces[i])
            outside_faces_indices.append(i)
    return outside_faces, outside_faces_indices


def find_duplicate_faces(inside_faces):
    face_sets = [set(face) for face in inside_faces]
    duplicate_faces = []
    duplicate_indices = []
    seen = set()
    for i, face_set in enumerate(face_sets):
        if tuple(face_set) in seen:
            continue
        for j in range(i + 1, len(face_sets)):
            if face_set == face_sets[j]:
                duplicate_faces.append(inside_faces[i])
                duplicate_faces.append(inside_faces[j])
                duplicate_indices.append(i)
                duplicate_indices.append(j)
                seen.add(tuple(face_set))
                seen.add(tuple(face_sets[j]))
                break
    return duplicate_faces, duplicate_indices


def build_shared_edge_map(faces, vertex_normals):
    shared_edges = defaultdict(list)
    for i, face in enumerate(faces):
        for j in range(len(face)):
            edge = (face[j], face[(j + 1) % len(face)])
            vn_edge = (vertex_normals[i][j], vertex_normals[i][(j + 1) % len(face)])
            shared_edges[(edge, vn_edge)].append(i)
            shared_edges[((face[(j + 1) % len(face)], face[j]),
                          (vertex_normals[i][(j + 1) % len(face)], vertex_normals[i][j]))].append(i)
    return shared_edges


def calculate_normal(v1, v2, v3):
    edge1 = v2 - v1
    edge2 = v3 - v2
    normal = np.cross(edge1, edge2)
    if np.linalg.norm(normal) != 0:
        return normal / np.linalg.norm(normal)
    return None

def are_faces_coplanar(vertices,  face1, face2):
    v1 = vertices[face1]
    v2 = vertices[face2]

    # get first normal
    normal1 = None
    for i in range(len(v1)):
        normal1 = calculate_normal(v1[i], v1[(i + 1) % len(v1)], v1[(i + 2) % len(v1)])
        if normal1 is not None:
            break

    # get second normal
    normal2 = None
    for i in range(len(v2)):
        normal2 = calculate_normal(v2[i], v2[(i + 1) % len(v2)], v2[(i + 2) % len(v2)])
        if normal2 is not None:
            break

    if normal1 is None or normal2 is None:
        return False

    # compare normals directions
    same_normal = np.allclose(normal1, normal2)

    return same_normal


def is_convex_polygon(vertices, polygon, normal):
    num_vertices = len(polygon)
    if num_vertices < 4:
        return True, polygon  # triangular 

    sign = 0
    new_polygon = []
    for i in range(num_vertices):
        p1 = vertices[polygon[i]]
        p2 = vertices[polygon[(i + 1) % num_vertices]]
        p3 = vertices[polygon[(i + 2) % num_vertices]]

        cross_product = np.cross(p2 - p1, p3 - p2)
        current_sign = np.sign(np.dot(cross_product, normal))

        if np.linalg.norm(cross_product) > 1e-6:  # if colinear
            new_polygon.append(polygon[(i + 1) % num_vertices])
        if current_sign == 0:
            continue
        if sign == 0:
            sign = current_sign
        elif sign != current_sign:
            return False, polygon

    return True, new_polygon


def merge_faces(vertices, faces, vertex_normals, normals):
    new_faces = []
    new_vertex_normals = []
    visited = set()
    shared_edges = build_shared_edge_map(faces, vertex_normals)

    for i in range(len(faces)):
        if i in visited:
            continue

        current_face = list(faces[i])
        current_normals = list(vertex_normals[i])
        merged_face = list(current_face)
        merged_normals = list(current_normals)
        face_queue = [i]

        while face_queue:
            current = face_queue.pop()
            current_face = list(faces[current])
            current_normals = list(vertex_normals[current])
            for j in range(len(current_face)):
                edge = (current_face[j], current_face[(j + 1) % len(current_face)])
                vn_edge = (current_normals[j], current_normals[(j + 1) % len(current_face)])
                opposite_edge = (current_face[(j + 1) % len(current_face)], current_face[j])
                opposite_vn_edge = (current_normals[(j + 1) % len(current_face)], current_normals[j])

                if (opposite_edge, opposite_vn_edge) in shared_edges:
                    neighbors = shared_edges[(opposite_edge, opposite_vn_edge)]
                else:
                    continue

                for neighbor in neighbors:
                    if neighbor in visited:
                        continue
                    if are_faces_coplanar(vertices, current_face, faces[neighbor]):
                        neighbor_face = list(faces[neighbor])
                        neighbor_normals = list(vertex_normals[neighbor])

                        try:
                            idx = neighbor_face.index(opposite_edge[0])
                        except ValueError:
                            continue

                        neighbor_face = neighbor_face[idx:] + neighbor_face[:idx]
                        neighbor_normals = neighbor_normals[idx:] + neighbor_normals[:idx]

                        temp_merged_face = merged_face[:]
                        temp_merged_normals = merged_normals[:]

                        if edge[0] in temp_merged_face and edge[1] in temp_merged_face:
                            insert_idx = temp_merged_face.index(edge[0]) + 1

                            for v, vn in zip(neighbor_face, neighbor_normals):
                                if v not in temp_merged_face:
                                    temp_merged_face.insert(insert_idx, v)
                                    temp_merged_normals.insert(insert_idx, vn)
                                    insert_idx += 1

                            normal = normals[current_normals[0]]
                            is_convex, cleaned_face = is_convex_polygon(vertices, temp_merged_face, normal)
                            if is_convex:
                                merged_face = cleaned_face
                                merged_normals = temp_merged_normals
                                face_queue.append(neighbor)
                                visited.add(neighbor)

        new_faces.append(merged_face)
        new_vertex_normals.append(merged_normals)
        visited.add(i)

    return new_faces, new_vertex_normals


def remove_unused_vertices(vertices, faces):
    used_vertices = set()
    for mat_faces in faces.values():
        for face in mat_faces:
            used_vertices.update(face)

    used_vertices = sorted(used_vertices)
    vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertices)}

    new_vertices = vertices[used_vertices]
    new_faces = {material: [[vertex_map[v] for v in face] for face in mat_faces] for material, mat_faces in
                 faces.items()}

    return new_vertices, new_faces


def merge_materials(materials):
    merged_materials = {}
    material_mapping = {}
    for mat_name, props in materials.items():
        props_tuple = tuple((k, tuple(v)) for k, v in sorted(props.items()))
        if props_tuple not in merged_materials:
            new_name = f'material_{len(merged_materials)}'
            merged_materials[props_tuple] = new_name
        material_mapping[mat_name] = merged_materials[props_tuple]
    merged_materials = {v: dict(props) for props, v in merged_materials.items()}
    return merged_materials, material_mapping


def convert_obj_to_ply(vertices, faces, normals, vertex_normals, face_materials, materials, ply_filename):
    if materials is None:
        print(f"Materials are missing. Skipping.")
        return

    filtered_faces, filtered_faces_indices = filter_outside_faces(vertices, faces)
    if filtered_faces is None:
        ply_filename_list=ply_filename.split("/")
        print(f"skip large face number file {ply_filename_list[-2]+':'+ply_filename_list[-1]}")
        return

    inside_faces = [faces[i] for i in range(len(faces)) if i not in filtered_faces_indices]
    duplicate_faces, duplicate_indices = find_duplicate_faces(inside_faces)

    all_faces = filtered_faces + duplicate_faces
    all_indices = filtered_faces_indices + duplicate_indices
    duplicate_labels = [0] * len(filtered_faces) + [1] * len(duplicate_faces)

    if len(all_faces) == 0:
        print(f"No outside or duplicate faces found. Skipping.")
        return

    merged_materials, material_mapping = merge_materials(materials)

    # update face material index
    updated_face_materials = [material_mapping.get(face_materials[idx], 'material_unknown') for idx in all_indices]

    with open(ply_filename, 'w') as file:
        file.write('ply\n')
        file.write('format ascii 1.0\n')
        file.write('comment Generated by OBJ to PLY converter\n')
        file.write('element vertex {}\n'.format(len(vertices)))
        file.write('property float x\n')
        file.write('property float y\n')
        file.write('property float z\n')
        file.write('element face {}\n'.format(len(all_faces)))
        file.write('property list uchar int vertex_indices\n')
        file.write('property uchar int material_index\n')
        file.write('property uchar int duplicate_label\n')

        unique_materials = set(updated_face_materials)
        file.write('element material {}\n'.format(len(unique_materials)))
        for key in ['Ka', 'Kd', 'Ks', 'Ke']:
            for suffix in ['red', 'green', 'blue']:
                file.write(f'property float {key}_{suffix}\n')
        file.write('property float Ns\n')
        file.write('property float d\n')
        file.write('property float illum\n')
        file.write('end_header\n')

        for vertex in vertices:
            file.write('{} {} {}\n'.format(*vertex))

        material_indices = {mat: idx for idx, mat in enumerate(unique_materials)}
        for face, mat, duplicate_label in zip(all_faces, updated_face_materials, duplicate_labels):
            mat_index = material_indices[mat]
            file.write(f'{len(face)} ' + ' '.join(map(str, face)) + f' {mat_index} {duplicate_label}\n')

        for mat in unique_materials:
            props = merged_materials[mat]
            for key in ['Ka', 'Kd', 'Ks', 'Ke']:
                values = props.get(key, [0.0, 0.0, 0.0])
                values = [float(v) if isinstance(v, (list, tuple)) else float(v) for v in values]
                file.write(' '.join(map(str, values)) + ' ')
            ns = props.get('Ns', 0.0)
            d = props.get('d', 1.0)
            illum = props.get('illum', 0)
            ns = float(ns[0]) if isinstance(ns, (list, tuple)) else float(ns)
            d = float(d[0]) if isinstance(d, (list, tuple)) else float(d)
            illum = float(illum[0]) if isinstance(illum, (list, tuple)) else float(illum)
            file.write(f"{ns} {d} {illum}\n")

    print(f"Converted to {ply_filename}")


def process_single_file(input_file, mtl_file, output_file):
    vertices, faces, normals, vertex_normals, face_materials, mtllib = read_obj(input_file, initial_max_faces)
    if vertices is None:
        input_file_list=input_file.split("/")
        print(f"initial face count exceed. Skipping {input_file_list[-4]+':'+input_file_list[-3]}")
        return

    materials = read_mtl(mtl_file)

    merged_faces = {}
    merged_vertex_normals = {}

    for material, mat_faces in faces.items():
        merged_faces[material], merged_vertex_normals[material] = merge_faces(vertices, mat_faces,
                                                                              vertex_normals[material], normals)

    vertices, merged_faces = remove_unused_vertices(vertices, merged_faces)

    new_faces = []
    new_face_materials = []
    for material, mat_faces in merged_faces.items():
        new_faces.extend(mat_faces)
        new_face_materials.extend([material] * len(mat_faces))

    new_vertex_normals = []
    for material, vn in merged_vertex_normals.items():
        new_vertex_normals.extend(vn)

    convert_obj_to_ply(vertices, new_faces, normals, new_vertex_normals, new_face_materials, materials, output_file)


def process_single_file_safe(input_file, mtl_file, output_file):
    try:
        process_single_file(input_file, mtl_file, output_file)
    except Exception as e:
        print(f"Failed to process {input_file}: {e}")

def process_directory(source_directory, output_directory, target_classes=None, max_workers=16):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        # Prepare the list of directories to process
        directories_to_process = [
            os.path.join(source_directory, class_folder)
            for class_folder in os.listdir(source_directory)
            if os.path.isdir(os.path.join(source_directory, class_folder)) and
            (not target_classes or class_folder in target_classes)
        ]

        for class_path in directories_to_process:
            class_folder = os.path.basename(class_path)
            for instance_folder in os.listdir(class_path):
                instance_path = os.path.join(class_path, instance_folder)
                if os.path.isdir(instance_path):
                    obj_filename = os.path.join(instance_path, 'models', 'model_normalized.obj')
                    mtl_filename = os.path.join(instance_path, 'models', 'model_normalized.mtl')
                    if os.path.exists(obj_filename) and os.path.exists(mtl_filename):
                        class_output_dir = os.path.join(output_directory, class_folder)
                        os.makedirs(class_output_dir, exist_ok=True)
                        ply_filename = os.path.join(class_output_dir, f'{instance_folder}.ply')

                        # Check if the target file already exists
                        if os.path.exists(ply_filename):
                            ply_filename_list=ply_filename.split("/") 
                            print(f"Exist target Skipping {ply_filename_list[-2]+':'+ply_filename_list[-1]}")
                            continue

                        futures.append(
                            executor.submit(process_single_file_safe, obj_filename, mtl_filename, ply_filename))

        # Monitoring the future completions
        for future in concurrent.futures.as_completed(futures):
            future.result()  # This will raise exceptions if any occur during file processing

source_directory = r"data/shapenet_download"
output_directory = r"data/shapenet_sim"
target_classes =["chair", "display",  
    "loudspeaker", "sofa", "table" ,"bathtub",  "bench", "bookshelf", "cabinet", "cellular telephone","file","knife","lamp","pot","vessel"]

os.makedirs(output_directory, exist_ok=True)
process_directory(source_directory, output_directory, target_classes=target_classes, max_workers=128)