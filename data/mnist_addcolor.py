# add color attribute to blank mnist ply files

import numpy as np
import os

def read_ply(filename):
    with open(filename, 'r') as file:
        # Skip header
        while True:
            line = file.readline().strip()
            if line == "end_header":
                break

        vertices = []
        faces = []
        while True:
            line = file.readline().strip()
            if line == "":
                break
            parts = line.split()
            if len(parts) == 3:
                vertices.append(tuple(map(float, parts)))
            elif len(parts) > 1:
                num_vertices = int(parts[0])
                face_indices = list(map(int, parts[1:num_vertices + 1]))
                faces.append(face_indices)

    return vertices, faces


def write_ply(filename, vertices, faces, face_colors):
    with open(filename, 'w') as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write("element vertex {}\n".format(len(vertices)))
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("element face {}\n".format(len(faces)))
        file.write("property list uchar int vertex_index\n")
        file.write("property uchar red\n")
        file.write("property uchar green\n")
        file.write("property uchar blue\n")
        file.write("end_header\n")

        for vertex in vertices:
            file.write("{} {} {}\n".format(*vertex))

        for face, color in zip(faces, face_colors):
            face_format = ' '.join(['{}'] * (len(face) + 4)) + '\n'
            file.write(face_format.format(len(face), *face, *color))


def classify_faces(vertices, faces):
    vertices_array = np.array(vertices)
    face_colors = []

    # Threshold for considering a face as primarily at z=1 or z=0
    z_threshold = 0.75

    # Identify vertices for the bottom face
    min_y = np.min(vertices_array[:, 1])
    bottom_indices = [i for i, v in enumerate(vertices) if v[1] == min_y and v[2] in [0, 1]]

    for face in faces:
        face_vertices = vertices_array[face]

        # Proportion of vertices at z=1 or z=0
        front_proportion = np.mean(face_vertices[:, 2] == 1)
        back_proportion = np.mean(face_vertices[:, 2] == 0)

        if front_proportion >= z_threshold:
            face_colors.append((255, 0, 0))  # Red for front face
        elif back_proportion >= z_threshold:
            face_colors.append((0, 0, 255))  # Blue for back face
        elif all(v in face for v in bottom_indices):
            face_colors.append((128, 0, 128))  # Purple for bottom face
        else:
            face_colors.append((0, 255, 0))  # Green for other faces

    return face_colors


def process_folder(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith(".ply"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            vertices, faces = read_ply(input_path)
            face_colors = classify_faces(vertices, faces)
            write_ply(output_path, vertices, faces, face_colors)
            print(f"Processed {filename}")


# Specify the input and output folder paths
input_folder = r"data/mnist"
output_folder = r"data/mnist_color"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

process_folder(input_folder, output_folder)

print("All files processed successfully.")
