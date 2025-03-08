# clean is the final step
import os
def clean_ply_faces(input_ply_filename, output_ply_filename):
    with open(input_ply_filename, 'r') as file:
        lines = file.readlines()

    header = []
    vertices = []
    faces = []
    materials = []
    in_face_section = False
    vertex_count = 0
    face_count = 0
    material_count = 0

    # Read header and count elements
    for line in lines:
        if line.startswith('element vertex'):
            vertex_count = int(line.split()[-1])
        elif line.startswith('element face'):
            face_count = int(line.split()[-1])
        elif line.startswith('element material'):
            material_count = int(line.split()[-1])
        header.append(line)
        if line.startswith('end_header'):
            break

    vertex_start = len(header)
    vertex_end = vertex_start + vertex_count

    face_start = vertex_end
    face_end = face_start + face_count

    material_start = face_end

    vertices = lines[vertex_start:vertex_end]
    materials = lines[material_start:]

    valid_faces = []
    for face in lines[face_start:face_end]:
        face_data = face.strip().split()
        vertex_count = int(face_data[0])
        if vertex_count >= 3:
            valid_faces.append(face)

    updated_face_count = len(valid_faces)

    with open(output_ply_filename, 'w') as file:
        for line in header:
            if line.startswith('element face'):
                file.write(f'element face {updated_face_count}\n')
            else:
                file.write(line)
        for vertex in vertices:
            file.write(vertex)
        for face in valid_faces:
            file.write(face)
        for material in materials:
            file.write(material)

def process_ply_directory(source_directory, output_directory):
    for class_folder in os.listdir(source_directory):
        class_path = os.path.join(source_directory, class_folder)
        if os.path.isdir(class_path):
            output_class_path = os.path.join(output_directory, class_folder)
            os.makedirs(output_class_path, exist_ok=True)
            for ply_file in os.listdir(class_path):
                if ply_file.endswith('.ply'):
                    input_ply_filename = os.path.join(class_path, ply_file)
                    output_ply_filename = os.path.join(output_class_path, ply_file)
                    clean_ply_faces(input_ply_filename, output_ply_filename)
                    print(f"Processed {input_ply_filename} -> {output_ply_filename}")

source_directory = r"data/shapenet_sim"
output_directory = r"data/shapenet_sim_clean"

os.makedirs(output_directory, exist_ok=True)
process_ply_directory(source_directory, output_directory)
