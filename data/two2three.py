import copy
import os.path
import torch
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_dense_adj

def find_directed_and_undirected_edges(edge_index):
    # Initialize masks for directed and undirected edges
    directed_mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
    undirected_mask = torch.zeros(edge_index.size(1), dtype=torch.bool)

    # Helper function to check if an edge is in edge_index
    def edge_exists(edge, edge_index):
        return ((edge_index[0] == edge[0]) & (edge_index[1] == edge[1])).any()

    for i in range(edge_index.size(1)):
        if undirected_mask[i]:  # Skip if already marked as undirected
            continue
        current_edge = edge_index[:, i]
        reverse_edge = current_edge.flip(0)
        if edge_exists(reverse_edge, edge_index):
            # Mark both current and reverse edges as undirected
            undirected_mask[i] = True
            reverse_index = ((edge_index[0] == reverse_edge[0]) & (
                edge_index[1] == reverse_edge[1])).nonzero(as_tuple=True)[0]
            undirected_mask[reverse_index] = True
        else:
            directed_mask[i] = True

    # # Extract directed and undirected edges based on the masks
    # directed_edges = edge_index[:, directed_mask]
    # undirected_edges = edge_index[:, undirected_mask]
    # return directed_edges, undirected_edges
    return directed_mask, undirected_mask


def calculate_angle(v1, v2):
    # [0-pi]
    cos_theta = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2))
    # possible that floating-point precision issues make 1.000001
    angle_rad = torch.acos(torch.clamp(cos_theta, -1, 1))

    # Determine the direction of rotation (counterclockwise or not)
    # For 2D vectors: If the sign of the cross product z-component is positive, v2 is counterclockwise from v1
    # In 2D, cross product "z-component" equivalent: v1.x*v2.y - v1.y*v2.x
    rotation_direction = v1[0]*v2[1] - v1[1]*v2[0]
    is_counterclockwise = rotation_direction > 0
    return angle_rad, is_counterclockwise


def find_small_faces(edge_index, pos):
    # Initialize containers for face detection
    faces = []
    edge_visited = set()
    adjacency_list = to_dense_adj(edge_index)[0]

    def find_next_node(current_node, current_edge_vec):
        # current_node is the pivot
        min_angle = 4
        max_angle = -1
        next_node, next_node_convex, next_node_concave = None, None, None
        candidate_nodes = torch.nonzero(adjacency_list[current_node]).squeeze()
        if candidate_nodes.dim() == 0:
            candidate_vec = pos[candidate_nodes] - pos[current_node]
            if all(candidate_vec == current_edge_vec) or (current_node, candidate_nodes.item()) in edge_visited:  # no jump back
                return None
            else:
                return candidate_nodes.item()
        else:
            for candidate_node in candidate_nodes:
                candidate_node = candidate_node.item()
                candidate_vec = pos[candidate_node] - pos[current_node]
                if (current_node, candidate_node) in edge_visited or all(candidate_vec == current_edge_vec):
                    continue
                else:
                    angle, is_counterclockwise = calculate_angle(
                        candidate_vec, current_edge_vec)
                    if is_counterclockwise:  # convex
                        if angle < min_angle:
                            min_angle = angle
                            next_node_convex = candidate_node
                    else:  # convace
                        if angle > max_angle:
                            max_angle = angle
                            next_node_concave = candidate_node
            next_node = next_node_convex if next_node_convex is not None else next_node_concave
        return next_node
    # Main loop to find faces
    for edge_idx in range(edge_index.size(1)):
        current_edge = (edge_index[0, edge_idx].item(), edge_index[1, edge_idx].item())
        if current_edge in edge_visited:
            continue  # Skip if this directed edge has been visited
        # Start a new face
        current_node, next_node = current_edge
        face = [current_node]
        while True:
            face.append(next_node)
            edge_visited.add((current_node, next_node))
            if next_node == face[0]:  # already a loop
                break
            # Find the next edge in the sequence to continue the face
            current_edge_vec = pos[current_node] - \
                pos[next_node]  # angle vertex is next_node
            temp_node = next_node
            next_node = find_next_node(next_node, current_edge_vec)
            if next_node is None or next_node == current_node:
                # Closed loop or no next node found; complete the face
                break
            current_node = temp_node
        if len(face) > 3:
            faces.append(face)
    return faces


def find_small_faces_clock(edge_index, pos):
    # Initialize containers for face detection
    faces = []
    edge_visited = set()
    adjacency_list = to_dense_adj(edge_index)[0]

    def find_next_node(current_node, current_edge_vec):
        min_angle = 4
        max_angle = -1
        next_node, next_node_convex, next_node_concave = None, None, None
        candidate_nodes = torch.nonzero(adjacency_list[current_node]).squeeze()
        if candidate_nodes.dim() == 0:
            candidate_vec = pos[candidate_nodes] - pos[current_node]
            if all(candidate_vec == current_edge_vec) or (current_node, candidate_nodes.item()) in edge_visited: 
                return None
            else:
                return candidate_nodes.item()            
        else:
            for candidate_node in candidate_nodes:
                candidate_node = candidate_node.item()
                candidate_vec = pos[candidate_node] - pos[current_node]
                if (current_node, candidate_node) in edge_visited or all(candidate_vec == current_edge_vec):
                    continue
                else:
                    angle, is_counterclockwise = calculate_angle(
                        candidate_vec, current_edge_vec)
                    if not is_counterclockwise:
                        if angle < min_angle:
                            min_angle = angle
                            next_node_convex = candidate_node
                    else:
                        if angle > max_angle:
                            max_angle = angle
                            next_node_concave = candidate_node
            next_node = next_node_convex if next_node_convex is not None else next_node_concave
        return next_node
    # Main loop to find faces
    for edge_idx in range(edge_index.size(1)):
        current_edge = (edge_index[0, edge_idx].item(),
                        edge_index[1, edge_idx].item())
        if current_edge in edge_visited:
            continue  # Skip if this directed edge has been visited
        # Start a new face
        current_node, next_node = current_edge
        face = [current_node]
        while True:
            face.append(next_node)
            edge_visited.add((current_node, next_node))
            if next_node == face[0]:  # already a loop
                break
            # Find the next edge in the sequence to continue the face
            current_edge_vec = pos[current_node] - \
                pos[next_node]  # angle vertex is next_node
            temp_node = next_node
            next_node = find_next_node(next_node, current_edge_vec)
            if next_node is None or next_node == current_node:
                # Closed loop or no next node found; complete the face
                break
            current_node = temp_node
        face = face[::-1]
        if len(face) > 3:
            faces.append(face)
    return faces


def extrude_shape(data):
    # data['faces']: top, bottom, side
    # data.edge_index : top, bottom, side
    num_original_vertices = data['vertices'].x.size(0)
    def get_dual_node(x): return x+num_original_vertices
    faces = []
    # pos
    original_pos = data.pos
    new_z_values = torch.zeros((num_original_vertices, 1))
    new_vertices_pos = torch.cat([original_pos, torch.ones(
        num_original_vertices, 1)], dim=1)  # Original vertices with z=0
    extruded_vertices_pos = torch.cat(
        [original_pos, new_z_values], dim=1)  # New vertices with z=0
    # Combine pos
    data.pos = torch.cat([new_vertices_pos, extruded_vertices_pos], dim=0)

    # edge_index
    original_edge_index = data[('vertices', 'inside', 'vertices')].edge_index
    bottom_edge_index = (original_edge_index +
                         num_original_vertices).flip(dims=[0])
    directed_edges, undirected_edges = find_directed_and_undirected_edges(
        original_edge_index)
    side_edge_index = []
    for boundary_edge in (original_edge_index[:, directed_edges]).t():
        i, j = boundary_edge
        side_edge_index += [torch.tensor((j, i)), torch.tensor((i, get_dual_node(i))), torch.tensor(
            (get_dual_node(i), get_dual_node(j))), torch.tensor((get_dual_node(j), j))]
        faces.append([j.item(), i.item(), get_dual_node(
            i).item(), get_dual_node(j).item(), j.item()])
    side_edge_index = torch.stack(side_edge_index, dim=1)
    # Combine edge_index
    data.edge_index = torch.cat(
        [original_edge_index, bottom_edge_index, side_edge_index], dim=1)

    # faces
    original_faces = find_small_faces(
        data[('vertices', 'inside', 'vertices')].edge_index, original_pos)
    bottom_faces = [face[::-1] for face in original_faces]
    bottom_faces = [[i + num_original_vertices for i in face]
                    for face in bottom_faces]
    data['faces'] = original_faces+bottom_faces+faces
    return data


def extrude_shape_clock(data):
    # data['faces']: top, bottom, side
    # data.edge_index : top, bottom, side
    num_original_vertices = data['vertices'].x.size(0)
    def get_dual_node(x): return x+num_original_vertices
    faces = []
    # pos
    original_pos = data.pos
    new_z_values = torch.zeros((num_original_vertices, 1))
    new_vertices_pos = torch.cat([original_pos, torch.ones(num_original_vertices, 1)], dim=1)  # Original vertices (2d) + z=0
    extruded_vertices_pos = torch.cat([original_pos, new_z_values], dim=1)  # New vertices with z=1
    # Combine pos
    data.pos = torch.cat([new_vertices_pos, extruded_vertices_pos], dim=0)

    # edge_index
    original_edge_index = data[('vertices', 'inside', 'vertices')].edge_index
    bottom_edge_index = copy.deepcopy(
        original_edge_index + num_original_vertices)
    original_edge_index = original_edge_index.flip(0)
    # directed edges for side face
    directed_edges, undirected_edges = find_directed_and_undirected_edges(
        original_edge_index)
    side_edge_index = []
    # side faces and edges
    for boundary_edge in (original_edge_index[:, directed_edges]).t():
        i, j = boundary_edge
        side_edge_index += [torch.tensor((j, i)), torch.tensor((i, get_dual_node(i))), torch.tensor(
            (get_dual_node(i), get_dual_node(j))), torch.tensor((get_dual_node(j), j))]
        faces.append([j.item(), i.item(), get_dual_node(
            i).item(), get_dual_node(j).item(), j.item()])
    side_edge_index = torch.stack(side_edge_index, dim=1)
    # Combine edge_index
    data.edge_index = torch.cat(
        [original_edge_index, bottom_edge_index, side_edge_index], dim=1)

    # faces
    original_faces = find_small_faces_clock(
        data[('vertices', 'inside', 'vertices')].edge_index, original_pos)
    bottom_faces = [face[::-1] for face in original_faces]
    bottom_faces = [[i + num_original_vertices for i in face]
                    for face in bottom_faces]
    data['faces'] = original_faces+bottom_faces+faces
    return data


def add_attributes(data):
    def calculate_normal(face, pos):
        # Retrieve positions for the first three vertices A, B, and C
        A = pos[face[0]]
        B = pos[face[1]]
        C = pos[face[2]]
        D=pos[face[-2]]

        # Compute vectors AB and BC
        AB = B - A
        BC = C - B
        AD=D-A

        # Compute the normal vector using the cross product of AB and AC
        normal = torch.cross(AB, BC)
        if torch.norm(normal) == 0:
            print("same line")
            normal = torch.cross(AB, AD)
        # Normalize the normal vector
        normal = normal / torch.norm(normal)
        assert (torch.sum(torch.isnan(normal)) ==0)
        return normal
    pos = data.pos
    edge_index = data.edge_index
    faces = data.faces
    edge_face = torch.zeros((edge_index.size(1)), dtype=torch.long)
    # `edge_face` attribute
    for face_index, face in enumerate(faces):
        for i in range(len(face)):
            # Identify edge in edge_index
            start_node = face[i]
            # Ensures circular connection in the face
            end_node = face[(i + 1) % len(face)]
            edge_position = (edge_index[0] == start_node) & (
                edge_index[1] == end_node)
            edge_face[edge_position] = face_index
    # `face_norm` attribute
    face_norm = [calculate_normal(face, pos) for face in faces]
    face_norm = torch.stack(face_norm)
    # New data structure without `vertices` and `(vertices, inside, vertices)`
    new_data = HeteroData()
    new_data.pos = pos
    new_data.edge_index = edge_index
    new_data.face_list = faces
    new_data.edge_face = edge_face
    new_data.face_norm = face_norm
    return new_data


def write_ply(hetero_data, filename="output.ply"):
    with open(filename, "w") as ply_file:
        # Write PLY header
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write(f"element vertex {len(hetero_data.pos)}\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write(f"element face {len(hetero_data.face_list)}\n")
        ply_file.write("property list uchar int vertex_index\n")
        ply_file.write("end_header\n")

        # Write vertex positions
        for pos in hetero_data.pos:
            ply_file.write(f"{pos[0]} {pos[1]} {pos[2]}\n")

        for face in hetero_data.face_list:
            face_vertex_count = len(face)-1
            face_indices_str = ' '.join(str(index) for index in face[:-1])
            ply_file.write(f"{face_vertex_count} {face_indices_str}\n")


if __name__ == '__main__':
    #  Example: find_directed_and_undirected_edges
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 3, 4, 5], [1, 0, 3, 2, 4, 3, 6]], dtype=torch.long)
    directed_edges, undirected_edges = find_directed_and_undirected_edges(
        edge_index)

    print("Directed Edges:", directed_edges)
    print("Undirected Edges:", undirected_edges)

    # Example: find_small_faces
    # v1=torch.tensor([1,-1],dtype=float)
    # v2=torch.tensor([1,-1],dtype=float)
    # calculate_angle(v1,v2)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 0, 2],  # Source nodes
        [1, 2, 3, 0, 2, 0]   # Destination nodes
    ], dtype=torch.long)
    pos = torch.tensor([
        [0, 0, 1, 1],  # Source nodes
        [1, 0, 0, 1]   # Destination nodes
    ], dtype=torch.float).t()
    faces = find_small_faces(edge_index, pos)
    print("Faces identified:", faces)

    # Example usage: extrude_shape, add_attributes
    data0 = HeteroData()
    data0['vertices'].x = torch.randn(4, 1)  # Random example vertices
    data0.pos = torch.tensor([
        [0, 0, 1, 1],  # Source nodes
        [1, 0, 0, 1]   # Destination nodes
    ], dtype=torch.float).t()
    # Assume edges are defined here
    data0[('vertices', 'inside', 'vertices')].edge_index = torch.tensor([
        [0, 1, 2, 3, 0, 2],  # Source nodes
        [1, 2, 3, 0, 2, 0]   # Destination nodes
    ], dtype=torch.long)
    data0[('vertices', 'inside', 'vertices')].edge_attr = torch.randn(13, 1)
    data1 = extrude_shape(data0)
    new_data = add_attributes(data1)

    def edge_exists_in_face(edge, face):
        # Check if the edge exists in the cyclic list representing a face
        for i in range(len(face) - 1):  # -1 because the last vertex is the same as the first
            if (face[i] == edge[0].item() and face[i + 1] == edge[1].item()):
                return True
        return False
    all([edge_exists_in_face(data1.edge_index[:, i], data1.faces[new_data.edge_face[i]])
        for i in range(len(new_data.edge_index.t()))])
