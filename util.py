import torch
import numpy as np
import scipy
import torch.nn.functional as F
from sklearn.metrics import precision_score,recall_score,f1_score,roc_auc_score
from torch_sparse import SparseTensor
from torch_geometric.utils import to_networkx

def is_connected(data):
    # Convert PyG data object to a NetworkX graph for easy BFS
    G = to_networkx(data, to_undirected=True)
    
    # Start BFS from the first node (index 0)
    start_node = 0
    visited = set()
    queue = [start_node]

    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.add(node)
            # Add unvisited neighbors
            queue.extend(n for n in G.neighbors(node) if n not in visited)
    
    # Check if all nodes were visited
    return len(visited) == data.num_nodes


def record(values,epoch,writer,phase="Train"):
    """ tfboard write """
    for key,value in values.items():
        writer.add_scalar(key+"/"+phase,value,epoch)           
def calculate(y_hat,y_true,y_hat_logit):
    """ calculate five metrics using y_hat, y_true, y_hat_logit """
    train_acc=(np.array(y_hat) == np.array(y_true)).sum()/len(y_true) 
    return train_acc
def calculate_full(y_hat,y_true,y_hat_logit):
    """ calculate five metrics using y_hat, y_true, y_hat_logit """
    train_acc=(np.array(y_hat) == np.array(y_true)).sum()/len(y_true) 
    recall=recall_score(y_true, y_hat,zero_division=0,average='micro')
    precision=precision_score(y_true, y_hat,zero_division=0,average='micro')
    Fscore=f1_score(y_true, y_hat,zero_division=0,average='micro')
    roc=roc_auc_score(y_true, scipy.special.softmax(np.array(y_hat_logit),axis=1)[:,1],average='micro',multi_class='ovr')
    one_hot_encoded_labels = np.zeros((len(y_true), 100))
    one_hot_encoded_labels[np.arange(len(y_true)), y_true] = 1
    roc=roc_auc_score(one_hot_encoded_labels, scipy.special.softmax(np.array(y_hat_logit),axis=1),average='micro',multi_class='ovr')
    return train_acc


def print_1(epoch,phase,values,color=None):
    """ print epoch info"""
    if color is not None:
        print(color( f"epoch[{epoch:d}] {phase}"+ " ".join([f"{key}={value:.3f}" for key, value in values.items()]) ))
    else:
        print(( f"epoch[{epoch:d}] {phase}"+ " ".join([f"{key}={value:.3f}" for key, value in values.items()]) ))

def get_angle(v1, v2):
    if v1.shape[1]==2:
        v1=F.pad(v1, (0, 1),value=0)
    if v2.shape[1]==2:
        v2= F.pad(v2, (0, 1),value=0)
    return torch.atan2( torch.cross(v1, v2, dim=1).norm(p=2, dim=1), (v1 * v2).sum(dim=1))
def get_theta(v1, v2):
    angle=get_angle(v1, v2)
    if v1.shape[1]==2:
        v1=F.pad(v1, (0, 1),value=0)
    if v2.shape[1]==2:
        v2= F.pad(v2, (0, 1),value=0)
    v = torch.cross(v1, v2, dim=1)[...,2]
    flag = torch.sign((v))
    flag[flag==0]=-1 
    return angle*flag   

def triplets(edge_index, num_nodes):
    row, col = edge_index  
    value = torch.arange(row.size(0), device=row.device)
    adj_t = SparseTensor(row=row, col=col, value=value,
                         sparse_sizes=(num_nodes, num_nodes))
    adj_t_col = adj_t[:,row]
    num_triplets = adj_t_col.set_value(None).sum(dim=0).to(torch.long)
    # Node indices (k->j->i) for triplets.
    idx_j = row.repeat_interleave(num_triplets)
    idx_i = col.repeat_interleave(num_triplets) 
    edx_2nd = value.repeat_interleave(num_triplets)
    idx_k = adj_t_col.t().storage.col() #save node k (in the triplets)
    edx_1st = adj_t_col.t().storage.value() 
    mask1 = (idx_i == idx_k) & (idx_j != idx_i)  # Remove go back triplets. 
    mask2 = (idx_i == idx_j) & (idx_j != idx_k)  # Remove repeat self loop triplets
    mask3 = (idx_j == idx_k) & (idx_i != idx_k)  # Remove self-loop neighbors 
    # mask4= (idx_j == idx_k) & (idx_i == idx_k) # remove (idx_j == idx_k) & (idx_i == idx_k) 0
    mask = ~(mask1 | mask2 | mask3) # 0 -> 0 -> 0 or # 0 -> 1 -> 2 
    # mask=~(mask1 | mask2 | mask3|mask4) #k -> j -> i
    idx_i, idx_j, idx_k, edx_1st, edx_2nd = idx_i[mask], idx_j[mask], idx_k[mask], edx_1st[mask], edx_2nd[mask]
    
    # count real number of triplets for each i
    num_triplets_real = torch.cumsum(num_triplets, dim=0) - torch.cumsum(~mask, dim=0)[torch.cumsum(num_triplets, dim=0)-1]

    return torch.stack([idx_i, idx_j, idx_k]), num_triplets_real.to(torch.long), edx_1st, edx_2nd

def is_valid_hetero_graph(graph) -> bool:
    """
    Check if a given HeteroData graph is valid.
    
    Args:
        graph (HeteroData): The graph to check.
    
    Returns:
        bool: True if the graph is valid, False otherwise.
    """
    num_vertices = graph['vertices'].x.size(0)
    num_faces = graph['face'].x.size(0)
    num_edges = graph['edge'].x.size(0)
    
    # Check if node positions are non-negative
    if (graph.pos.shape[0]!= num_vertices):
        print("Invalid graph: pos shape!= num_vertices")
        return False
    if (graph['edge_face'].max()!= num_faces-1 or graph['edge_face'].min()!= 0):
        print("Invalid edge_face number")
        return False   
    if (graph['edge_face'].shape[0]!= num_edges):
        print("graph['edge_face'].shape[0]!= num_edges")
        return False  
    # Check if edge indices are within the valid range
    if (graph['vertices', 'to', 'vertices'].edge_index < 0).any() or \
       (graph['vertices', 'to', 'vertices'].edge_index >= num_vertices).any():
        print("Invalid graph: edge indices out of range for vertices")
        return False
    
    if (graph['edge', 'on', 'face'].edge_index < 0).any() or \
       (graph['edge', 'on', 'face'].edge_index[1] >= num_faces).any():
        assert(False)
        print("Invalid graph: edge indices out of range for edges")
        return False
    
    if (graph['edge', 'on', 'face'].edge_index[0] >= num_edges).any() :   
        print("Invalid graph: edge indices out of range for edges")
        return False
    
    # Check if face normals have valid shape
    if graph.f_norm.size(1) != 3:
        print("Invalid graph: face normals do not have 3 components")
        return False
    
    # Check if vertices, faces, and edges features have valid shapes
    if graph['vertices'].x.size(1) != 3:
        print("Invalid graph: vertices features do not have 3 components")
        return False
    
    return True

def check_hetero_dataset(dataset):
    """
    Check each graph in the HeteroData dataset for validity.
    
    Args:
        dataset (Dataset): The dataset to check.
    
    Returns:
        List[bool]: A list indicating the validity of each graph.
    """
    validity = []
    for i, graph in enumerate(dataset):
        if not is_valid_hetero_graph(graph):
            print(f"Graph {i} is invalid")
        validity.append(is_valid_hetero_graph(graph))
    
    return validity