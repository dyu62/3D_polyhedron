from math import pi as PI
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from torch.nn import ModuleList, Parameter 
from torch_geometric.nn import  Linear
from torch_geometric.nn.dense.linear import Linear
from torch_scatter import scatter
from util import get_angle
def calculate_face_angle(v1, v2):
    return PI-torch.atan2( torch.cross(v1, v2, dim=1).norm(p=2, dim=1), (v1 * v2).sum(dim=1))

class Smodel(nn.Module):
    def __init__(self,  h_channel=16,face_attri_size=23, localdepth=1,num_interactions=3,finaldepth=3,batchnorm=True,edge_rep=True,geo_encoding_dim=10):
        super(Smodel,self).__init__()
        self.training=True
        self.h_channel = h_channel
        self.face_attri_size=face_attri_size
        self.localdepth = localdepth
        self.num_interactions=num_interactions
        self.finaldepth=finaldepth
        self.batchnorm = batchnorm        
        self.activation=nn.ReLU()
        self.att = Parameter(torch.ones(2),requires_grad=True)
        self.edge_rep=edge_rep
        """ 
        edge feature mlp if use edge rep
        """
        self.mlp_geo = ModuleList()
        for i in range(self.localdepth):
            if i == 0:
                self.mlp_geo.append(Linear(geo_encoding_dim+2*self.face_attri_size, h_channel))
            else:
                self.mlp_geo.append(Linear(h_channel, h_channel))
            if self.batchnorm == True:
                self.mlp_geo.append(nn.BatchNorm1d(h_channel))
            self.mlp_geo.append(self.activation)      
        """ 
        edge feature mlp if use coords
        """            
        self.mlp_geo_backup = ModuleList()
        for i in range(self.localdepth):
            if i == 0:
                self.mlp_geo_backup.append(Linear(6, h_channel))
            else:
                self.mlp_geo_backup.append(Linear(h_channel, h_channel))
            if self.batchnorm == True:
                self.mlp_geo_backup.append(nn.BatchNorm1d(h_channel))
            self.mlp_geo_backup.append(self.activation)        
            

        self.interactions= ModuleList()
        for i in range(self.num_interactions):
            block = SPNN(self.face_attri_size,self.h_channel,self.activation,self.finaldepth,self.batchnorm,\
                num_input_geofeature=self.h_channel,edge_rep=self.edge_rep)
            self.interactions.append(block)

    def forward(self, inputs):
        input_feature,coords,face_norm,face_x,edge_whichface,edge_index,edge_index_2rd, edx_jk, edx_ij,batch,edge_rep=inputs
        i, j, k = edge_index_2rd 
        norm_ij=face_norm[edge_whichface[edx_ij]]
        norm_jk=face_norm[edge_whichface[edx_jk]]
        angle_face=calculate_face_angle(norm_ij,norm_jk)
        angle_face=angle_face*0
        
        distance_ij=(coords[j] - coords[i]).norm(p=2, dim=1)
        distance_jk=(coords[j] - coords[k]).norm(p=2, dim=1)
        theta_ijk = get_angle(coords[j] - coords[i], coords[k] - coords[j])
        same_face= (edge_whichface[edx_ij]==edge_whichface[edx_jk])
        if edge_rep:
            """ 
            use proposed edge representation
            """ 
            geo_encoding_coords=torch.cat([coords[j],coords[i]],dim=-1)
            geo_encoding_4=torch.cat([distance_ij[:,None],distance_jk[:,None],theta_ijk[:,None],angle_face[:,None]],dim=-1)
            geo_encoding=torch.cat([geo_encoding_4,geo_encoding_coords],dim=-1)
            if self.mlp_geo[0].weight.shape[1]==4+2*self.face_attri_size:
                geo_encoding=geo_encoding_4
            if self.face_attri_size!=0:
                geo_encoding=torch.cat([geo_encoding,face_x[edge_whichface[edx_ij]],face_x[edge_whichface[edx_jk]]],dim=-1)
        else:
            """ simple coords """
            geo_encoding=torch.cat([coords[j],coords[i]],dim=-1)
        for lin in self.mlp_geo:
            geo_encoding=lin(geo_encoding)
        """ 
        GNN calculate the node feature
        """
        node_feature= input_feature
        node_feature_list=[]
        for interaction in self.interactions:
            node_feature =  interaction(node_feature,geo_encoding,edge_index_2rd,edx_jk,edx_ij,edge_whichface,self.att,same_face)
            node_feature_list.append(node_feature)
        return node_feature_list
    
class SPNN(torch.nn.Module):
    def __init__(
        self,
        face_x_size,
        hidden_channels,
        activation=torch.nn.ReLU(),
        finaldepth=3,
        batchnorm=True,
        num_input_geofeature=0,
        edge_rep=True
    ):
        super(SPNN, self).__init__()
        self.face_x_size=face_x_size
        self.activation = activation
        self.finaldepth = finaldepth
        self.batchnorm = batchnorm
        self.num_input_geofeature=num_input_geofeature
        self.edge_rep=edge_rep

        self.WMLP_list = ModuleList()
        for _ in range(2):
            WMLP = ModuleList()
            for i in range(self.finaldepth + 1):
                if i == 0:
                    WMLP.append(Linear(hidden_channels*3+num_input_geofeature, hidden_channels))
                else:
                    WMLP.append(Linear(hidden_channels, hidden_channels))  
                if self.batchnorm == True:
                    WMLP.append(nn.BatchNorm1d(hidden_channels))
                WMLP.append(self.activation)
            self.WMLP_list.append(WMLP)
 
    def forward(self, node_feature,geo_encoding,edge_index_2rd,edx_jk,edx_ij,edge_whichface,att,same_face):
        
        num_node=node_feature.shape[0]
        i,j,k = edge_index_2rd
        node_attr_0st = node_feature[i]
        node_attr_1st = node_feature[j]
        node_attr_2 = node_feature[k]

        concatenated_vector = torch.cat([node_attr_0st,node_attr_1st,node_attr_2,geo_encoding],dim=-1,)
        x_i = concatenated_vector

        masks=[same_face,~same_face]
        x_output=torch.zeros(x_i.shape[0],self.WMLP_list[0][0].weight.shape[0],device=x_i.device)
        
        for index in range(len(masks)):
            WMLP=self.WMLP_list[index]
            x=x_i[masks[index]]
            for lin in WMLP:
                x=lin(x)    
            x = F.leaky_relu(x)*att[index]
            x_output[masks[index]]+=x 
        out_feature = scatter(x_output, i, dim=0, dim_size=num_node,reduce='add') 
        return out_feature
