import argparse
import copy
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import cm
torch.autograd.set_detect_anomaly(True)
import time
from datetime import datetime
import torch.nn as nn
import torch_geometric
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import MLP
from torch_geometric.nn.pool import global_add_pool
import datasets
import models
import util
from CONST import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug',default="F", type=str)
    parser.add_argument('--train_batch', default=32, type=int)
    parser.add_argument('--test_batch', default=32, type=int)
    parser.add_argument('--edge_rep', type=str, default="T")
    parser.add_argument('--batchnorm', type=str, default="T")
    parser.add_argument('--extent_norm', type=str, default="T")
    parser.add_argument('--maskface',default="F",type=str) 
    parser.add_argument('--geo_encoding',default="4",type=str) 
    parser.add_argument('--rotate',default="T",type=str)
    
    parser.add_argument('--loss_coef', default=0.1, type=float)
    parser.add_argument('--h_ch', default=512, type=int)
    parser.add_argument('--localdepth', type=int, default=1)
    parser.add_argument('--num_interactions', type=int, default=4) 
    parser.add_argument('--finaldepth', type=int, default=4)
    parser.add_argument('--classifier_depth', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.0)

    parser.add_argument('--dataset', type=str, default='building')
    parser.add_argument('--log', type=str, default="T") 
    parser.add_argument('--test_per_round', type=int, default=10)
    parser.add_argument('--patience', type=int, default=30) 
    parser.add_argument('--nepoch', type=int, default=501)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    args.log=True if args.log=="T" else False 
    args.edge_rep=True if args.edge_rep=="T" else False
    args.batchnorm=True if args.batchnorm=="T" else False
    args.save_dir=os.path.join('save',args.dataset)
    return args


def contrastive_loss(embeddings,labels,margin):
    # embeddings=output=torch.randn(4,3)
    # labels=torch.tensor([1,0,1,0])
    
    positive_mask = labels.view(-1, 1) == labels.view(1, -1)
    negative_mask = ~positive_mask

    # Calculate the number of positive and negative pairs
    num_positive_pairs = positive_mask.sum() - labels.shape[0]
    num_negative_pairs = negative_mask.sum()

    # If there are no negative pairs, return a placeholder loss
    if num_negative_pairs==0 or num_positive_pairs== 0:
        # print("all pos or neg")
        return torch.tensor(0, dtype=torch.float)
    # Calculate the pairwise Euclidean distances between embeddings
    distances = torch.cdist(embeddings, embeddings)/np.sqrt(embeddings.shape[1])
    
    if num_positive_pairs>num_negative_pairs:
        # Sample an equal number of + pairs 
        positive_indices = torch.nonzero(positive_mask)
        random_positive_indices = torch.randperm(len(positive_indices))[:num_negative_pairs]
        selected_positive_indices = positive_indices[random_positive_indices]

        # Select corresponding negative pairs
        negative_mask.fill_diagonal_(False)
        negative_distances = distances[negative_mask].view(-1, 1)
        positive_distances = distances[selected_positive_indices[:,0],selected_positive_indices[:,1]].view(-1, 1)
    else: # case for most datasets
        # Sample an equal number of - pairs 
        negative_indices = torch.nonzero(negative_mask)
        random_negative_indices = torch.randperm(len(negative_indices))[:num_positive_pairs]
        selected_negative_indices = negative_indices[random_negative_indices]

        # Select corresponding positive pairs
        positive_mask.fill_diagonal_(False)
        positive_distances = distances[positive_mask].view(-1, 1)
        negative_distances = distances[selected_negative_indices[:,0],selected_negative_indices[:,1]].view(-1, 1)

    # Calculate the loss for positive and negative pairs
    loss = (positive_distances - negative_distances + margin).clamp(min=0).mean()
    return loss
 
    
def forward(args,data,model,mlpmodel,optimizer,device,criterion):
    data = data.to(device)
    edge_index=data[('vertices', 'to', 'vertices')]['edge_index']
    coords,batch,face_norm,edge_whichface,face_x=data.pos, data['vertices'].batch,\
        data.f_norm,data[('edge', 'on', 'face')]['edge_index'][1,:],data["face"].x
    if face_x.shape[1]>3:
        if args.dataset in ["shapenet"]:
            face_x=face_x[:,[3,4,5,13]] #kd d
        else:
            face_x=face_x[:,3:]
        if args.maskface=="T":
            face_x=torch.zeros_like(face_x,device=device)
            face_norm=torch.zeros_like(face_norm,device=device)
    else:
        face_x=None
        if args.maskface=="T":
            face_norm=torch.zeros_like(face_norm,device=device)        
    label=data.y.long().view(-1)
    num_nodes=coords.shape[0]
    edge_index_2rd, num_triplets_real, edx_jk, edx_ij = util.triplets(edge_index, num_nodes)
    optimizer.zero_grad()
    input_feature=torch.zeros([coords.shape[0],args.h_ch],device=device) 
    output=model([input_feature,coords,face_norm,face_x,edge_whichface,edge_index, edge_index_2rd,edx_jk, edx_ij,batch,args.edge_rep])  
    output=torch.cat(output,dim=1)

    graph_embeddings=global_add_pool(output,batch)
    graph_embeddings.clamp_(max=1e6)
    c_loss=contrastive_loss(graph_embeddings,label,margin=1)
    output=mlpmodel(graph_embeddings)

    loss = criterion(output, label) 
    loss+=c_loss*args.loss_coef
    return loss,c_loss*args.loss_coef,output,label
def train(args,train_Loader,model,mlpmodel,optimizer,device,criterion ):
    epochloss=0
    epochcloss=0
    y_hat, y_true,y_hat_logit = [], [], []        
    optimizer.zero_grad()
    model.train()
    mlpmodel.train()
    for i,data in enumerate(train_Loader):
        loss,c_loss,output,label  =forward(args,data,model,mlpmodel,optimizer,device,criterion)

        loss.backward()
        optimizer.step()
        epochloss+=loss.detach().cpu()
        epochcloss+=c_loss.detach().cpu()
        
        _, pred = output.topk(1, dim=1, largest=True, sorted=True)
        pred,label,output=pred.cpu(),label.cpu(),output.cpu()
        y_hat += list(pred.detach().numpy().reshape(-1))
        y_true += list(label.detach().numpy().reshape(-1))
        y_hat_logit+=list(output.detach().numpy())
    return epochloss.item()/len(train_Loader),epochcloss.item()/len(train_Loader),y_hat, y_true,y_hat_logit

def test(args,loader,model,mlpmodel,optimizer,device,criterion ):
    y_hat, y_true,y_hat_logit = [], [], []

    loss_total, pred_num = 0, 0
    model.eval()
    mlpmodel.eval()
    with torch.no_grad():
        for data in loader:
            loss,c_loss,output,label  =forward(args,data,model,mlpmodel,optimizer,device,criterion)

            _, pred = output.topk(1, dim=1, largest=True, sorted=True)
            pred,label,output=pred.cpu(),label.cpu(),output.cpu()
            y_hat += list(pred.detach().numpy().reshape(-1))
            y_true += list(label.detach().numpy().reshape(-1))
            y_hat_logit+=list(output.detach().numpy())
            
            pred_num += len(label.reshape(-1, 1))
            loss_total += loss.detach() * len(label.reshape(-1, 1))
    return loss_total/pred_num,y_hat, y_true, y_hat_logit

def run_training(args,train_Loader,val_Loader,test_Loader):
    """
    build dir 
    """
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir,exist_ok=True)
    tensorboard_dir=os.path.join(args.save_dir,'log')
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir,exist_ok=True)
    model_dir=os.path.join(args.save_dir,'model')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir,exist_ok=True)    
    info_dir=os.path.join(args.save_dir,'info')
    if not os.path.exists(info_dir):
        os.makedirs(info_dir,exist_ok=True)  
    """
    set model seed 
    """
    # Seed = random.randint(1, 10000)
    Seed=3407
    print("Random Seed: ", Seed)
    print(args)
    random.seed(Seed)
    torch.manual_seed(Seed)  
    np.random.seed(Seed)  
                
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion=nn.CrossEntropyLoss()
    if args.dataset in ["building"]:
        x_out=10
        face_attri_size=0
    elif args.dataset in ["mnist_color"]:
        x_out=10
        face_attri_size=3
    elif args.dataset in ["shapenet"]:
        x_out=len(shapenet_class_names)
        face_attri_size=4
    elif args.dataset in ["modelnet"]:
        x_out=len(modelnet_class_names )
        face_attri_size=0       
    model=models.Smodel(h_channel=args.h_ch, face_attri_size=face_attri_size, \
                    localdepth=args.localdepth,num_interactions=args.num_interactions,\
                    finaldepth=args.finaldepth,batchnorm=args.batchnorm,edge_rep=args.edge_rep,geo_encoding_dim=int(args.geo_encoding))
    mlpmodel=MLP(in_channels=args.h_ch*args.num_interactions, hidden_channels=args.h_ch,out_channels=x_out, num_layers=args.classifier_depth,dropout=args.dropout)

    model.to(device), mlpmodel.to(device)
    opt_list=list(model.parameters())+list(mlpmodel.parameters())

    optimizer = torch.optim.Adam( opt_list, lr=args.lr)    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=args.patience, min_lr=1e-8)   

    best_val_trigger = -1
    old_lr=1e3
    suffix="{}{}-{}_{}".format(datetime.now().strftime("%h"),
                                    datetime.now().strftime("%d"),
                                    datetime.now().strftime("%H"),
                                    datetime.now().strftime("%M"))        
    if args.log: writer = SummaryWriter(os.path.join(tensorboard_dir,suffix))
    
    for epoch in range(args.nepoch):
        train_loss,train_closs,y_hat, y_true,y_hat_logit=train(args,train_Loader,model,mlpmodel,optimizer,device,criterion )
        
        train_acc=util.calculate(y_hat,y_true,y_hat_logit)
        try:util.record({"loss":train_loss,"closs":train_closs,"acc":train_acc},epoch,writer,"Train") 
        except: pass
        util.print_1(epoch,'Train',{"loss":train_loss,"closs":train_closs,"acc":train_acc})
        if epoch % args.test_per_round == 0:
            val_loss, yhat_val, ytrue_val, yhatlogit_val = test(args,val_Loader,model,mlpmodel,optimizer,device,criterion )
            test_loss, yhat_test, ytrue_test, yhatlogit_test = test(args,test_Loader,model,mlpmodel,optimizer,device,criterion )
            
            val_acc=util.calculate(yhat_val,ytrue_val,yhatlogit_val)
            try:util.record({"loss":val_loss,"acc":val_acc},epoch,writer,"Val")
            except: pass
            util.print_1(epoch,'Val',{"loss":val_loss,"acc":val_acc},color=blue) 
            test_acc=util.calculate(yhat_test,ytrue_test,yhatlogit_test)
            try:util.record({"loss":test_loss,"acc":test_acc},epoch,writer,"Test")            
            except: pass
            util.print_1(epoch,'Test',{"loss":test_loss,"acc":test_acc},color=blue)
            val_trigger=val_acc
            if val_trigger > best_val_trigger:
                best_val_trigger = val_trigger
                best_model = copy.deepcopy(model)
                best_mlpmodel=copy.deepcopy(mlpmodel)
                best_info=[epoch,val_trigger]
        """ 
        update lr when epoch≥30
        """
        if epoch >= 30:
            lr = scheduler.optimizer.param_groups[0]['lr']
            if old_lr!=lr:
                print(red('lr'), epoch, (lr), sep=', ')
                old_lr=lr
            scheduler.step(val_trigger)        
    """
    use best model to get best model result 
    """
    val_loss, yhat_val, ytrue_val, yhat_logit_val  = test(args,val_Loader,best_model,best_mlpmodel,optimizer,device,criterion)
    test_loss, yhat_test, ytrue_test, yhat_logit_test= test(args,test_Loader,best_model,best_mlpmodel,optimizer,device,criterion)

    val_acc=util.calculate(yhat_val,ytrue_val,yhat_logit_val)
    util.print_1(best_info[0],'BestVal',{"loss":val_loss,"acc":val_acc},color=blue)
    test_acc=util.calculate(yhat_test,ytrue_test,yhat_logit_test)
    util.print_1(best_info[0],'BestTest',{"loss":test_loss,"acc":test_acc},color=blue)
                                                            
    """
    save training info and best result 
    """
    if args.log:
        result_file=os.path.join(info_dir, suffix)
        with open(result_file, 'w') as f:
            print("Random Seed: ", Seed,file=f)
            last_train_acc=train_acc
            print(f"acc  val : {val_acc:.3f}, Test : {test_acc:.3f}, Train : {last_train_acc:.3f}", file=f)
            print(f"Best info: {best_info}", file=f)
            for i in [[a,getattr(args, a)] for a in args.__dict__]:
                print(i,sep='\n',file=f)
        to_save_dict={'model':best_model.state_dict(),'mlpmodel':best_mlpmodel.state_dict(),'args':args,'labels':ytrue_test,'yhat':yhat_test,'yhat_logit':yhat_logit_test}
        torch.save(to_save_dict, os.path.join(model_dir,suffix+'.pth') )
    print("done")   
    return  test_acc
def main():
    args = get_args()
    Seed = 0
    random.seed(Seed)
    torch.manual_seed(Seed)  
    np.random.seed(Seed) 
    if args.dataset in ["building"]:
        args.data_dir=os.path.join("data","building_graph") 
    elif args.dataset in ["mnist_color"]:
        args.data_dir=os.path.join("data","mnist_color_graph") 
    elif args.dataset in ["shapenet"]:
        args.data_dir=os.path.join("data","shapenet_graph") 
    elif args.dataset in ["modelnet"]:
        args.data_dir=os.path.join("data","modelnet_graph")            
    if  args.dataset in ["mnist_color",'building','shapenet',"modelnet"]:
        train_ds,val_ds,test_ds=datasets.get_dataset(args.data_dir)
          
    if args.rotate=="T":
        np.random.seed(42)
        train_ds,val_ds,test_ds=datasets.rotate_ds(train_ds),datasets.rotate_ds(val_ds),datasets.rotate_ds(test_ds)
    if args.extent_norm=="T":
        train_ds= datasets.affine_transform_to_range(train_ds,target_range=(-1, 1))
        val_ds= datasets.affine_transform_to_range(val_ds,target_range=(-1, 1))
        test_ds= datasets.affine_transform_to_range(test_ds,target_range=(-1, 1))
    def rename_face_norm(ds):
        for data in ds:
            data['f_norm']=data['face_norm']
            del  data['face_norm'] 
        return ds 
    train_ds,val_ds,test_ds= rename_face_norm(train_ds) ,rename_face_norm(val_ds),    rename_face_norm(test_ds)  
    validity_t=util.check_hetero_dataset(train_ds)
    validity_v=util.check_hetero_dataset(val_ds)
    validity_t=util.check_hetero_dataset(test_ds)
                    
    train_loader = torch_geometric.loader.DataLoader(train_ds,batch_size=args.train_batch, shuffle=False,pin_memory=True,drop_last=True) 
    val_loader = torch_geometric.loader.DataLoader(val_ds, batch_size=args.test_batch, shuffle=False, pin_memory=True)
    test_loader = torch_geometric.loader.DataLoader(test_ds,batch_size=args.test_batch, shuffle=False,pin_memory=True)
    test_acc=run_training(args,train_loader,val_loader,test_loader)


if __name__ == '__main__':
    main()
    