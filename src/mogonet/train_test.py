""" Training and testing of the model
"""
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn.functional as F
from mogonet.models import init_model_dict, init_optim
from mogonet.utils import one_hot_tensor, cal_sample_weight, gen_adj_mat_tensor, gen_test_adj_mat_tensor, cal_adj_mat_parameter, getViewList, check_adj_para


cuda = True if torch.cuda.is_available() else False

# Helper function to prepare the data
def prepare_trte_data(data_folder, view_list=None):
    """
    Gets all the *tr.csv and *te.csv in the data_folder, and transforms these to list of tensors, then
    storing it on several returned objects

    Parameters
    ----------
        data_folder: string
            path to read the data
        view_list: list
            list of integers to be viewed, [1,2,3] here

    Returns
    ---------
        data_train_list: list
            list of tensors of the train data
        data_all_list: list
            list of tensors of combined train and test data
        idx_dict: dict
            dict that corresponds to the label (id) of both train,
            and test data
        labels: np.ndarray
            numpy array that stores the actual class of each observation
    """
    if view_list is None:
        print("Did not provide custom view list, finding it now")
        view_list = getViewList(data_folder)
    num_view = len(view_list)
    # Get the labels and transform it to integer to map it
    labels_tr = np.loadtxt(os.path.join(data_folder, "labels_tr.csv"), delimiter=',').astype(int)
    labels_te = np.loadtxt(os.path.join(data_folder, "labels_te.csv"), delimiter=',').astype(int)
    # Initialize list to store results
    data_tr_list = []
    data_te_list = []

    # Reads the data in the csv files with _tr / _te
    # And append it correspondently to its list
    for i in view_list:
        data_tr_list.append(np.loadtxt(os.path.join(data_folder, str(i)+"_tr.csv"), delimiter=','))
        data_te_list.append(np.loadtxt(os.path.join(data_folder, str(i)+"_te.csv"), delimiter=','))
    num_tr = data_tr_list[0].shape[0]
    num_te = data_te_list[0].shape[0]
    data_mat_list = []
    for i in range(num_view):
        data_mat_list.append(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0))
    data_tensor_list = []

    for i in range(len(data_mat_list)):
        data_tensor_list.append(torch.FloatTensor(data_mat_list[i]))
        if cuda:
            data_tensor_list[i] = data_tensor_list[i].cuda()
    idx_dict = {}
    idx_dict["tr"] = list(range(num_tr))
    idx_dict["te"] = list(range(num_tr, (num_tr+num_te)))
    data_train_list = []
    data_all_list = []
    for i in range(len(data_tensor_list)):
        data_train_list.append(data_tensor_list[i][idx_dict["tr"]].clone())
        data_all_list.append(torch.cat((data_tensor_list[i][idx_dict["tr"]].clone(),
                                       data_tensor_list[i][idx_dict["te"]].clone()),0))
    labels = np.concatenate((labels_tr, labels_te))

    return data_train_list, data_all_list, idx_dict, labels


def gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter):
    adj_metric = "cosine" # cosine distance
    adj_train_list = []
    adj_test_list = []
    for i in range(len(data_tr_list)):
        adj_parameter_adaptive = cal_adj_mat_parameter(adj_parameter, data_tr_list[i], adj_metric)
        adj_train_list.append(gen_adj_mat_tensor(data_tr_list[i], adj_parameter_adaptive, adj_metric))
        adj_test_list.append(gen_test_adj_mat_tensor(data_trte_list[i], trte_idx, adj_parameter_adaptive, adj_metric))

    return adj_train_list, adj_test_list


def train_epoch(data_list, adj_list, label, one_hot_label, sample_weight, model_dict, optim_dict, train_VCDN=True):
    loss_dict = {}
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    for m in model_dict:
        model_dict[m].train()
    num_view = len(data_list)
    for i in range(num_view):
        optim_dict["C{:}".format(i+1)].zero_grad()
        ci_loss = 0
        ci = model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_list[i],adj_list[i]))
        ci_loss = torch.mean(torch.mul(criterion(ci, label),sample_weight))
        ci_loss.backward()
        optim_dict["C{:}".format(i+1)].step()
        loss_dict["C{:}".format(i+1)] = ci_loss.detach().cpu().numpy().item()
    if train_VCDN and num_view >= 2:
        optim_dict["C"].zero_grad()
        c_loss = 0
        ci_list = []
        for i in range(num_view):
            ci_list.append(model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_list[i],adj_list[i])))
        c = model_dict["C"](ci_list)
        c_loss = torch.mean(torch.mul(criterion(c, label),sample_weight))
        c_loss.backward()
        optim_dict["C"].step()
        loss_dict["C"] = c_loss.detach().cpu().numpy().item()

    return loss_dict


def test_epoch(data_list, adj_list, te_idx, model_dict):
    for m in model_dict:
        model_dict[m].eval()
    num_view = len(data_list)
    ci_list = []
    for i in range(num_view):
        ci_list.append(model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_list[i],adj_list[i])))
    if num_view >= 2:
        c = model_dict["C"](ci_list)
    else:
        c = ci_list[0]
    c = c[te_idx,:]
    prob = F.softmax(c, dim=1).data.cpu().numpy()

    return prob

# decoupled to two functions

def train_model():

# This is the main function
def train_test(
    data_folder,
    lr_e_pretrain, lr_e, lr_c,
    num_epoch_pretrain, num_epoch, view_list=None, 
    num_class=2,test_interval=50, adj_parameter=8
                ):
    """
    Parameters:

    data_folder:            Path to to contain directory structure as shown in the docs, should contain blocks_tr.csv
                            blocks_te.csv, labels_tr.csv, labels_te.csv.
    view_list:              Names of the omics/block name, provided cli arg from nextflow
    num_class:              Levels in the response class, could also be provided from upstream process
    lr_e_pretrain:          Learning for pre train epochs
    lr_e:                   Learning rate of non-pre train epoch
    lr_c:                   Learning rate ....
    num_epoch_pretrain:     Number of epochs to supply for pretrain
    test_interval:          Interval to test epoch, recommend 50 for demo, 200 for cluster
    adj_parameter:          Hyperparameter for adjacency matrix, tuneable, FIX LATER
    """
    # Check if view_list provided or not 
    if view_list is None:
        print("Did not provide custom view list, finding it now")
        view_list = getViewList(data_folder)
    # Check if cuda used or not
    if cuda:
        print("Found cuda, using GPU")
    else:
        print("Cuda not found, using CPU")
    
    # Parameters 
    num_view = len(view_list)
    dim_hvcdn = pow(num_class,num_view)
    # ----------------------------------
    # Modify LATER!!!!!!!
    dim_he_list = [50] * num_view # Need a more robust way to decide this
    # But it should be of length N (numbers of blocks), so might need to column number
    # in each block minus some constant, such this number < column numebr of specific block
    # ---------------------------------
    data_tr_list, data_trte_list, trte_idx, labels_trte = prepare_trte_data(data_folder, view_list)
    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)
    sample_weight_tr = cal_sample_weight(labels_trte[trte_idx["tr"]], num_class)
    sample_weight_tr = torch.FloatTensor(sample_weight_tr)
    if cuda:
        labels_tr_tensor = labels_tr_tensor.cuda()
        onehot_labels_tr_tensor = onehot_labels_tr_tensor.cuda()
        sample_weight_tr = sample_weight_tr.cuda()
    # TODO: FIX THIS AND MAKE IT MORE VERBOSE
    # adj_parameter needs to be <= numples of rows in train data
    adj_parameter = check_adj_param_size(adj_parameter, data_tr_list)
    adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter)

    # Feature dimension of each block in train data list
    dim_list = [x.shape[1] for x in data_tr_list]
    # Initialize a model dictionary to update later
    # Dim_he_list could be tuneable like [some_tune_num] * num_view or different numbers of each view
    # like [k1, k2, k3, ... ] or just [k, k, k, ...], to list of length is equal to num_view
    model_dict = init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hvcdn)
    for m in model_dict:
        if cuda:
            model_dict[m].cuda()

    print("\nPretrain GCNs...")
    optim_dict = init_optim(num_view, model_dict, lr_e_pretrain, lr_c)
    for epoch in range(num_epoch_pretrain):
        # Notice the model_dict is being changed under the hood
        # for every model_dict[m].state_dict()
        train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor,
                    onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict, train_VCDN=False)
    print("\nTraining...")
    optim_dict = init_optim(num_view, model_dict, lr_e, lr_c)
    for epoch in range(num_epoch+1):
        # Notice the model_dict is being changed under the hood
        # for every model_dict[m].state_dict()
        train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor,
                    onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict)
        if epoch % test_interval == 0:
            te_prob = test_epoch(data_trte_list, adj_te_list, trte_idx["te"], model_dict)
            print("\nTest: Epoch {:d}".format(epoch))
            if num_class == 2:
                print("Test ACC: {:.3f}".format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                print("Test F1: {:.3f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                print("Test AUC: {:.3f}".format(roc_auc_score(labels_trte[trte_idx["te"]], te_prob[:,1])))
            else:
                print("Test ACC: {:.3f}".format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                print("Test F1 weighted: {:.3f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='weighted')))
                print("Test F1 macro: {:.3f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')))