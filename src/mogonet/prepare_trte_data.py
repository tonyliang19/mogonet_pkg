import torch
import numpy as np
import os
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
    cuda = True if torch.cuda.is_available() else False
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