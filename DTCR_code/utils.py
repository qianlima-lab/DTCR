# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 15:29:48 2016

@author: root
"""
import tensorflow as tf
import numpy as np
import sys, time
from scipy.misc import comb
from sklearn import metrics


def transfer_labels(labels):
    indexes = np.unique(labels)
    num_classes = indexes.shape[0]
    num_samples = labels.shape[0]

    for i in range(num_samples):
        new_label = np.argwhere(labels[i] == indexes)[0][0]
        labels[i] = new_label
    return labels, num_classes


def get_fake_sample(data):
    sample_nums = data.shape[0]
    series_len = data.shape[1]
    mask = np.ones(shape=[sample_nums, series_len])
    rand_list = np.zeros(shape=[sample_nums, series_len])

    fake_position_nums = int(series_len * 0.2)
    fake_position = np.random.randint(low=0, high=series_len, size=[sample_nums, fake_position_nums])

    for i in range(fake_position.shape[0]):
        for j in range(fake_position.shape[1]):
            mask[i, fake_position[i, j]] = 0

    for i in range(rand_list.shape[0]):
        count = 0
        for j in range(rand_list.shape[1]):
            if j in fake_position[i]:
                rand_list[i, j] = data[i, fake_position[i, count]]
                count += 1
    fake_data = data * mask + rand_list * (1 - mask)
    real_fake_labels = np.zeros(shape=[sample_nums * 2, 2])
    for i in range(sample_nums * 2):
        if i < sample_nums:
            real_fake_labels[i, 0] = 1
        else:
            real_fake_labels[i, 1] = 1
    return fake_data, real_fake_labels


def _rnn_reformat(x, input_dims, n_steps):
    """
    This function reformat input to the shape that standard RNN can take. 
    Inputs:
        x -- a tensor of shape (batch_size, n_steps, input_dims).
    Outputs:
        x_reformat -- a list of 'n_steps' tenosrs, each has shape (batch_size, input_dims).
    """
    # permute batch_size and n_steps
    x_ = tf.transpose(x, [1, 0, 2])
    # reshape to (n_steps*batch_size, input_dims)
    x_ = tf.reshape(x_, [-1, input_dims])
    # split to get a list of 'n_steps' tensors of shape (batch_size, input_dims)
    x_reformat = tf.split(x_, n_steps, 0)
    return x_reformat


def _rnn_reformat_denoise(x, input_dims, n_steps, batch_size):
    """
    This function reformat input to the shape that standard RNN can take. 
    Inputs:
        x -- a tensor of shape (batch_size, n_steps, input_dims).
    Outputs:
        x_reformat -- a list of 'n_steps' tenosrs, each has shape (batch_size, input_dims).
    """
    # x_ = x +  np.random.uniform(-1,1,(batch_size, n_steps, input_dims))
    # x_ = x +  np.random.normal(size=(batch_size, n_steps, input_dims))
    # permute batch_size and n_steps
    x_ = tf.transpose(x, [1, 0, 2])
    # reshape to (n_steps*batch_size, input_dims)
    x_ = tf.reshape(x_, [-1, input_dims])
    # split to get a list of 'n_steps' tensors of shape (batch_size, input_dims)
    x_reformat = tf.split(x_, n_steps, 0)
    return x_reformat


def load_data(filename):
    data_label = np.loadtxt(filename, delimiter=',')
    data = data_label[:, 1:]
    label = data_label[:, 0].astype(np.int32)
    return data, label


def evaluation(prediction, label):
    acc = cluster_acc(label, prediction)
    nmi = metrics.normalized_mutual_info_score(label, prediction)
    ari = metrics.adjusted_rand_score(label, prediction)
    ri = rand_index_score(label, prediction)
    print((acc, nmi, ari, ri))
    return ri, nmi, acc


def rand_index_score(clusters, classes):
    tp_plus_fp = comb(np.bincount(clusters), 2).sum()
    tp_plus_fn = comb(np.bincount(classes), 2).sum()
    A = np.c_[(clusters, classes)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(clusters))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def next_batch(batch_size, data):
    # assert data.shape[0] == label.shape[0]
    row = data.shape[0]
    batch_len = int(row / batch_size)
    left_row = row - batch_len * batch_size

    for i in range(batch_len):
        batch_input = data[i * batch_size: (i + 1) * batch_size, :]
        yield (batch_input, False)

    if left_row != 0:
        need_more = batch_size - left_row
        need_more = np.random.choice(np.arange(row), size=need_more)
        yield (np.concatenate((data[-left_row:, :], data[need_more]), axis=0), True)


