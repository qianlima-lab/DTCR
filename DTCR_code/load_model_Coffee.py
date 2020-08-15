# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 15:29:48 2016

@author: root
"""
import tensorflow as tf
import utils
import math
import sys
import os
import numpy as np
import copy
import drnn
import rnn_cell_extensions
from tensorflow.python.ops import variable_scope
from sklearn import metrics
from sklearn.cluster import KMeans
from numpy import linalg as LA
import warnings
import shutil


class Config(object):
    """Train config."""
    batch_size = None
    hidden_size = [100, 50, 50]
    dilations = [1, 2, 4]
    num_steps = None
    embedding_size = None
    learning_rate = 1e-4
    cell_type = 'GRU'
    lamda = 1
    class_num = None
    denosing = True  # False
    sample_loss = True  # False


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


class RNN_clustering_model(object):

    def __init__(self, config):
        self.batch_size = config.batch_size
        self.hidden_size = config.hidden_size
        self.dilations = config.dilations
        self.num_steps = config.num_steps
        self.embedding_size = config.embedding_size
        self.cell_type = config.cell_type
        self.lamda = config.lamda
        self.class_num = config.class_num
        self.denosing = config.denosing
        self.sample_loss = config.sample_loss
        self.K = config.class_num

    # self.fully_units = config.fully_units

    def build_model(self):
        input = tf.placeholder(tf.float32, [None, self.num_steps], name='inputs')  # input
        noise = tf.placeholder(tf.float32, [None, self.num_steps], name='noise')

        real_fake_label = tf.placeholder(tf.float32, [None, 2], name='real_fake_label')

        F_new_value = tf.placeholder(tf.float32, [None, self.K], name='F_new_value')
        # F = tf.Variable(tf.eye(self.batch_size,num_columns = self.K), trainable = False)
        F = tf.get_variable('F', shape=[self.batch_size, self.K],
                            initializer=tf.orthogonal_initializer(gain=1.0, seed=None, dtype=tf.float32),
                            trainable=False)

        # inputs has shape (batch_size, n_steps, embedding_size)
        inputs = tf.reshape(input, [-1, self.num_steps, self.embedding_size])
        noises = tf.reshape(noise, [-1, self.num_steps, self.embedding_size])

        # a list of 'n_steps' tenosrs, each has shape (batch_size, embedding_size)
        # encoder_inputs = utils._rnn_reformat(x = inputs, input_dims = self.embedding_size, n_steps = self.num_steps)

        # noise_input has shape (batch_size, n_steps, embedding_size)
        if self.denosing:
            print('Noise')
            noise_input = inputs + noises
        else:
            print('Non_noise')
            noise_input = inputs

        reverse_noise_input = tf.reverse(noise_input, axis=[1])
        decoder_inputs = utils._rnn_reformat(x=noise_input, input_dims=self.embedding_size, n_steps=self.num_steps)
        targets = utils._rnn_reformat(x=inputs, input_dims=self.embedding_size, n_steps=self.num_steps)

        if self.cell_type == 'LSTM':
            raise ValueError('LSTMs have not support yet!')
        # cell = tf.contrib.rnn.BasicLSTMCell( np.sum(self.hidden_size) *2)
        elif self.cell_type == 'GRU':
            cell = tf.contrib.rnn.GRUCell(np.sum(self.hidden_size) * 2)

        cell = rnn_cell_extensions.LinearSpaceDecoderWrapper(cell, self.embedding_size)

        lf = None
        if self.sample_loss:
            print
            'Sample Loss'

            def lf(prev, i):
                return prev

        # encoder_output has shape 'layer' list of tensor [batch_size, n_steps, hidden_size]
        with tf.variable_scope('fw'):
            _, encoder_output_fw = drnn.drnn_layer_final(noise_input, self.hidden_size, self.dilations, self.num_steps,
                                                         self.embedding_size, self.cell_type)
        
        with tf.variable_scope('bw'):
            _, encoder_output_bw = drnn.drnn_layer_final(reverse_noise_input, self.hidden_size, self.dilations,
                                                         self.num_steps, self.embedding_size, self.cell_type)

        if self.cell_type == 'LSTM':
            raise ValueError('LSTMs have not support yet!')
        elif self.cell_type == 'GRU':
            fw = []
            bw = []
            for i in range(len(self.hidden_size)):
                fw.append(encoder_output_fw[i][:, -1, :])
                bw.append(encoder_output_bw[i][:, -1, :])
            encoder_state_fw = tf.concat(fw, axis=1)
            encoder_state_bw = tf.concat(bw, axis=1)

            # encoder_state has shape [batch_size, sum(hidden_size)*2]
            encoder_state = tf.concat([encoder_state_fw, encoder_state_bw], axis=1)

        decoder_outputs, _ = tf.contrib.legacy_seq2seq.rnn_decoder(decoder_inputs=decoder_inputs,
                                                                   initial_state=encoder_state, cell=cell,
                                                                   loop_function=lf)

        if self.cell_type == 'LSTM':
            hidden_abstract = encoder_state.h
        elif self.cell_type == 'GRU':
            hidden_abstract = encoder_state

        # F_update
        F_update = tf.assign(F, F_new_value)

        real_hidden_abstract = tf.split(hidden_abstract, 2)[0]

        # W has shape [sum(hidden_size)*2, batch_size]
        W = tf.transpose(real_hidden_abstract)
        WTW = tf.matmul(real_hidden_abstract, W)
        FTWTWF = tf.matmul(tf.matmul(tf.transpose(F), WTW), F)

        with tf.name_scope("loss_reconstruct"):
            loss_reconstruct = tf.losses.mean_squared_error(labels=tf.split(targets, 2, axis=1)[0], predictions=tf.split(decoder_outputs, 2, axis=1)[0])

        with tf.name_scope("k-means_loss"):
            loss_k_means = tf.trace(WTW) - tf.trace(FTWTWF)

        with tf.name_scope("discriminative_loss"):
            weight1 = weight_variable(shape=[hidden_abstract.get_shape().as_list()[1], 128])
            bias1 = bias_variable(shape=[128])

            weight2 = weight_variable(shape=[128, 2])
            bias2 = bias_variable(shape=[2])

            hidden = tf.nn.relu(tf.matmul(hidden_abstract, weight1) + bias1)
            output = tf.matmul(hidden, weight2) + bias2
            predict = tf.reshape(output, shape=[-1, 2])
	    discriminative_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=real_fake_label))

        with tf.name_scope("loss_total"):
            loss = loss_reconstruct + self.lamda / 2 * loss_k_means + discriminative_loss

        regularization_loss = 0.0
        for i in range(len(tf.trainable_variables())):
            regularization_loss += tf.nn.l2_loss(tf.trainable_variables()[i])
        loss = loss + 1e-4 * regularization_loss
        input_tensors = {
            'inputs': input,
            'noise': noise,
            'F_new_value': F_new_value,
            'real_fake_label': real_fake_label
        }
        loss_tensors = {
            'loss_reconstruct': loss_reconstruct,
            'loss_k_means': loss_k_means,
            'regularization_loss': regularization_loss,
            'discriminative_loss': discriminative_loss,
            'loss': loss
        }
        output_tensor = {'prediction': predict}
        return input_tensors, loss_tensors, hidden_abstract, F_update, output_tensor


def run_model(filename, config):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True

    testing_data, testing_label = utils.load_data(filename)
    testing_label, num_classes = utils.transfer_labels(testing_label)

    config.class_num = num_classes
    config.embedding_size = 1
    config.batch_size = testing_data.shape[0]
    config.num_steps = testing_data.shape[1]

    test_noise_data = np.zeros(shape=testing_data.shape)
    
    with tf.Session(config=gpu_config) as sess:
        model = RNN_clustering_model(config=config)
        input_tensors, loss_tensors, hidden_abstract, F_update, output_tensor = model.build_model()

        sess.run(tf.global_variables_initializer())
        
        model_path = './Model/model.ckpt'
	saver = tf.train.Saver()
	saver.restore(sess, model_path)
	 
	test_total_abstract = sess.run(hidden_abstract,
				feed_dict={input_tensors['inputs']: testing_data, 
				input_tensors['noise']: test_noise_data
				})

	test_hidden_val = np.array(test_total_abstract).reshape(-1, np.sum(config.hidden_size) * 2)
        km = KMeans(n_clusters=num_classes)
	km_idx = km.fit_predict(test_hidden_val)
	ri, nmi, acc = utils.evaluation(prediction=km_idx, label=testing_label)
	
def main():
    config = Config()
    filename = './Coffee/Coffee_TEST'
    config.lamda = 1e-1
    config.hidden_size = [100,50,50]
    config.dilations = [1,2,4]
	
    run_model(filename, config)

if __name__ == "__main__":
    main()
