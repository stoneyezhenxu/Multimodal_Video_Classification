#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : train.py
# @Author: stoneye
# @Date  : 2023/09/01
# @Contact : stoneyezhenxu@gmail.com


import math

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib import rnn



class TrnModel():
    """
    Trn model: light-weight model
     1. A new attention layer is added to replace the pool layer.
     2. Added residual connection
    """

    def forward(self, is_training, fea_input,
                dropout_keep_prob, fea_type,
                true_frames, max_frames, name_scope,
                cluster_size=256, rgb_fea_size=1536,
                audio_fea_size=128, aver_pool_size=5):
        """
        :param is_training:   #train:True;  test|valid:False
        :param fea_input:     # rgb_fea or audio_fea
        :param dropout_keep_prob:
        :param fea_type:     #rgb or audio
        :param true_frames:  #
        :param max_frames:   #
        :param name_scope:   #
        :param cluster_size:
        :param rgb_fea_size:
        :param audio_fea_size:
        :param aver_pool_size:
        :return:
        """
        with tf.variable_scope(name_scope):
            if fea_type == "rgb":
                fea_size = rgb_fea_size
            elif fea_type == "audio":
                fea_size = audio_fea_size
            else:
                raise ValueError('fea_type must be [rgb,audio]')

            model_input = fea_input  # [batch_size,frame,fea_size]
            frame_nums = max_frames
            video_attention = AttentionLayers(feature_size=fea_size,
                                               frame_nums=frame_nums,
                                               cluster_size=cluster_size)
            aver_mean = video_attention.forward(model_input, is_training)

            input_x_mlp = tf.layers.average_pooling1d(model_input,
                                                      pool_size=aver_pool_size,
                                                      strides=1,
                                                      padding='SAME',
                                                      name="trn_rgb_max_pool")
            # [batch,segment,fea_size/2]
            trn_layer1 = slim.fully_connected(input_x_mlp,
                                              num_outputs=fea_size // 2,
                                              scope="trn_later1_rgb")

            # [batch,segment *fea_size/2]
            trn_layer1_reshape = tf.reshape(trn_layer1, [-1, frame_nums * fea_size // 2])
            trn_layer1_reshape = tf.nn.dropout(trn_layer1_reshape, dropout_keep_prob)
            trn_layer2 = slim.fully_connected(trn_layer1_reshape,
                                              num_outputs=fea_size // 2,
                                              scope="trn_later2_rgb")
            trn_layer3 = slim.fully_connected(trn_layer2,
                                              num_outputs=fea_size,
                                              activation_fn=None,
                                              scope="trn_later3_rgb")
            aver_mean_layer4 = slim.fully_connected(aver_mean,
                                                    num_outputs=fea_size,
                                                    activation_fn=None,
                                                    scope="trn_later4_rgb")
            trn_feature = tf.nn.relu(trn_layer3 + aver_mean_layer4)
            return trn_feature


class Nextvlad():
    """
    nextvlad model: heavy-model
    Core idea: Introduce the idea of k-means clustering into inter-frame fusion.
    """

    def __init__(self, feature_size, max_frames,
                 cluster_size, is_training=True,
                 expansion=2, groups=None):
        """
        :param feature_size:   #rgb:1536; audio:128
        :param max_frames:     #
        :param cluster_size:   #the size of cluster
        :param is_training:    #train:True; test|valid:False
        :param expansion:      #
        :param groups:         #
        """
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.cluster_size = cluster_size
        self.expansion = expansion
        self.groups = groups

    def forward(self, input, mask=None):
        input = slim.fully_connected(input,
                                     self.expansion * self.feature_size,
                                     activation_fn=None,
                                     weights_initializer=slim.variance_scaling_initializer())

        attention = slim.fully_connected(input,
                                         self.groups,
                                         activation_fn=tf.nn.sigmoid,
                                         weights_initializer=slim.variance_scaling_initializer())
        if mask is not None:
            attention = tf.multiply(attention, tf.expand_dims(mask, -1))
        attention = tf.reshape(attention, [-1, self.max_frames * self.groups, 1])
        tf.summary.histogram("sigmoid_attention", attention)
        feature_size = self.expansion * self.feature_size // self.groups

        cluster_weights = tf.get_variable("cluster_weights",
                                          [self.expansion * self.feature_size,
                                           self.groups * self.cluster_size],
                                          initializer=slim.variance_scaling_initializer())

        reshaped_input = tf.reshape(input, [-1, self.expansion * self.feature_size])
        activation = tf.matmul(reshaped_input, cluster_weights)

        activation = slim.batch_norm(
            activation,
            center=True,
            scale=True,
            is_training=self.is_training,
            scope="cluster_bn",
            fused=False)

        activation = tf.reshape(activation, [-1, self.max_frames * self.groups, self.cluster_size])
        activation = tf.nn.softmax(activation, axis=-1)
        activation = tf.multiply(activation, attention)
        a_sum = tf.reduce_sum(activation, -2, keep_dims=True)
        cluster_weights2 = tf.get_variable("cluster_weights2",
                                           [1, feature_size, self.cluster_size],
                                           initializer=slim.variance_scaling_initializer())
        a = tf.multiply(a_sum, cluster_weights2)
        activation = tf.transpose(activation, perm=[0, 2, 1])
        reshaped_input = tf.reshape(input, [-1, self.max_frames * self.groups, feature_size])
        vlad = tf.matmul(activation, reshaped_input)
        vlad = tf.transpose(vlad, perm=[0, 2, 1])
        vlad = tf.subtract(vlad, a)
        vlad = tf.nn.l2_normalize(vlad, 1)
        vlad = tf.reshape(vlad, [-1, self.cluster_size * feature_size])
        vlad = slim.batch_norm(vlad,
                               center=True,
                               scale=True,
                               is_training=self.is_training,
                               scope="vlad_bn",
                               fused=False)

        return vlad


class NextvladModel():
    """
    nextvlad implementation
    """

    def forward(self, is_training, fea_input,
                dropout_keep_prob, fea_type,
                true_frames, max_frames, name_scope,
                expansion=2, hidden1_size=1024,
                gating_reduction=8, rgb_fea_size=1536,
                audio_fea_size=128, rgb_cluster_size=256,
                audio_cluster_size=128, rgb_groups=8,
                audio_groups=4):
        """
        :param is_training:     #train:True; test|valid:False
        :param fea_input:       # [batch,frame,fea_size]
        :param dropout_keep_prob: #keep dropout radio
        :param fea_type:        #rgb or audio
        :param true_frames:     #
        :param max_frames:      #
        :param name_scope:      #
        :param expansion:       #
        :param hidden1_size:    #
        :param gating_reduction: #
        :param rgb_fea_size:     # 1536
        :param audio_fea_size:   #128
        :param rgb_cluster_size: #256
        :param audio_cluster_size: #128
        :param rgb_groups:       #8
        :param audio_groups:     #4
        :return:
        """

        with tf.variable_scope(name_scope):
            if fea_type == 'rgb':
                fea_size = rgb_fea_size
                cluster_size = rgb_cluster_size
                groups = rgb_groups
            elif fea_type == 'audio':
                fea_size = audio_fea_size
                cluster_size = audio_cluster_size
                groups = audio_groups
            expansion = expansion
            hidden1_size = hidden1_size
            gating_reduction = gating_reduction
            mask = tf.sequence_mask(true_frames, max_frames, dtype=tf.float32)
            nextvlad_obj = Nextvlad(fea_size,
                                    max_frames,
                                    cluster_size,
                                    is_training,
                                    groups=groups,
                                    expansion=expansion)
            vlad = nextvlad_obj.forward(fea_input, mask=mask)
            vlad = slim.dropout(vlad, keep_prob=dropout_keep_prob,
                                is_training=is_training,
                                scope="vlad_dropout")
            vlad_dim = vlad.get_shape().as_list()[1]
            hidden1_weights = tf.get_variable("hidden1_weights",
                                              [vlad_dim, hidden1_size],
                                              initializer=slim.variance_scaling_initializer())

            activation = tf.matmul(vlad, hidden1_weights)
            activation = slim.batch_norm(
                activation,
                center=True,
                scale=True,
                is_training=is_training,
                scope="hidden1_bn",
                fused=False)

            gating_weights_1 = tf.get_variable("gating_weights_1",
                                               [hidden1_size, hidden1_size // gating_reduction],
                                               initializer=slim.variance_scaling_initializer())
            gates = tf.matmul(activation, gating_weights_1)
            gates = slim.batch_norm(
                gates,
                center=True,
                scale=True,
                is_training=is_training,
                activation_fn=slim.nn.relu,
                scope="gating_bn")

            gating_weights_2 = tf.get_variable("gating_weights_2",
                                               [hidden1_size // gating_reduction, hidden1_size],
                                               initializer=slim.variance_scaling_initializer())
            gates = tf.matmul(gates, gating_weights_2)
            gates = tf.sigmoid(gates)
            activation = tf.multiply(activation, gates)
            return activation


class TextExactor():
    """
    Text-related model:
        including implementations of Bi-LSTM and TextCNN
    """

    def _bilstm_feature(self, embedding_descript,
                        hidden_size, des_sequence_length,
                        dtype=tf.float32, reuse=None):
        """
        Bi-LSTM
        :param embedding_descript:   #[batch,seq_len,emb_size]
        :param hidden_size:          #
        :param des_sequence_length:  #seq_len
        :param dtype:                #
        :param reuse:                #
        :return:
        """
        with tf.variable_scope('bilstm_feature', reuse=reuse):
            lstm_fw_cell = rnn.BasicLSTMCell(hidden_size)
            lstm_bw_cell = rnn.BasicLSTMCell(hidden_size)
            _, states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=lstm_fw_cell,
                cell_bw=lstm_bw_cell,
                inputs=embedding_descript,  # [batch,max_len,emb_size=128]
                sequence_length=des_sequence_length,
                time_major=False,
                dtype=dtype)
            states_fw, states_bw = states
            _, h_fw = states_fw
            _, h_bw = states_bw
            bilstm_final_h = tf.concat([h_fw, h_bw], -1)  # [batch,2h]
        return bilstm_final_h

    def _textcnn_feature(self, embedding_descript,
                         embedding_size, filter_sizes,
                         num_filters, reuse=None,
                         dropout_keep_prob=1.0):
        """
         TextCnn
        :param embedding_descript:  #[batch,seq_len,emb_size]
        :param embedding_size:      #
        :param filter_sizes:        #The height of the convolution kernel --> extract n-gram features
        :param num_filters:         #the nums of convolution kernel
        :param reuse:               #
        :param dropout_keep_prob:   #
        :return:
        """

        with tf.variable_scope('CNN_feature', reuse=reuse):
            # [batch,max_len,emb_size,1]
            embedded_descript_expanded = tf.expand_dims(embedding_descript, -1)
            pooled_outputs_descript = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.variable_scope("conv-maxpool-%s" % filter_size):
                    filter_shape = [filter_size, embedding_size, 1, num_filters]
                    kernel_w = tf.get_variable("kernel_w", shape=filter_shape,
                                               initializer=tf.truncated_normal_initializer(stddev=0.1),
                                               dtype=tf.float32)
                    kernel_b = tf.get_variable("kernel_b", shape=[num_filters],
                                               initializer=tf.constant_initializer(0.1),
                                               dtype=tf.float32)
                    embedded_descript_expanded = tf.cast(embedded_descript_expanded,
                                                         dtype=tf.float32)
                    conv_descript = tf.nn.conv2d(embedded_descript_expanded,
                                                 kernel_w,
                                                 strides=[1, 1, 1, 1],
                                                 padding="VALID",
                                                 name="conv_descript_feature")
                    app_descript_feature = tf.nn.relu(tf.nn.bias_add(conv_descript, kernel_b),
                                                      name="relu")
                    # [batch,max_len-filter_size+1,1,num_filters]

                    seq_len = app_descript_feature.get_shape().as_list()[1]
                    pooled_app_descript_feature = tf.nn.max_pool(app_descript_feature,
                                                                 ksize=[1, seq_len, 1, 1],
                                                                 strides=[1, 1, 1, 1],
                                                                 padding='VALID',
                                                                 name="pool_q1")
                    # [batch,1,num_filters]
                    pooled_outputs_descript.append(pooled_app_descript_feature)
            num_filters_total = num_filters * len(filter_sizes)
            # [batch,1, num_filters_total]
            h_pool_descript = tf.concat(pooled_outputs_descript, 3)
            h_drop_app_descript = tf.reshape(h_pool_descript,
                                             [-1, num_filters_total],
                                             name="descript_flat")
            return h_drop_app_descript


class AttentionLayers():
    def __init__(self, feature_size, frame_nums, cluster_size):
        """
        Map the frame number to cluster_size using the attention mechanism
        [batch,frame_nums,feature_size] --> [batch,cluster_size,feature_size]
        :param feature_size:  #rgb、audio
        :param frame_nums:    #frame_nums
        :param cluster_size:  # the size of cluster
        """
        self.feature_size = feature_size
        self.frame_nums = frame_nums
        self.cluster_size = cluster_size

    def forward(self, model_input, is_training):
        instance = model_input
        instance = tf.reshape(instance, [-1, self.feature_size])
        dr_weights = tf.get_variable("dr_weights",
                                     [self.feature_size, self.cluster_size],
                                     initializer=tf.random_normal_initializer(
                                         stddev=1 / math.sqrt(self.feature_size)))
        drins = tf.matmul(instance, dr_weights)  # batch,frame_num,cluster_size
        drins = slim.batch_norm(
            drins,
            center=True,
            scale=True,
            is_training=is_training,
            scope="drins_bn")
        attention_weights = tf.get_variable("attention_weights",
                                            [self.cluster_size, self.cluster_size / 2],
                                            initializer=tf.random_normal_initializer(
                                                stddev=1 / math.sqrt(self.feature_size)))
        attention_gate = tf.get_variable("attention_gate",
                                         [self.cluster_size, self.cluster_size / 2],
                                         initializer=tf.random_normal_initializer(
                                             stddev=1 / math.sqrt(self.feature_size)))
        selection_weights = tf.get_variable("selection_weights",
                                            [self.cluster_size / 2, 1],
                                            initializer=tf.random_normal_initializer(
                                                stddev=1 / math.sqrt(self.cluster_size / 2)))
        att_gate = tf.sigmoid(tf.matmul(drins, attention_gate))  # batch,frame_num,cluster/2
        attention = tf.multiply(tf.tanh(tf.matmul(drins, attention_weights)), att_gate)
        selection = tf.matmul(attention, selection_weights)
        selection = tf.nn.softmax(tf.reshape(selection, [-1, self.frame_nums]), axis=1)
        selection = tf.reshape(selection, [-1, self.frame_nums, 1])
        instance = tf.reshape(instance, [-1, self.frame_nums, self.feature_size])
        instt = tf.transpose(instance, perm=[0, 2, 1])
        instance_att = tf.squeeze(tf.matmul(instt, selection), axis=2)
        return instance_att


class ModelUtil():
    def _exponential_decay_with_warmup(self, warmup_step,
                                       learning_rate_base,
                                       global_step,
                                       learning_rate_step,
                                       learning_rate_decay,
                                       staircase=False):
        """
        Exponential decay learning rate with warmup
        :param warmup_step:         #
        :param learning_rate_base:  #
        :param global_step:         #
        :param learning_rate_step:  #
        :param learning_rate_decay: #
        :param staircase:           #False,ExponentialDecay
        :return:
        """
        with tf.name_scope("exponential_decay_with_warmup"):
            # linear_increase = learning_rate_base * tf.cast(global_step / warmup_step, tf.float32)
            exponential_decay = tf.train.exponential_decay(learning_rate_base,
                                                           global_step - warmup_step,
                                                           learning_rate_step,
                                                           learning_rate_decay,
                                                           staircase=staircase)
            # learning_rate = tf.cond(global_step <= warmup_step,
            #                         lambda: linear_increase,
            #                         lambda: exponential_decay)
            return exponential_decay

    def _get_word_id_dict(self, in_file):
        """
        mapping dict: token-->id
        :param in_file:   token_text \t token_id
        :return:
        """
        word_id_dict = {}
        with open(in_file, 'r')as fr:
            for line in fr:
                line_split = line.rstrip('\n').split('\t')
                if len(line_split) != 2:
                    continue
                token_text = line_split[0]
                token_id = int(line_split[1])
                if token_text not in word_id_dict:
                    word_id_dict[token_text] = token_id
        return word_id_dict

    def _init_vocab_and_emb(self, word_vocab, pre_train_emb_path):
        """
        :param word_vocab:   token_text \t token_id
        :param pre_train_emb_path:  the path of pretrain word embedding
        :return:
        """

        word_id_dict = self._get_word_id_dict(word_vocab)
        word_embeddings = self._load_pretrained_embedding(word_id_dict=word_id_dict,
                                                          pre_train_emb_path=pre_train_emb_path)
        return word_embeddings

    def _load_pretrained_embedding(self, word_id_dict, pre_train_emb_path, embedding_dim=200):
        """
        Load the pre-trained emb AI Lab open source, 200 for each dimension
        https://ai.tencent.com/ailab/nlp/en/embedding.html
        :param word_id_dict:        #
        :param pre_train_emb_path: #
        :param embedding_dim:    #the fea size of emb
        :return:
        """
        trained_embedding = {}
        word_emb = 0
        with open(pre_train_emb_path, 'r')as fr:
            num = 0
            for line in fr:
                num += 1
                if num == 1:
                    continue
                contents = line.rstrip('\n').split()
                if len(contents) != embedding_dim + 1:
                    continue
                trained_embedding[contents[0]] = list(map(float, contents[1:]))
        word_embeddings = np.random.standard_normal([len(word_id_dict), embedding_dim])

        for token, token_id in word_id_dict.items():
            if token in trained_embedding:
                word_emb += 1
                word_embeddings[token_id] = trained_embedding[token]
        word_embeddings[0] = [0.0] * embedding_dim  # padId   0
        word_embeddings[1] = [0.1] * embedding_dim  # unkId   1
        word_embeddings = word_embeddings.astype(np.float32)
        return word_embeddings

    def se_module(self, is_training, activation, gating_reduction=8, name_scope=''):
        """
        SE Gate --> Further select the fusion between features
        :param is_training:       #train: True ; test|valid: False
        :param activation:        #
        :param gating_reduction:  #
        :param name_scope:        #
        :return:
        """
        with tf.variable_scope(name_scope):
            hidden1_size = activation.get_shape().as_list()[1]
            gating_weights_1 = tf.get_variable("gating_weights_1",
                                               [hidden1_size, hidden1_size // gating_reduction],
                                               initializer=slim.variance_scaling_initializer())
            gates = tf.matmul(activation, gating_weights_1)
            gates = slim.batch_norm(
                gates,
                center=True,
                scale=True,
                is_training=is_training,
                activation_fn=slim.nn.relu,
                scope="gating_bn")
            gating_weights_2 = tf.get_variable("gating_weights_2",
                                               [hidden1_size // gating_reduction, hidden1_size],
                                               initializer=slim.variance_scaling_initializer())
            gates = tf.matmul(gates, gating_weights_2)
            gates = tf.sigmoid(gates)
            activation = tf.multiply(activation, gates)
            return activation

    def scale_l2(self, x, norm_length):
        """
        The gradient is normalized according to l2 method
        :param x:            #gradient
        :param norm_length:  #the radio of perturbation
        :return:
        """
        alpha = tf.reduce_max(tf.abs(x), (1, 2), keep_dims=True) + 1e-12
        l2_norm = alpha * tf.sqrt(
            tf.reduce_sum(tf.pow(x / alpha, 2), (1, 2), keep_dims=True) + 1e-6)
        x_unit = x / l2_norm
        return norm_length * x_unit

    def add_perturbation(self, embedded, loss, norm_length=5):
        """
        add adversarial perturbation
        :param embedded:     # the target for adding adversarial perturbation
        :param loss:         # loss
        :param norm_length:  # radio of perturbation
        :return:
        """
        grad, = tf.gradients(
            loss,
            embedded,
            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
        grad = tf.stop_gradient(grad)
        perturb = self.scale_l2(grad, norm_length=norm_length)  # epsilon-->norm_length
        return embedded + perturb

    def tag_hmc_layer(self, fea_vector, dropout_keep_prob, tag_nums,
                      ml_tag, name_scope, radio=0.5, hidden_size=1024):
        """
        hmc：optimize of "Hierarchical Multi-label Classification"
        :param fea_vector:            #dense fes
        :param dropout_keep_prob:     #keep dropout
        :param tag_nums:              #tag nums
        :param ml_tag:                #multi-label
        :param name_scope:            #
        :param radio:                 # the radio of global and  local
        :param hidden_size:           #
        :return:
        """

        with tf.variable_scope(name_scope):
            concate_features_se_drop = fea_vector
            hmc_ag_1 = slim.fully_connected(concate_features_se_drop,
                                            num_outputs=hidden_size,
                                            scope="hmc_WG_1")
            hmc_ag_1_drop = tf.nn.dropout(hmc_ag_1, dropout_keep_prob)
            hmc_ag_1_drop_x = tf.concat([hmc_ag_1_drop, concate_features_se_drop], -1)
            hmc_ag_2 = slim.fully_connected(hmc_ag_1_drop_x,
                                            num_outputs=hidden_size,
                                            scope="hmc_WG_2")
            hmc_ag_2_drop = tf.nn.dropout(hmc_ag_2, dropout_keep_prob)
            hmc_ag_2_drop_x = tf.concat([hmc_ag_2_drop, hmc_ag_1_drop_x], -1)

            hmc_ag_3 = slim.fully_connected(hmc_ag_2_drop_x,
                                            num_outputs=hidden_size // 2,
                                            scope="hmc_WG_3")
            hmc_ag_3_drop = tf.nn.dropout(hmc_ag_3, dropout_keep_prob)
            hmc_ag_3_drop_x = tf.concat([hmc_ag_3_drop, hmc_ag_2_drop_x], -1)
            hmc_pg_scores = slim.fully_connected(hmc_ag_3_drop_x,
                                                 num_outputs=tag_nums,
                                                 activation_fn=None,
                                                 normalizer_fn=None,
                                                 biases_initializer=tf.constant_initializer(0.0),
                                                 scope="hmc_pg")
            hmc_pg_prob = tf.nn.sigmoid(hmc_pg_scores, name='hmc_pg_sigmoid')
            # local tag
            hmc_al_3 = slim.fully_connected(hmc_ag_3_drop,
                                            num_outputs=hidden_size // 2,
                                            scope="hmc_WT_3")
            hmc_al_3_drop = tf.nn.dropout(hmc_al_3, dropout_keep_prob)
            hmc_pl_3_scores = slim.fully_connected(hmc_al_3_drop,
                                                   num_outputs=tag_nums,
                                                   activation_fn=None,
                                                   normalizer_fn=None,
                                                   biases_initializer=tf.constant_initializer(0.0),
                                                   scope="hmc_pl_3")
            hmc_pl_3_prob = tf.nn.sigmoid(hmc_pl_3_scores, name='hmc_pl_3_sigmoid')
            hmc_pf_prob = radio * hmc_pl_3_prob + (1 - radio) * hmc_pg_prob  # [:, :tag_nums]
            # local loss
            pl_3_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=hmc_pl_3_scores,
                                                        labels=ml_tag,
                                                        name="hmc_pl_3_loss"))
            # global loss
            pg_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=hmc_pg_scores,
                                                        labels=ml_tag,
                                                        name="hmc_pg_loss"))
            # local loss + global loss
            return pl_3_loss + pg_loss, hmc_pf_prob

    def cate_hmc_layer(self, fea_vector, dropout_keep_prob, cate_nums,
                       ml_tag, name_scope, radio=0.5, hidden_size=1024):
        """
        hmc：Optimization and improvement base on "Hierarchical Multi-label Classification"
        :param fea_vector:            #dense fea
        :param dropout_keep_prob:     #
        :param tag_nums:              #the cate nums
        :param ml_tag:                #multi-label
        :param name_scope:            #
        :param radio:                 # the radio of global and local
        :param hidden_size:           #
        :return:
        """

        with tf.variable_scope(name_scope):
            concate_features_se_drop = fea_vector
            hmc_ag_1 = slim.fully_connected(concate_features_se_drop,
                                            num_outputs=hidden_size,
                                            scope="hmc_WG_1")
            hmc_ag_1_drop = tf.nn.dropout(hmc_ag_1, dropout_keep_prob)
            hmc_ag_1_drop_x = tf.concat([hmc_ag_1_drop, concate_features_se_drop], -1)
            hmc_ag_2 = slim.fully_connected(hmc_ag_1_drop_x,
                                            num_outputs=hidden_size,
                                            scope="hmc_WG_2")
            hmc_ag_2_drop = tf.nn.dropout(hmc_ag_2, dropout_keep_prob)
            hmc_ag_2_drop_x = tf.concat([hmc_ag_2_drop, hmc_ag_1_drop_x], -1)

            hmc_ag_3 = slim.fully_connected(hmc_ag_2_drop_x,
                                            num_outputs=hidden_size // 2,
                                            scope="hmc_WG_3")
            hmc_ag_3_drop = tf.nn.dropout(hmc_ag_3, dropout_keep_prob)
            hmc_ag_3_drop_x = tf.concat([hmc_ag_3_drop, hmc_ag_2_drop_x], -1)
            hmc_pg_scores = slim.fully_connected(hmc_ag_3_drop_x,
                                                 num_outputs=cate_nums,
                                                 activation_fn=None,
                                                 normalizer_fn=None,
                                                 biases_initializer=tf.constant_initializer(0.0),
                                                 scope="hmc_pg")
            hmc_pg_prob = tf.nn.softmax(hmc_pg_scores, name='hmc_pg_sigmoid')
            # local tag
            hmc_al_3 = slim.fully_connected(hmc_ag_3_drop,
                                            num_outputs=hidden_size // 2,
                                            scope="hmc_WT_3")
            hmc_al_3_drop = tf.nn.dropout(hmc_al_3, dropout_keep_prob)
            hmc_pl_3_scores = slim.fully_connected(hmc_al_3_drop,
                                                   num_outputs=cate_nums,
                                                   activation_fn=None,
                                                   normalizer_fn=None,
                                                   biases_initializer=tf.constant_initializer(0.0),
                                                   scope="hmc_pl_3")
            hmc_pl_3_prob = tf.nn.softmax(hmc_pl_3_scores, name='hmc_pl_3_sigmoid')
            hmc_pf_prob = radio * hmc_pl_3_prob + (1 - radio) * hmc_pg_prob  # [:, :tag_nums]
            # local loss
            pl_3_loss = tf.reduce_mean(
                tf.nn.softmax(logits=hmc_pl_3_scores,
                              labels=ml_tag,
                              name="hmc_pl_3_loss"))
            # global loss
            pg_loss = tf.reduce_mean(
                tf.nn.softmax(logits=hmc_pg_scores,
                              labels=ml_tag,
                              name="hmc_pg_loss"))
            # local loss + global loss
            return pl_3_loss + pg_loss, hmc_pf_prob
