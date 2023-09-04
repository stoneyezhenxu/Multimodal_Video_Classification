#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : train.py
# @Author: stoneye
# @Date  : 2023/09/01
# @Contact : stoneyezhenxu@gmail.com

import tensorflow as tf
import tensorflow.contrib.slim as slim
import utils
from models import ModelUtil
from models import NextvladModel
from models import TextExactor
from models import TrnModel
from tensorflow.python.client import device_lib


class VideoModel():

    def __init__(self, args):
        """
        :param args: config params
        """

        # init the config
        self.init_config()

        model_util_obj = ModelUtil()
        if self.model_type == 'nextvlad':
            model_obj = NextvladModel()  # nextvlad model
        else:
            model_obj = TrnModel()  # trn model
        title_obj = TextExactor()  #  text model
        # 加载预先训练的embeddding
        word_embeddings = model_util_obj._init_vocab_and_emb(word_vocab=args.word_vocab,
                                                            pre_train_emb_path=args.pre_Train_path)
        self.word_embeddings = tf.Variable(word_embeddings,
                                           name='word_embeddings',
                                           dtype=tf.float32)
        self.init_placeholder()
        # build_graph,support sigle gpu or multi gpus
        self.total_loss, self.tag_total_prob, self.cate_total_prob, self.train_op = \
            self.multi_gpu_bulid_graph(num_gpu=self.num_gpu,
                                       lr=self.lr,
                                       ad_strength=self.ad_strength,
                                       word_embeddings=self.word_embeddings,
                                       tag_gt_label=self.tag_gt_label,
                                       cate_gt_label=self.cate_gt_label,
                                       tag_nums=self.tag_nums,
                                       cate_nums=self.cate_nums,
                                       rgb_fea_input=self.input_video_rgb_feature,
                                       rgb_fea_true_frame=self.rgb_fea_true_frame,
                                       audio_fea_input=self.input_video_audio_feature,
                                       audio_fea_true_frame=self.audio_fea_true_frame,
                                       max_frames_rgb=self.max_frames_rgb,
                                       max_frames_audio=self.max_frames_audio,
                                       title_fea_input=self.title_id_int_list,
                                       word_sequence_length=self.word_sequence_length,
                                       model_obj=model_obj,
                                       title_obj=title_obj,
                                       is_training=self.is_training,
                                       dropout_keep_prob=self.dropout_keep_prob,
                                       model_util_obj=model_util_obj,
                                       task_type=self.task_type)


    def init_config(self):
        # task name:["cate","tag","cate_and_tag"]
        # 1)"cate": only cate task; 2)"tag":only tag task;  3)"cate_and_tag": multi-task, cate and tag
        self.task_type = args.task_type
        # nums of gpu
        self.num_gpu = args.num_gpu
        # learning rate
        self.lr = args.lr
        # ratio of adversarial perturbations
        self.ad_strength = args.ad_strength
        # the num of tag
        self.tag_nums = args.tag_nums
        # the num of cate
        self.cate_nums = args.cate_nums
        # the num of video frames
        self.max_frames_rgb = args.rgb_frames
        # the num of audio frames
        self.max_frames_audio = args.audio_frames
        # the max length word(word id) of title
        self.title_max_len = args.title_max_len
        # main aggregate model : light-weight: trn ; heavy-weight: nextvlad
        self.model_type = args.model_type
        # the feature size of img frames.
        self.rgb_fea_size = args.rgb_fea_size
        # the feature size of audio frames.
        self.audio_fea_size = args.audio_fea_size

    def init_placeholder(self):
        """
        :return:
        """

        # title:[batch,max_len]
        self.title_id_int_list = tf.placeholder(tf.int32,
                                                shape=[None, self.title_max_len])
        word_sequence_length = tf.reduce_sum(tf.sign(
            self.title_id_int_list), axis=1)  # [batch,]
        self.word_sequence_length = tf.cast(word_sequence_length, tf.int32)  # [batch,]

        # cate ground truth
        self.cate_gt_label = tf.placeholder(tf.float32,
                                            shape=[None, self.tag_nums],
                                            name="cate_gt_label")

        # tag ground truth
        self.tag_gt_label = tf.placeholder(tf.float32,
                                           shape=[None, self.tag_nums],
                                           name="tag_gt_label")
        # rgb fea
        self.input_video_rgb_feature = tf.placeholder(tf.float32,
                                                      shape=[None, self.max_frames_rgb,
                                                             self.rgb_fea_size])
        # the num of rgb frames
        self.rgb_fea_true_frame = tf.placeholder(tf.int32,
                                                 shape=[None, ])
        # the num of audio frames
        self.input_video_audio_feature = tf.placeholder(tf.float32,
                                                        shape=[None,
                                                               self.max_frames_audio,
                                                               self.audio_fea_size])
        # audio frames
        self.audio_fea_true_frame = tf.placeholder(tf.int32,
                                                   shape=[None, ])
        # keep dropout
        self.dropout_keep_prob = tf.placeholder(tf.float32,
                                                name="dropout_keep_prob")
        # is train stage or not
        self.is_training = tf.placeholder(tf.bool,
                                          name="is_training")

    def cal_loss(self, rgb_fea, rgb_fea_true_frame,
                 audio_fea, audio_fea_true_frame,
                 title_emb_fea, word_sequence_length,
                 tag_gt_label, cate_gt_label,
                 tag_nums, cate_nums,
                 max_frames_rgb, max_frames_audio,
                 is_training, dropout_keep_prob, model_obj,
                 title_obj, model_util_obj, reuse,
                 task_type,
                 hidden_size=256, embedding_size=200,
                 num_filters=100, num_outputs=1024,
                 ):

        with tf.variable_scope("cl_loss_from_emb", reuse=reuse):

            # rgb dense vector
            rgb_cluster_fea = model_obj.forward(is_training=is_training,
                                                fea_input=rgb_fea,
                                                dropout_keep_prob=dropout_keep_prob,
                                                fea_type='rgb',
                                                max_frames=max_frames_rgb,
                                                true_frames=rgb_fea_true_frame,
                                                name_scope='rgb_cluster_fea')
            # audio dense vector
            audio_cluster_fea = model_obj.forward(is_training=is_training,
                                                  fea_input=audio_fea,
                                                  dropout_keep_prob=dropout_keep_prob,
                                                  fea_type='audio',
                                                  max_frames=max_frames_audio,
                                                  true_frames=audio_fea_true_frame,
                                                  name_scope='audio_cluster_fea')
            # title dense vector  baesd on bilstm model
            bilstm_title_feature = title_obj._bilstm_feature(
                embedding_descript=title_emb_fea,
                hidden_size=hidden_size,
                des_sequence_length=word_sequence_length,
                dtype=tf.float32,
                reuse=None)
            # title dense vector based on textcnn model
            textcnn_title_feature = title_obj._text_cnn_feature(
                embedding_descript=title_emb_fea,
                embedding_size=embedding_size,
                filter_sizes=list(map(int, "2,3,4,5".split(","))),
                num_filters=num_filters,
                reuse=None
            )

            title_fea = tf.concat([bilstm_title_feature, textcnn_title_feature], axis=1)

            title_fea_drop = slim.dropout(title_fea,
                                          keep_prob=dropout_keep_prob,
                                          is_training=is_training,
                                          scope="title_fea_drop")
            title_fea_dense = slim.fully_connected(inputs=title_fea_drop,
                                                   num_outputs=num_outputs,
                                                   activation_fn=None,
                                                   scope="title_fea_dense")
            # batch normalization
            title_fea_dense_bn = slim.batch_norm(
                title_fea_dense,
                center=True,
                scale=True,
                is_training=is_training,
                scope="title_fea_dense_bn",
                fused=False)

            with slim.arg_scope([slim.fully_connected],
                                normalizer_fn=slim.batch_norm,
                                normalizer_params={'is_training': is_training, 'center': True,
                                                   'scale': True}):
                # multi-modal
                total_fea = tf.concat([rgb_cluster_fea, audio_cluster_fea, title_fea_dense_bn], 1)

                # se gate
                concate_features_se = model_util_obj._se_module(is_training=is_training,
                                                               activation=total_fea,
                                                               name_scope="concat_se")
                concate_features_se_drop = tf.nn.dropout(concate_features_se, dropout_keep_prob)



                if task_type == 'cate':
                    cate_total_loss, cate_total_prob = ModelUtil.cate_hmc_layer(
                        fea_vector=concate_features_se_drop,
                        dropout_keep_prob=dropout_keep_prob,
                        cate_nums=cate_nums,
                        ml_tag=cate_gt_label,
                        name_scope='cate1_total_loss')
                    tag_total_prob = tf.zeros_like(tag_gt_label)
                    return cate_total_loss, tag_total_prob, cate_total_prob

                elif task_type == 'tag':
                    tag_total_loss, \
                    tag_total_prob = ModelUtil.tag_hmc_layer(fea_vector=concate_features_se_drop,
                                                             dropout_keep_prob=dropout_keep_prob,
                                                             tag_nums=tag_nums,
                                                             ml_tag=tag_gt_label,
                                                             name_scope='tag_total_loss')

                    cate_total_prob = tf.zeros_like(cate_gt_label)

                    return tag_total_loss, tag_total_prob, cate_total_prob

                elif task_type == 'cate_and_tag':
                    cate_total_loss, cate_total_prob = ModelUtil.cate_hmc_layer(
                        fea_vector=concate_features_se_drop,
                        dropout_keep_prob=dropout_keep_prob,
                        cate_nums=cate_nums,
                        ml_tag=cate_gt_label,
                        name_scope='cate1_total_loss')

                    tag_total_loss, \
                    tag_total_prob = ModelUtil.tag_hmc_layer(fea_vector=concate_features_se_drop,
                                                             dropout_keep_prob=dropout_keep_prob,
                                                             tag_nums=tag_nums,
                                                             ml_tag=tag_gt_label,
                                                             name_scope='tag_total_loss')
                    return cate_total_loss + tag_total_loss, tag_total_prob, cate_total_prob
                else:
                    raise Exception('task_type:{} not in [cate,tag,cate_and_tag]')

    def multi_gpu_bulid_graph(self, num_gpu, lr, ad_strength, word_embeddings,
                              tag_gt_label, cate_gt_label, tag_nums, cate_nums,
                              title_fea_input, word_sequence_length,
                              rgb_fea_input, rgb_fea_true_frame,
                              audio_fea_input, audio_fea_true_frame,
                              max_frames_rgb, max_frames_audio,
                              model_obj, title_obj, is_training,
                              dropout_keep_prob, model_util_obj, task_type):
        """

        :param num_gpu:         # the nums of gpu
        :param lr:              #learning rate
        :param ad_strength:     # adversarial perturbation
        :param word_embeddings: #word embedding [batch,emb_size]
        :param tag_gt_label:          # tag gt label [batch,tag_nums]
        :param cate_gt_label:          #cate gt label [batch,cate_nums]
        :param tag_nums:        # the nums of tag
        :param cate_nums:        # the nums of cate
        :param title_fea_input:  # title fea [batch,seq_len]
        :param word_sequence_length:  # the truth length of title
        :param rgb_fea_input:         #rgb fea [batch,frame,fea_size]
        :param rgb_fea_true_frame:    #the truth frames of rgb fea
        :param audio_fea_input:     #audio fea [batch,frame,fea_size]
        :param audio_fea_true_frame: #the truth frames of audio fea
        :param max_frames_rgb:      #the max frames of rgb
        :param max_frames_audio:  #the max frames of audio
        :param model_obj:        #aggregate model: nextvlad or trn
        :param title_obj:       #textcnn or Bi-LSTM
        :param is_training:     # True or False
        :param dropout_keep_prob:  # float
        :param model_util_obj:     #
        :param task_type: #the type of task：cate, tag, cate_and_tag：multi-task,cate & tag
        :return:
        """

        local_device_protos = device_lib.list_local_devices()
        gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
        gpus = gpus[:num_gpu]
        num_gpus = len(gpus)

        if num_gpus > 0:
            print("Using the following GPUs to train: {}".format(gpus))
            num_towers = num_gpus
            device_string = '/gpu:%d'
        else:
            print("No GPUs found. Training on CPU.")
            num_towers = 1
            device_string = '/cpu:%d'

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        optimizer = tf.train.AdamOptimizer(lr)

        tower_rgb_fea_input = tf.split(rgb_fea_input, num_towers)
        tower_rgb_fea_true_frame = tf.split(rgb_fea_true_frame, num_towers)
        tower_audio_fea_input = tf.split(audio_fea_input, num_towers)
        tower_audio_fea_true_frame = tf.split(audio_fea_true_frame, num_towers)
        tower_title_fea_input = tf.split(title_fea_input, num_towers)

        tower_word_sequence_length = tf.split(word_sequence_length, num_towers)
        tower_tag_gt_label = tf.split(tag_gt_label, num_towers)
        tower_cate_gt_label = tf.split(cate_gt_label, num_towers)

        tower_gradients = []
        tower_predict_tag_probs = []
        tower_predict_cate_probs = []
        tower_total_losses = []

        for i in range(num_towers):
            with tf.device(device_string % i):
                with (tf.variable_scope(("tower"), reuse=True if i > 0 else None)):
                    with (slim.arg_scope([slim.model_variable, slim.variable],
                                         device="/cpu:0" if num_gpu != 1 else "/gpu:0")):
                        result = self.build_graph(
                            word_embeddings=word_embeddings,
                            rgb_fea_input=tower_rgb_fea_input[i],
                            rgb_fea_true_frame=tower_rgb_fea_true_frame[i],
                            audio_fea_input=tower_audio_fea_input[i],
                            audio_fea_true_frame=tower_audio_fea_true_frame[i],
                            max_frames_rgb=max_frames_rgb,
                            max_frames_audio=max_frames_audio,
                            title_fea_input=tower_title_fea_input[i],
                            word_sequence_length=tower_word_sequence_length[i],
                            tag_gt_label=tower_tag_gt_label[i],
                            cate_gt_label=tower_cate_gt_label[i],
                            is_training=is_training,
                            ad_strength=ad_strength,
                            tag_nums=tag_nums,
                            cate_nums=cate_nums,
                            model_obj=model_obj,
                            title_obj=title_obj,
                            dropout_keep_prob=dropout_keep_prob,
                            model_util_obj=model_util_obj,
                            task_type=task_type
                        )

                        cl_tag_prob = result["tag_prob"]
                        tower_predict_tag_probs.append(cl_tag_prob)
                        cl_cate_prob = result["cate_prob"]
                        tower_predict_cate_probs.append(cl_cate_prob)

                        loss = result["loss"]

                        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                        with tf.control_dependencies(update_ops):
                            tower_total_losses.append(loss)
                            gradients = \
                                optimizer.compute_gradients(loss, colocate_gradients_with_ops=False)

                            tower_gradients.append(gradients)

        total_loss = tf.reduce_mean(tf.stack(tower_total_losses))
        total_tag_prob = tf.concat(tower_predict_tag_probs, 0)
        total_cate_prob = tf.concat(tower_predict_cate_probs, 0)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            merged_gradients = utils.combine_gradients(tower_gradients)
            train_op = optimizer.apply_gradients(merged_gradients, global_step=self.global_step)

        return total_loss, total_tag_prob, total_cate_prob, train_op

    def build_graph(self, word_embeddings, tag_gt_label,
                    cate_gt_label, tag_nums, cate_nums, ad_strength,
                    rgb_fea_input, rgb_fea_true_frame,
                    audio_fea_input, audio_fea_true_frame,
                    title_fea_input, word_sequence_length,
                    max_frames_rgb, max_frames_audio,
                    is_training, model_obj, title_obj,
                    dropout_keep_prob, model_util_obj, task_type):
        # [batch,25,emb_size]
        embedded_title = tf.nn.embedding_lookup(word_embeddings,
                                                title_fea_input)

        # sigmoid cross entropy loss
        cl_loss, cl_tag_prob, cl_cate_prob = self.cal_loss(rgb_fea=rgb_fea_input,
                                                           rgb_fea_true_frame=rgb_fea_true_frame,
                                                           max_frames_rgb=max_frames_rgb,
                                                           max_frames_audio=max_frames_audio,
                                                           audio_fea=audio_fea_input,
                                                           audio_fea_true_frame=audio_fea_true_frame,
                                                           title_emb_fea=embedded_title,
                                                           word_sequence_length=word_sequence_length,
                                                           is_training=is_training,
                                                           tag_gt_label=tag_gt_label,
                                                           cate_gt_label=cate_gt_label,
                                                           tag_nums=tag_nums,
                                                           cate_nums=cate_nums,
                                                           dropout_keep_prob=dropout_keep_prob,
                                                           model_obj=model_obj,
                                                           title_obj=title_obj,
                                                           model_util_obj=model_util_obj,
                                                           task_type=task_type,
                                                           reuse=None)

        # add the perturbation on rgb fea
        rgb_fea_perturbated = model_util_obj.add_perturbation(rgb_fea_input, cl_loss,
                                                             norm_length=ad_strength)

        # add the perturbation on audio fea
        audio_fea_perturbated = model_util_obj.add_perturbation(audio_fea_input, cl_loss,
                                                               norm_length=ad_strength)
        # add the perturbation on text(title) fea
        title_emb_fea_perturbated = model_util_obj.add_perturbation(embedded_title, cl_loss,
                                                                   norm_length=ad_strength)

        #  sigmoid cross entropy loss of perturbation
        ad_loss, _, _ = self.cal_loss(rgb_fea=rgb_fea_perturbated,
                                      rgb_fea_true_frame=rgb_fea_true_frame,
                                      max_frames_rgb=max_frames_rgb,
                                      max_frames_audio=max_frames_audio,
                                      audio_fea=audio_fea_perturbated,
                                      audio_fea_true_frame=audio_fea_true_frame,
                                      title_emb_fea=title_emb_fea_perturbated,
                                      word_sequence_length=word_sequence_length,
                                      is_training=is_training,
                                      tag_gt_label=tag_gt_label,
                                      cate_gt_label=cate_gt_label,
                                      tag_nums=tag_nums,
                                      cate_nums=cate_nums,
                                      dropout_keep_prob=dropout_keep_prob,
                                      model_obj=model_obj,
                                      title_obj=title_obj,
                                      model_util_obj=model_util_obj,
                                      task_type=task_type,
                                      reuse=True)


        return {'loss': cl_loss + ad_loss, 'tag_prob': cl_tag_prob, 'cate_prob': cl_cate_prob}
