#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : train.py
# @Author: stoneye
# @Date  : 2023/09/01
# @Contact : stoneyezhenxu@gmail.com


import argparse
import os
import sys
import time

import data
import numpy as np
import tensorflow as tf
from eval_metrics import cal_eval_metrics
from video_model import VideoModel

parser = argparse.ArgumentParser()
# the type of model: "trn" or "nextvlad"
parser.add_argument("--model_type",
                    help="the model for [trn or nextvlad]",
                    default="nextvlad",
                    type=str)
# the type of task: "cate" or "tag" or "multi-task"
parser.add_argument("--task_type",
                    help=" task_type for [cate or tag or cate_and_tag]",
                    default="cate",
                    type=str)
# the type of padding: "segment" or "max_padding"
parser.add_argument("--segment_or_max_padding",
                    help="[segment or max_padding ]",
                    default="segment",
                    type=str)
# the type of mode : "train" or "predict"
parser.add_argument("--mode_type",
                    help="train | predict]",
                    default="train",
                    type=str)
# the num of gpu
parser.add_argument("--num_gpu",
                    help="the nums of gpu ",
                    default=2,
                    type=int)
# the fea size of rgb
parser.add_argument("--rgb_fea_size",
                    help="rgb_fea_size number",
                    default=1536,
                    type=int)

# the fea size of audio
parser.add_argument("--audio_fea_size",
                    help="audio_fea_size number",
                    default=128,
                    type=int)

# the num of cate
parser.add_argument("--cate_nums",
                    help="class number",
                    default=201,
                    type=int)
# the num of tag
parser.add_argument("-tag_nums", "--tag_nums",
                    help="class number",
                    default=20000,
                    type=int)
# the max length of title
parser.add_argument("--title_max_len",
                    help="title_max_len",
                    default=30,
                    type=int)
# the radio of adversarial perturbation
parser.add_argument("--ad_strength",
                    help="ad_strength",
                    default=0.5,
                    type=float)
# the max length of rgb frame
parser.add_argument("--rgb_frames",
                    help="the size of batch ",
                    default=16,
                    type=int)
# the max length of audio frame
parser.add_argument("--audio_frames",
                    help="the size of batch ",
                    default=64,
                    type=int)
# batch_size
parser.add_argument("--batch_size",
                    help="the size of batch ",
                    default=4,
                    type=int
                    )
# learning rate
parser.add_argument("--lr",
                    help="learning rate",
                    default=1e-3,
                    type=float)
# the radio of keep dropout
parser.add_argument("--keep_dropout",
                    help="dropout rate",
                    default=0.8,
                    type=float)
# the path of word vocab dict
parser.add_argument("--word_vocab",
                    help="the data path of word vocab",
                    default="../vocab/tencent_ailab_50w_200_emb_vocab.txt",
                    type=str)
# the path of trainset : train*_.tfrecord
parser.add_argument("--train_path",
                    help="the data path of train_rgb",
                    default="../vocab",
                    type=str)
# the path of validset : valid*_.tfrecord
parser.add_argument("--valid_path",
                    help="the data path of valid_rgb",
                    default="../vocab",
                    type=str)
# the path of testset : test*_.tfrecord
parser.add_argument("--test_path",
                    help="the data path of test_rgb",
                    default="../vocab",
                    type=str)
# AiLab Tencent open sources word embedding
parser.add_argument("--pre_trained_emb_path",
                    help="the data path of pre_trained_emb_path",
                    default="../vocab/head10.emb",
                    type=str)
# the frequence of report
parser.add_argument("--report_freq",
                    help="frequency to report loss",
                    default=1,
                    type=int)
# how many times of reporting per epoch
parser.add_argument("--one_epoch_step_eval_num",
                    help="frequency to do validation",
                    default=4,
                    type=int)
# epoch
parser.add_argument("--epoch",
                    help="the number of epoch",
                    default=10,
                    type=int)
# the max size of model to saved
parser.add_argument("--max_to_keep",
                    help="the max model save",
                    default=3,
                    type=int)
# early_stop condition
parser.add_argument("--early_stop_num",
                    help="the num of early_stop ",
                    default=30,
                    type=int)
# threshold for logit
parser.add_argument("--threshold",
                    help="the threshold for output",
                    default=0.5,
                    type=float)
# the path of save model
parser.add_argument("--model_dir",
                    help="the dir of save model",
                    default="./models",
                    type=str)
# the path of predict cate
parser.add_argument("--predict_cate_output",
                    help="the data path of cate predict out path",
                    default="./predict_testset_cate_infos.txt",
                    type=str)
# the path of predict tag
parser.add_argument("--predict_tag_output",
                    help="the data path of predict_out",
                    default="./predict_testset_tag_infos.txt",
                    type=str)

args = parser.parse_args()


def creat_model(session, params):
    model = VideoModel(params)
    ckpt = tf.train.get_checkpoint_state(params.model_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from {}".format(ckpt.model_checkpoint_path))
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess=session,
                      save_path=ckpt.model_checkpoint_path)
    else:
        if not os.path.exists(params.model_dir):
            os.makedirs(params.model_dir)
        print("Created new model parameters..")
        session.run(tf.global_variables_initializer())
    return model


def train():
    valid_cate_max_f1, valid_tag_max_f1 = -9999, -9999
    early_stop_num = 0
    #Use the gpu first, otherwise use cpu resources.
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        #build trainset iterator
        train_iter = data.Itertool(sess=sess,
                                   data_path=args.train_path,
                                   batch_size=args.batch_size,
                                   title_max_len=args.title_max_len,
                                   cate_num=args.cate_nums,
                                   tag_num=args.tag_nums,
                                   segment_or_max_padding=args.segment_or_max_padding,
                                   rgb_frame=args.rgb_frames,
                                   audio_frame=args.audio_frames,
                                   mode='train')

        # build validset iterator
        valid_iter = data.Itertool(sess=sess,
                                   data_path=args.valid_path,
                                   batch_size=args.batch_size,
                                   title_max_len=args.title_max_len,
                                   cate_num=args.cate_nums,
                                   tag_num=args.tag_nums,
                                   segment_or_max_padding=args.segment_or_max_padding,
                                   rgb_frame=args.rgb_frames,
                                   audio_frame=args.audio_frames,
                                   mode='test')


        total_total_dataset_num_num = train_iter.total_dataset_num
        one_epoch_step = int(total_total_dataset_num_num / args.batch_size)
        model = creat_model(session=sess, params=args)
        saver = tf.train.Saver(tf.global_variables(),
                               max_to_keep=args.max_to_keep)
        print("Begin training..")
        for epoch in range(args.epoch):
            old = time.time()
            if early_stop_num >= args.early_stop_num:
                print("early_stop !")
                break
            total_loss_list = []
            tag_train_total_p_list, tag_train_total_r_list, tag_train_total_f1_list = [], [], []
            cate_train_total_p_list, cate_train_total_r_list, cate_train_total_f1_list = [], [], []

            # one batch-size data of dataset
            for samples in train_iter.yield_one_batch_data():
                _, padding_title_np, tag_gt_label_np, cate_gt_label_np, padding_rgb_fea_np, \
                rgb_fea_ture_frame_np, padding_audio_fea_np, \
                audio_fea_ture_frame_np = tuple(samples)
                feed = dict(
                    zip([model.title_id_int_list,
                         model.tag_gt_label,
                         model.cate_gt_label,
                         model.input_video_rgb_feature,
                         model.rgb_fea_true_frame,
                         model.input_video_audio_feature,
                         model.audio_fea_true_frame,
                         model.dropout_keep_prob,
                         model.is_training],
                        [padding_title_np,
                         tag_gt_label_np,
                         cate_gt_label_np,
                         padding_rgb_fea_np,
                         rgb_fea_ture_frame_np,
                         padding_audio_fea_np,
                         audio_fea_ture_frame_np,
                         args.keep_dropout,
                         True]))
                # update model params
                tag_probs, cate_probs, total_loss, _ = sess.run(
                    [model.tag_total_prob, model.cate_total_prob,
                     model.total_loss, model.train_op], feed)
                total_loss_list.append(total_loss)

                tag_gl_tag_dense = [np.where(x == 1)[0].tolist()
                                    for x in tag_gt_label_np]
                tag_pred_tag_dense = [np.where(x > args.threshold)[0].tolist()
                                      for x in tag_probs]

                tag_total_p, tag_total_r, \
                tag_total_f1 = cal_eval_metrics(tag_gl_tag_dense,
                                                tag_pred_tag_dense)
                tag_train_total_p_list.append(tag_total_p)
                tag_train_total_r_list.append(tag_total_r)
                tag_train_total_f1_list.append(tag_total_f1)

                cate_gl_dense = [[np.argmax(x)] for x in cate_gt_label_np]
                cate_pred_dense = [[np.argmax(x)] for x in cate_probs]
                cate_total_p, cate_total_r, \
                cate_total_f1 = cal_eval_metrics(cate_gl_dense, cate_pred_dense)

                # cal tag p,r,f
                tag_train_total_p_list.append(tag_total_p)
                tag_train_total_r_list.append(tag_total_r)
                tag_train_total_f1_list.append(tag_total_f1)
                # cal cate p,r,f
                cate_train_total_p_list.append(cate_total_p)
                cate_train_total_r_list.append(cate_total_r)
                cate_train_total_f1_list.append(cate_total_f1)

                sys.stdout.flush()
                if model.global_step.eval() % args.report_freq == 0:
                    print("report_freq: ", args.report_freq)
                    print(
                        'Train-->  Epoch:{}, Step:{} Loss:{:.4f}\n'
                        'cate--> t_P:{:.4f} t_R:{:.4f} t_F1:{:.4f}\n'
                        'tag-->t_P:{:.4f} t_R:{:.4f} t_F1:{:.4f}'.format(
                            epoch, model.global_step.eval(),
                            1.0 * np.sum(total_loss_list) / len(total_loss_list),
                            1.0 * np.sum(cate_train_total_p_list) / len(cate_train_total_p_list),
                            1.0 * np.sum(cate_train_total_r_list) / len(cate_train_total_r_list),
                            1.0 * np.sum(cate_train_total_f1_list) / len(cate_train_total_f1_list),
                            1.0 * np.sum(tag_train_total_p_list) / len(tag_train_total_p_list),
                            1.0 * np.sum(tag_train_total_r_list) / len(tag_train_total_r_list),
                            1.0 * np.sum(tag_train_total_f1_list) / len(tag_train_total_f1_list)))

                    total_loss_list = []
                    tag_train_total_p_list, tag_train_total_r_list, tag_train_total_f1_list = [], [], []
                    cate_train_total_p_list, cate_train_total_r_list, cate_train_total_f1_list = [], [], []

                # eval the metrics on validset
                if model.global_step.eval() % one_epoch_step == 0:

                    cate_total_p, cate_total_r, cate_total_f1, \
                    tag_total_p, tag_total_r, tag_total_f1, \
                    valid_total_loss_list_aver = do_eval(model=model,
                                                         data_obj=valid_iter,
                                                         sess=sess)
                    print('\n***********************')
                    print(
                        'Valid--> Epoch:{}; Step:{}; Valid: loss:{:.4f}\n'
                        'cate--> t_P:{:.4f} t_R:{:.4f} t_F1:{:.4f}\n'
                        'tag-->t_P:{:.4f} t_R:{:.4f} t_F1:{:.4f}'.format(
                            epoch, model.global_step.eval(), valid_total_loss_list_aver,
                            cate_total_p, cate_total_r, cate_total_f1,
                            tag_total_p, tag_total_r, tag_total_f1
                        ))
                    print('***********************\n')

                    # Save the model with the highest accuracy in the current validset
                    if args.task_type == 'cate':
                        if cate_total_f1 > valid_cate_max_f1:
                            valid_cate_max_f1 = cate_total_f1
                            checkpoint_path = os.path.join(args.model_dir, 'model.ckpt')
                            saver.save(sess=sess, save_path=checkpoint_path,
                                       global_step=model.global_step.eval())
                            early_stop_num = 0
                        else:
                            early_stop_num += 1

                    if args.task_type == 'tag':
                        if tag_total_f1 > valid_tag_max_f1:
                            valid_tag_max_f1 = tag_total_f1
                            checkpoint_path = os.path.join(args.model_dir, 'model.ckpt')
                            saver.save(sess=sess,
                                       save_path=checkpoint_path,
                                       global_step=model.global_step.eval())
                            early_stop_num = 0
                        else:
                            early_stop_num += 1

                    if args.task_type == 'cate_and_tag':
                        if tag_total_f1 > valid_tag_max_f1 \
                                and cate_total_f1 > valid_cate_max_f1:
                            valid_tag_max_f1 = tag_total_f1
                            valid_cate_max_f1 = cate_total_f1
                            checkpoint_path = os.path.join(args.model_dir, 'model.ckpt')
                            saver.save(sess=sess, save_path=checkpoint_path, global_step=model.global_step.eval())
                            early_stop_num = 0
                        else:
                            early_stop_num += 1

                    print("valid processing is finished!")

            new = time.time()
            print("{}th epoch run {:.3f} minutes:".format(epoch, (new - old) / 60))


def do_eval(model, data_obj, sess):
    print("Valid infer is process !")
    total_loss_list, tag_gt_list, tag_pt_list, cate_gt_list, cate_pt_list = [], [], [], [], []

    for samples in data_obj.yield_one_batch_data():
        _, padding_title_np, tag_gt_label_np, \
        cate_gt_label_np, padding_rgb_fea_np, \
        rgb_fea_ture_frame_np, padding_audio_fea_np, \
        audio_fea_ture_frame_np = tuple(samples)
        feed = dict(
            zip([model.title_id_int_list,
                 model.tag_gt_label,
                 model.cate_gt_label,
                 model.input_video_rgb_feature,
                 model.rgb_fea_true_frame,
                 model.input_video_audio_feature,
                 model.audio_fea_true_frame,
                 model.dropout_keep_prob,
                 model.is_training],
                [padding_title_np,
                 tag_gt_label_np,
                 cate_gt_label_np,
                 padding_rgb_fea_np,
                 rgb_fea_ture_frame_np,
                 padding_audio_fea_np,
                 audio_fea_ture_frame_np,
                 1.0, False]))

        tag_probs, cate_probs, total_loss = sess.run(
            [model.tag_total_prob, model.cate_total_prob, model.total_loss], feed)

        total_loss_list.append(total_loss)
        tag_gt_list.extend([np.where(x == 1)[0].tolist() for x in tag_gt_label_np])
        tag_pt_list.extend([np.where(x > args.threshold)[0].tolist() for x in tag_probs])
        cate_gt_list.extend([[np.argmax(x)] for x in cate_gt_label_np])
        cate_pt_list.extend([[np.argmax(x)] for x in cate_probs])

    total_loss_list_aver = 1.0 * np.sum(total_loss_list) / len(total_loss_list)
    tag_total_p, tag_total_r, tag_total_f1 = cal_eval_metrics(tag_gt_list, tag_pt_list)
    cate_total_p, cate_total_r, cate_total_f1 = cal_eval_metrics(cate_gt_list, cate_pt_list)

    return cate_total_p, cate_total_r, cate_total_f1, \
           tag_total_p, tag_total_r, tag_total_f1, total_loss_list_aver


def predict():
    """
    predict stage
    :return:
    """
    print("predict  is process !")
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        #build the testset iterator
        test_iter = data.Itertool(sess=sess,
                                  data_path=args.test_path,
                                  batch_size=args.batch_size,
                                  title_max_len=args.title_max_len,
                                  tag_num=args.tag_nums,
                                  cate_num=args.cate_nums,
                                  segment_or_max_padding=args.segment_or_max_padding,
                                  rgb_frame=args.rgb_frames,
                                  audio_frame=args.audio_frames,
                                  mode='test')

        model = creat_model(session=sess, params=args)

        for samples in test_iter.yield_one_batch_data():
            raw_vid_np, padding_title_np, tag_gt_label_np, \
            cate_gt_label_np, padding_rgb_fea_np, \
            rgb_fea_ture_frame_np, padding_audio_fea_np, \
            audio_fea_ture_frame_np = tuple(samples)
            feed = dict(
                zip([model.title_id_int_list,
                     model.tag_gt_label,
                     model.cate_gt_label,
                     model.input_video_rgb_feature,
                     model.rgb_fea_true_frame,
                     model.input_video_audio_feature,
                     model.audio_fea_true_frame,
                     model.dropout_keep_prob,
                     model.is_training],
                    [padding_title_np,
                     tag_gt_label_np,
                     cate_gt_label_np,
                     padding_rgb_fea_np,
                     rgb_fea_ture_frame_np,
                     padding_audio_fea_np,
                     audio_fea_ture_frame_np,
                     1.0,
                     False]))



            if args.task_type == 'cate':
                predict_probs = sess.run(
                    [model.cate_total_prob], feed)
                with open(args.predict_cate_output, 'a+')as fw:
                    for vid, prob in zip(raw_vid_np.tolist(),
                                         predict_probs.tolist()):
                        prob = ['{:.4f}'.format(token) for token in prob]
                        fw.write(vid + '\t' + ' '.join(prob) + '\n')

            if args.task_type == 'tag':
                predict_probs = sess.run(
                    [model.tag_total_prob], feed)
                with open(args.predict_tag_output, 'a+')as fw:
                    for vid, prob in zip(raw_vid_np.tolist(),
                                         predict_probs.tolist()):
                        prob = ['{:.4f}'.format(token) for token in prob]
                        fw.write(vid + '\t' + ' '.join(prob) + '\n')

            if args.task_type == 'cate_and_tag':
                predict_cate_probs, predict_tag_probs = sess.run(
                    [model.cate_total_prob, model.tag_total_prob], feed)
                with open(args.predict_cate_output, 'a+')as fw:
                    for vid, prob in zip(raw_vid_np.tolist(), predict_cate_probs.tolist()):
                        prob = ['{:.4f}'.format(token) for token in prob]
                        fw.write(str(vid.decode()) + '\t' + ' '.join(prob) + '\n')

                with open(args.predict_tag_output, 'a+')as fw:
                    for vid, prob in zip(raw_vid_np.tolist(), predict_tag_probs.tolist()):
                        prob = ['{:.4f}'.format(token) for token in prob]
                        fw.write(vid + '\t' + ' '.join(prob) + '\n')


def main():
    if args.mode_type == "train":
        train()  # train stage
    elif args.mode_type == "predict":
        predict()  # predict stage
    else:
        raise Exception('mode_type must be in [train,predict]')


if __name__ == '__main__':
    main()
