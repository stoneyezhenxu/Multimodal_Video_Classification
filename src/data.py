#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : train.py
# @Author: stoneye
# @Date  : 2023/09/01
# @Contact : stoneyezhenxu@gmail.com

import os

import numpy as np
import tensorflow as tf
from numpy.random import randint


class Itertool():
    def __init__(self,
                 sess,
                 data_path,
                 title_max_len,
                 segment_or_max_padding,
                 batch_size,
                 cate_num,
                 tag_num,
                 rgb_frame,
                 audio_frame,
                 mode='test',
                 num_parallel_reads=6,
                 each_tfrecord_samples_num=5000,
                 rgb_fea_size=1536,
                 audio_fea_size=128,
                 buffer_size=2000,
                 prefetch_samples=1000):
        """
        :param sess:  #the instance of session
        :param data_path:  #the dir path of tfrecord
        :param title_max_len: #the max length of title ids .
        :param segment_or_max_padding: #the sample method, either segment or max_padding
        :param batch_size:  #the size of batch
        :param cate_num:  #the num of cate
        :param tag_num:  #the num of tag
        :param rgb_frame: #the max length of rgb-frames fea .
        :param audio_frame: #the max length of audio-frames fea.
        :param mode: #the mode for train or test
        :param num_parallel_reads: #the size of  read the file in parallel.
        :param each_tfrecord_samples_num: #the size of sample of each tfrecord file
        :param rgb_fea_size:  #the size of frame fea，default is 1536
        :param audio_fea_size: #the size of audio fea，default is 128
        :param buffer_size:  #the size of data for shuffle
        :param prefetch_samples:  #the size of data for pre-fetch
        """

        self.sess = sess
        self.data_path = data_path
        self.batch_size = batch_size
        self.segment_or_max_padding = segment_or_max_padding
        self.max_frame_rgb = rgb_frame
        self.max_frame_audio = audio_frame
        self.cate_num = cate_num
        self.tag_num = tag_num
        self.mode = mode
        self.title_max_len = title_max_len
        self.num_parallel_reads = num_parallel_reads
        self.each_tfrecord_samples_num = each_tfrecord_samples_num
        self.rgb_fea_size = rgb_fea_size
        self.audio_fea_size = audio_fea_size
        self.buffer_size = buffer_size
        self.prefetch_samples = prefetch_samples

        if segment_or_max_padding == "segment":
            self.padding_func = self.segment_fea
        else:
            self.padding_func = self.padding_max_frames

        filenames = [os.path.join(data_path, x)
                     for x in os.listdir(data_path)
                     if x.endswith('tfrecord')]
        # total train sample nums: file_nums * 5000
        self.total_dataset_num = len(filenames) * self.each_tfrecord_samples_num
        self.num_parallel_reads = min(self.num_parallel_reads, len(filenames))
        dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=self.num_parallel_reads)
        dataset = dataset.map(self.parse_function,
                              num_parallel_calls=self.num_parallel_reads)
        if self.mode == 'train':
            dataset = dataset.shuffle(buffer_size=self.buffer_size)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(self.batch_size)
        iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
        self.next_element = iterator.get_next()
        self.dataset_init_op = iterator.make_initializer(dataset)

    def padding_max_frames(self, fea_list, max_frame):
        '''
        Unify the features: Unify the audio frames and video frames into a fixed dimension to perform matrix operations
        :param fea_list:  numpy,  [frame,fea_size]
        :param max_frame: int
        :return:
        '''

        feature_size = fea_list.shape[1]
        fea_list = list(fea_list)
        fea_list_len = len(fea_list)
        if fea_list_len > max_frame:
            fea_true_frame = max_frame
            fea_list = fea_list[:max_frame]
        else:
            fea_true_frame = fea_list_len
            fea_list = np.vstack((fea_list, np.zeros((max_frame - len(fea_list), feature_size))))
        return np.asarray(fea_list, dtype=np.float32), np.asarray(fea_true_frame, dtype=np.int32)

    def segment_fea(self, fea_list, segment_num):
        '''
         Sparsely sample the features: Unify the audio and video frames shape to a fixed dimension in order to perform matrix operations
        :param fea_list: numpy,  [frame,fea_size]
        :param segment_num: int
        :return:
        '''
        fea_len = len(fea_list)
        if self.mode == 'train' and fea_len > segment_num:
            average_duration = fea_len // segment_num
            begin_index = np.multiply(list(range(segment_num)), average_duration)
            random_index = randint(average_duration, size=segment_num)
            indexs = list(begin_index + random_index)
            segment_fea_list_x = [fea_list[_index] for _index in indexs]
        else:
            tick = (fea_len) / float(segment_num)
            indexs = np.array([int(tick / 2.0 + tick * x) for x in range(segment_num)])
            segment_fea_list_x = [fea_list[_index] for _index in indexs]
        return np.asarray(segment_fea_list_x), np.asarray(segment_num, dtype=np.int32)

    def padding_title_id_list(self, text_id_list, max_num, pad_id=0):
        '''
        Unify the shape of the feature: unify title to a fixed dimension in order to perform matrix operations
        :param text_id_list:
        :param max_num:
        :return:
        '''
        text_id_list = text_id_list.decode().split(';')
        if len(text_id_list) > 0:
            title_id_int_list = [int(one_id) for one_id in text_id_list if one_id]
        else:
            title_id_int_list = []
        if len(title_id_int_list) > max_num:
            title_id_int_list = title_id_int_list[:max_num]
        else:
            title_id_int_list = np.vstack((title_id_int_list,
                                           np.zeros((max_num - len(title_id_int_list),
                                                     pad_id))))

        return np.asarray(title_id_int_list, dtype=np.int32)

    def parse_cate_gt_label(self, cate_id):
        '''
        :param cate_id:
        :return:
        '''
        cate2_gt_label = [0.0] * self.cate_num
        if self.mode == 'test':
            return cate2_gt_label
        else:
            cate2_gt_label[cate_id] = 1.0
            return cate2_gt_label

    def parse_tag_multi_gt_label(self, tag_list):
        '''

        :param cate_id:
        :return:
        '''
        tag_multi_gt_label = [0.0] * self.tag_num
        for tag_id in tag_list:
            if tag_id >= self.tag_num:
                continue
            tag_multi_gt_label[tag_id] = 1.0
        return tag_multi_gt_label

    def parse_function(self, example_proto):
        '''
        parse the infos from tfrecord file
        :param example_proto:
        :return:
        '''
        dics = {
            'vid': tf.FixedLenFeature(shape=(), dtype=tf.string),
            'rgb_fea': tf.FixedLenFeature(shape=(), dtype=tf.string),
            'audio_fea': tf.FixedLenFeature(shape=(), dtype=tf.string),
            'rgb_shape': tf.FixedLenFeature(shape=(2,), dtype=tf.int64),
            'audio_shape': tf.FixedLenFeature(shape=(2,), dtype=tf.int64),
            'title': tf.FixedLenFeature(shape=(), dtype=tf.string),
            'tag_gt_label': tf.FixedLenFeature(shape=(), dtype=tf.string),
            'cate_gt_label': tf.FixedLenFeature(shape=(), dtype=tf.int64)
        }

        parsed_example = tf.parse_single_example(example_proto, dics)
        # rgb_fea: [frame,rgb_fea]
        raw_rgb_fea = tf.reshape(tf.decode_raw(parsed_example['rgb_fea'], tf.float32),
                                 parsed_example['rgb_shape'])
        # audio_fea: [frame,audio_fea]
        raw_audio_fea = tf.reshape(tf.decode_raw(parsed_example['audio_fea'], tf.float32),
                                   parsed_example['audio_shape'])
        raw_vid = parsed_example['vid']  # video id
        raw_title = parsed_example['title']  # world id split from title

        # rgb_fea
        padding_rgb_fea, rgb_fea_ture_frame = tf.py_func(func=self.padding_func,
                                                         inp=[parsed_example['rgb_shape'],
                                                              raw_rgb_fea,
                                                              self.max_frame_rgb,
                                                              self.mode],
                                                         Tout=[tf.float32, tf.int32])
        # audio_fea
        padding_audio_fea, audio_fea_ture_frame = tf.py_func(func=self.padding_func,
                                                             inp=[parsed_example['rgb_shape'],
                                                                  raw_audio_fea,
                                                                  self.max_frame_audio,
                                                                  self.mode],
                                                             Tout=[tf.float32, tf.int32])
        # title_fes
        padding_title = tf.py_func(func=self.padding_title_id_list,
                                   inp=[raw_title, self.title_max_len],
                                   Tout=tf.int32)
        # tag label
        tag_gt_label = tf.py_func(func=self.parse_tag_multi_gt_label,
                                  inp=[parsed_example['tag_gt_label']],
                                  Tout=tf.float32)
        # cate label
        cate_gt_label = tf.py_func(func=self.parse_cate_gt_label,
                                   inp=[parsed_example['cate_gt_label']],
                                   Tout=tf.float32)

        return raw_vid, padding_title, tag_gt_label, \
            cate_gt_label, padding_rgb_fea, \
            rgb_fea_ture_frame, padding_audio_fea, \
            audio_fea_ture_frame

    def yield_one_batch_data(self):

        self.sess.run(self.dataset_init_op)
        while True:
            try:

                raw_vid_np, padding_title_np, tag_gt_label_np, \
                    cate_gt_label_np, padding_rgb_fea_np, \
                    rgb_fea_ture_frame_np, padding_audio_fea_np, \
                    audio_fea_ture_frame_np = self.sess.run(self.next_element)
                yield [raw_vid_np, padding_title_np,
                       tag_gt_label_np, cate_gt_label_np,
                       padding_rgb_fea_np, rgb_fea_ture_frame_np,
                       padding_audio_fea_np, audio_fea_ture_frame_np]

            except tf.errors.OutOfRangeError:
                print("End of dataset")
                break
