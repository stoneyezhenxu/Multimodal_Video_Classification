#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : train.py
# @Author: stoneye
# @Date  : 2023/09/01
# @Contact : stoneyezhenxu@gmail.com

import sys
import time
import random
import tensorflow as tf
import os
import numpy as np
import json
from efficient_and_vggish_embedding.rgb_audio_feature_extraction import RgbAudioExtractorClass
from clip_ref.extract_clip import CLIP


def get_efficient_fea_by_images_dir(efficient_and_vggish_extractor, images_dir):
    '''

    Args:
        efficient_and_vggish_extractor:
        images_dir:

    Returns:

    '''
    rgb_filename_list = sorted(os.listdir(images_dir))
    rgb_path_list = [os.path.join(images_dir, filename) for filename in rgb_filename_list]
    rgb_emb = efficient_and_vggish_extractor.image_efficient_feature_extractor(rgb_path_list)
    return rgb_emb.cpu().numpy()


def get_vggish_fea_by_wav_path(efficient_and_vggish_extractor, wav_path):
    '''

    Args:
        efficient_and_vggish_extractor:
        wav_path:

    Returns:

    '''
    audio_emb = efficient_and_vggish_extractor.audio_vggish_feature_extractor(wav_path)
    return audio_emb.cpu().numpy()


def get_train_vid2dict(base_dir):
    vid2dict = {}
    for cur_vid in os.listdir(base_dir):
        vid = cur_vid.split('.wav')[0]

        vid2dict[vid] = 1
    return vid2dict


def main():
    tfrecord_name = sys.argv[1]
    BASE_WAV = '/data/pcg_ceph/stoneye/2021_mcn_project/data/wavs'
    BASE_IMAGE = '/data/pcg_ceph/stoneye/2021_mcn_project/data/images'
    OUT_TFRECORD = '/data/pcg_ceph/stoneye/2021_mcn_project/data/tfrecord/{}.tfrecord'.format(tfrecord_name)

    vid2dict = get_train_vid2dict(base_dir=BASE_WAV)

    clip_extractor = CLIP(weights='/data/pcg_ceph/ccks/tfrecord/clip_vgg_fea_extractor/clip_ref/models/model.pt')
    efficient_and_vggish_extractor = RgbAudioExtractorClass(
        model_path='/data/pcg_ceph/ccks/tfrecord/clip_vgg_fea_extractor/efficient_and_vggish_embedding/models',
        is_xiaoshipin=True)

    fw = tf.compat.v1.python_io.TFRecordWriter(OUT_TFRECORD)

    num = 0
    for vid, _ in vid2dict.items():

        try:

            # # 额外新增的特征
            start = time.time()
            wav_path = os.path.join(BASE_WAV, '{}.wav'.format(vid))
            images_dir = os.path.join(BASE_IMAGE, vid)
            # #3、extractor wav vggish fea
            vggish_wav_emb = get_vggish_fea_by_wav_path(efficient_and_vggish_extractor=efficient_and_vggish_extractor,
                                                        wav_path=wav_path)
            print("vggish_wav_emb.shape: ", vggish_wav_emb.shape)

            # 8、extractor  clip_ref frames fea
            clip_frames_rgb_emb = clip_extractor.extract_fea(images_dir, is_one_images=False)
            print("clip_frames_rgb_emb.shape: ", clip_frames_rgb_emb.shape)

            shape_vggish_wav_emb = [x for x in vggish_wav_emb.shape]
            shape_clip_frames_rgb_emb = [x for x in clip_frames_rgb_emb.shape]

            final_vggish_wav_emb = vggish_wav_emb.reshape(-1).astype(np.float16)
            final_clip_frames_rgb_emb = clip_frames_rgb_emb.reshape(-1).astype(np.float16)

            features = {}
            features['vid'] = tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[vid.encode()]))

            features['vggish_wav_emb'] = tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[final_vggish_wav_emb.tostring()]))
            features['clip_frames_rgb_emb'] = tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[final_clip_frames_rgb_emb.tostring()]))
            features['shape_vggish_wav_emb'] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=shape_vggish_wav_emb))
            features['shape_clip_frames_rgb_emb'] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=shape_clip_frames_rgb_emb))

            tf_features = tf.train.Features(feature=features)
            tf_example = tf.train.Example(features=tf_features)
            fw.write(tf_example.SerializeToString())
            end = time.time()
            num += 1
            print("success!!!")
            print("cur num:{};  cost_time:{}".format(num, end - start))
            # f_success.write(vid + '\n')
        except Exception as e:
            print("\n\nerror: vid:{} error_infos:{}\n\n".format(vid, e))
            # f_error.write(vid+'\n')
            continue


if __name__ == '__main__':
    main()
