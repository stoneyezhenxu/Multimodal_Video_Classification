#1、train cate
cd src
python -u train.py \
  --mode_type train \
  --task_type cate \
  --model_type nextvlad \
  --segment_or_max_padding segment \
  --num_gpu 2 \
  --cate_nums 206 \
  --tag_nums 20000 \
  --title_max_len 30 \
  --ad_strength 0.5 \
  --rgb_frames 64 \
  --audio_frames 64 \
  --batch_size 100 \
  --lr 0.001 \
  --keep_dropout 0.8 \
  --word_vocab /data/ceph/vocab/tencent_ailab_50w_200_emb_vocab.txt \
  --train_path /data/ceph/dataset/tfrecord/train \
  --valid_path /data/ceph/dataset/tfrecord/valid \
  --test_path /data/ceph/dataset/tfrecord/test \
  --pre_Train_path /data/ceph/dataset/vocab/head_50w_Tencent_AILab_emb_200.txt \
  --report_freq 1000 \
  --one_epoch_step_eval_num 4 \
  --epoch 20 \
  --model_dir ./models/cate_model | tee ./log/train_cate.log

#2、train tag
cd src
python -u train.py \
  --mode_type train \
  --task_type tag \
  --model_type nextvlad \
  --segment_or_max_padding segment \
  --num_gpu 2 \
  --cate_nums 206 \
  --tag_nums 20000 \
  --title_max_len 30 \
  --ad_strength 0.5 \
  --rgb_frames 64 \
  --audio_frames 64 \
  --batch_size 100 \
  --lr 0.001 \
  --keep_dropout 0.8 \
  --word_vocab /data/ceph/dataset/vocab/tencent_ailab_50w_200_emb_vocab.txt \
  --train_path /data/ceph/dataset/tfrecord/train \
  --valid_path /data/ceph/dataset/tfrecord/valid \
  --test_path /data/ceph/dataset/tfrecord/test \
  --pre_Train_path /data/ceph/dataset/vocab/head_50w_Tencent_AILab_emb_200.txt \
  --report_freq 1000 \
  --one_epoch_step_eval_num 4 \
  --epoch 20 \
  --model_dir ./models/tag_model | tee ./log/train_tag.log

#3、train multi_task:cate_and_tag
cd src
python -u train.py \
  --mode_type train \
  --task_type cate_and_tag \
  --model_type nextvlad \
  --segment_or_max_padding segment \
  --num_gpu 2 \
  --cate_nums 206 \
  --tag_nums 20000 \
  --title_max_len 30 \
  --ad_strength 0.5 \
  --rgb_frames 64 \
  --audio_frames 64 \
  --batch_size 100 \
  --lr 0.001 \
  --keep_dropout 0.8 \
  --word_vocab /data/ceph/dataset/vocab/tencent_ailab_50w_200_emb_vocab.txt \
  --train_path /data/ceph/dataset/tfrecord/train \
  --valid_path /data/ceph/dataset/tfrecord/valid \
  --test_path /data/ceph/dataset/tfrecord/test \
  --pre_Train_path /data/ceph/dataset/vocab/head_50w_Tencent_AILab_emb_200.txt \
  --report_freq 1000 \
  --one_epoch_step_eval_num 4 \
  --epoch 20 \
  --model_dir ./models/cate_and_tag_model | tee ./log/train_cate_and_tag.log
