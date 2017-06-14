#!/bin/bash
python tf_convert_data.py --dataset_name=pascalvoc --dataset_dir=/home/jph/data/VOC/VOCdevkit_trainval/VOC2007/ --output_name=voc_2007_train --output_dir=/home/jph/data/TF/voc07
python tf_convert_data.py --dataset_name=pascalvoc --dataset_dir=/home/jph/data/VOC/VOCdevkit_test/VOC2007/ --output_name=voc_2007_test --output_dir=/home/jph/data/TF/voc07

python tf_convert_data.py --dataset_name=pascalvoc --dataset_dir=/home/jph/data/VOC/VOCdevkit/VOC0712trainval/ --output_name=voc_0712_train --output_dir=/home/jph/data/TF/voc0712
python tf_convert_data.py --dataset_name=pascalvoc --dataset_dir=/home/jph/data/VOC/VOCdevkit/VOC07test/ --output_name=voc_07_test --output_dir=/home/jph/data/TF/voc0712

pip3 install enum34
# train on voc07
#python3 train_ssd_network.py --train_dir=/home/jph/exp/ssd0601257 --dataset_dir=/home/jph/data/TF/voc07 --dataset_name=pascalvoc_2007 --dataset_split_name=train --model_name=ssd_300_vgg --checkpoint_path=./checkpoints/ssd_300_vgg --save_summaries_secs=60 --save_interval_secs=600 --weight_decay=0.0005 --optimizer=adam --learning_rate=0.001 --batch_size=32
python3 train_ssd_network.py --train_dir=/home/jph/exp/ssd06111319 --dataset_dir=/home/jph/data/TF/voc07 --dataset_name=pascalvoc_2007 --dataset_split_name=train --model_name=ssd_300_vgg --checkpoint_path=/home/jph/modelZoo/TF/vgg_16.ckpt --checkpoint_model_scope=vgg_16 --checkpoint_exclude_scopes=ssd_300_vgg/conv6,ssd_300_vgg/conv7,ssd_300_vgg/block8,ssd_300_vgg/block9,ssd_300_vgg/block10,ssd_300_vgg/block11,ssd_300_vgg/block4_box,ssd_300_vgg/block7_box,ssd_300_vgg/block8_box,ssd_300_vgg/block9_box,ssd_300_vgg/block10_box,ssd_300_vgg/block11_box --trainable_scopes=ssd_300_vgg/conv6,ssd_300_vgg/conv7,ssd_300_vgg/block8,ssd_300_vgg/block9,ssd_300_vgg/block10,ssd_300_vgg/block11,ssd_300_vgg/block4_box,ssd_300_vgg/block7_box,ssd_300_vgg/block8_box,ssd_300_vgg/block9_box,ssd_300_vgg/block10_box,ssd_300_vgg/block11_box --save_summaries_secs=60 --save_interval_secs=600 --weight_decay=0.0005 --optimizer=adam --learning_rate=0.001 --learning_rate_decay_factor=0.94 --batch_size=32

python3 eval_ssd_network.py --eval_dir=/home/jph/exp/ssd06111319 --dataset_dir=/home/jph/data/TF/voc07 --dataset_name=pascalvoc_2007 --dataset_split_name=test --model_name=ssd_300_vgg --checkpoint_path=/home/jph/exp/ssd06111319/ --batch_size=1

# train on voc0712
python3 train_ssd_network.py --train_dir=/home/jph/exp/ssd0712_06111413 --dataset_dir=/home/jph/data/TF/voc0712 --dataset_name=pascalvoc_0712 --dataset_split_name=train --model_name=ssd_300_vgg --checkpoint_path=/home/jph/modelZoo/TF/vgg_16.ckpt --checkpoint_model_scope=vgg_16 --checkpoint_exclude_scopes=ssd_300_vgg/conv6,ssd_300_vgg/conv7,ssd_300_vgg/block8,ssd_300_vgg/block9,ssd_300_vgg/block10,ssd_300_vgg/block11,ssd_300_vgg/block4_box,ssd_300_vgg/block7_box,ssd_300_vgg/block8_box,ssd_300_vgg/block9_box,ssd_300_vgg/block10_box,ssd_300_vgg/block11_box --trainable_scopes=ssd_300_vgg/conv6,ssd_300_vgg/conv7,ssd_300_vgg/block8,ssd_300_vgg/block9,ssd_300_vgg/block10,ssd_300_vgg/block11,ssd_300_vgg/block4_box,ssd_300_vgg/block7_box,ssd_300_vgg/block8_box,ssd_300_vgg/block9_box,ssd_300_vgg/block10_box,ssd_300_vgg/block11_box --save_summaries_secs=60 --save_interval_secs=600 --weight_decay=0.0005 --optimizer=adam --learning_rate=0.001 --learning_rate_decay_factor=0.94 --batch_size=32
python3 train_ssd_network.py --train_dir=/home/jph/exp/ssd0712_06111413_2 --dataset_dir=/home/jph/data/TF/voc0712 --dataset_name=pascalvoc_0712 --dataset_split_name=train --model_name=ssd_300_vgg --checkpoint_path=/home/jph/exp/ssd0712_06111413 --checkpoint_model_scope=ssd_300_vgg --save_summaries_secs=60 --save_interval_secs=600 --weight_decay=0.0005 --optimizer=adam --learning_rate=0.00001 --learning_rate_decay_factor=0.94 --batch_size=32

CUDA_VISIBLE_DEVICES='' python3 eval_ssd_network.py --eval_dir=/home/jph/exp/ssd0712_06111413 --dataset_dir=/home/jph/data/TF/voc0712 --dataset_name=pascalvoc_0712 --dataset_split_name=test --model_name=ssd_300_vgg --checkpoint_path=/home/jph/exp/ssd0712_06111413/ --batch_size=1

# convert models
export PYTHONPATH=/home/jph/code/caffe/python/
python caffe_to_tensorflow.py --model_name=ssd_300_vgg --num_classes=21 --caffemodel_path=/home/jph/code/SSD-Tensorflow/ckpt/models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel
python3 eval_ssd_network.py --eval_dir=/home/jph/code/SSD-Tensorflow/ckpt/models/VGGNet/VOC0712/SSD_300x300/ --dataset_dir=/home/jph/data/TF/voc07 --dataset_name=pascalvoc_2007 --dataset_split_name=test --model_name=ssd_300_vgg --checkpoint_path=/home/jph/code/SSD-Tensorflow/ckpt/models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.ckpt --batch_size=1


## debug YOLOV2 merge
python3 train_yolov2_network.py --train_dir=/home/jph/exp/yolov2_debug --dataset_dir=/home/jph/data/TF/voc0712 --dataset_name=pascalvoc_0712 --dataset_split_name=train --model_name=yolov2_416 --checkpoint_model_scope=ssd_300_vgg --save_summaries_secs=60 --save_interval_secs=600 --weight_decay=0.0005 --optimizer=adam --learning_rate=0.00001 --learning_rate_decay_factor=0.94 --batch_size=32 --preprocessing_name=ssd_300_vgg


