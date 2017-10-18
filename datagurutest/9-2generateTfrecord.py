# coding: utf-8

# In[2]:

import tensorflow as tf
import os
import random
import math
import sys

# In[3]:

# 验证集数量
_NUM_TEST = 100
# 随机种子
_RANDOM_SEED = 0
# 数据块
_NUM_SHARDS = 5
# 数据集路径
DATASET_DIR = "/home/menu/PycharmProjects/deeplearning/data/train/"
# 标签文件名字
LABELS_FILENAME = "/home/menu/PycharmProjects/deeplearning/data/train/labels.txt"


# 定义tfrecord文件的路径+名字  就是字符串的组成   %s    %05d  都是占位符，拿后面的参数代进去
def _get_dataset_filename(dataset_dir, split_name, shard_id):
    output_filename = 'image_%s_%05d-of-%05d.tfrecord' % (split_name, shard_id, _NUM_SHARDS)
    return os.path.join(dataset_dir, output_filename)


# 判断tfrecord文件是否存在
def _dataset_exists(dataset_dir):
    for split_name in ['train', 'test']:
        for shard_id in range(_NUM_SHARDS):
            # 定义tfrecord文件的路径+名字
            output_filename = _get_dataset_filename(dataset_dir, split_name, shard_id)
        if not tf.gfile.Exists(output_filename):
            return False
    return True


# 获取所有文件路径以及图片分类马名称
# 传入 图片的路径
def _get_filenames_and_classes(dataset_dir):
    # 数据目录，就是 图片路径下面 子类型 文件夹
    # 把 路径保存在 directories  list 里面
    directories = []
    # 分类名称  把 文件夹的名称保存在directories
    class_names = []
    for filename in os.listdir(dataset_dir):
        # 合并文件路径
        path = os.path.join(dataset_dir, filename)
        # 判断该路径是否为目录
        if os.path.isdir(path):
            # 加入数据目录
            directories.append(path)
            # 加入类别名称
            class_names.append(filename)

    photo_filenames = []
    # 循环每个分类的文件夹
    for directory in directories:
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            # 把图片加入图片列表
            photo_filenames.append(path)

    return photo_filenames, class_names


# 使用的是整数 的转换
def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


# 使用的是字符串的话 调用 tf.train.BytesList  转换下
def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


# 调用tf.train.Example  转成tfrecord
def image_to_tfexample(image_data, image_format, class_id):
    # Abstract base class for protocol messages.
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_data),
        'image/format': bytes_feature(image_format),
        'image/class/label': int64_feature(class_id),
    }))


def write_label_file(labels_to_class_names, dataset_dir, filename=LABELS_FILENAME):
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'w') as f:
        for label in labels_to_class_names:
            class_name = labels_to_class_names[label]
            f.write('%d:%s\n' % (label, class_name))


# 把数据转为TFRecord格式
# split_name  是传 train  还是test
# filenames  图片的路径 list
# 分类字典  比如 'house': 3
# dataset_dir   图片目录的路径
def _convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir):
    assert split_name in ['train', 'test']
    # 计算每个数据块有多少数据
    # 把数据做切分，如果数据比较少的话 不用切分，存在一个 tfrecord 文件，数据比较大的时候 可以进行切分到几个 分片里面
    # 这里_NUM_SHARDS  是5个分片
    num_per_shard = int(len(filenames) / _NUM_SHARDS)
    # 去运行会话
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # 在会话里面去循环每一个分片
            for shard_id in range(_NUM_SHARDS):
                # 定义tfrecord文件的路径+名字
                output_filename = _get_dataset_filename(dataset_dir, split_name, shard_id)
                # 这个就固定的 使用TFRecordWriter
                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    # 每一个数据块开始的位置
                    start_ndx = shard_id * num_per_shard
                    # 每一个数据块最后的位置
                    end_ndx = min((shard_id + 1) * num_per_shard, len(filenames))
                    for i in range(start_ndx, end_ndx):
                        # 有可能 有些图片 是损坏的，损坏的文件 用TFRecordWriter  读文件就会报错
                        try:
                            # 把处理到 第几个图片打印出来
                            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (i + 1, len(filenames), shard_id))
                            sys.stdout.flush()
                            # 读取图片
                            image_data = tf.gfile.FastGFile(filenames[i], 'r').read()
                            # 获得 最后一个文件夹的名字 就是分类的名字   获得图片的类别名称
                            class_name = os.path.basename(os.path.dirname(filenames[i]))
                            # 找到类别名称对应的id
                            class_id = class_names_to_ids[class_name]
                            # 生成tfrecord文件    把图片的数据 ，图片的格式  以及图片分类的id
                            example = image_to_tfexample(image_data, b'jpg', class_id)
                            # 把数据写入到    DATASET_DIR  这个目录下面
                            tfrecord_writer.write(example.SerializeToString())
                        except IOError as e:
                            print("Could not read:", filenames[i])
                            print("Error:", e)
                            print("Skip it\n")

    sys.stdout.write('\n')
    sys.stdout.flush()


if __name__ == '__main__':
    # 判断tfrecord文件是否存在
    if _dataset_exists(DATASET_DIR):
        print('tfcecord文件已存在')
    else:
        # 获得所有图片以及分类
        photo_filenames, class_names = _get_filenames_and_classes(DATASET_DIR)
        # 把分类转为字典格式，类似于{'house': 3, 'flower': 1, 'plane': 4, 'guitar': 2, 'animal': 0}
        class_names_to_ids = dict(zip(class_names, range(len(class_names))))

        # 把数据切分为训练集和测试集
        # 设置seed  每次的随机数 都一样的
        random.seed(_RANDOM_SEED)
        # 调用shuffle  就可以把 photo_filenames  list 里面的记录给打乱  photo_filenames list  是所有图片的路径
        random.shuffle(photo_filenames)
        training_filenames = photo_filenames[_NUM_TEST:]
        testing_filenames = photo_filenames[:_NUM_TEST]

        # 数据转换
        _convert_dataset('train', training_filenames, class_names_to_ids, DATASET_DIR)
        _convert_dataset('test', testing_filenames, class_names_to_ids, DATASET_DIR)

        # 输出labels文件
        labels_to_class_names = dict(zip(range(len(class_names)), class_names))
        write_label_file(labels_to_class_names, DATASET_DIR)


# In[ ]:



