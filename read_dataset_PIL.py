import os
import random
import time
import tensorflow as tf
from tensorflow import keras
from PIL import Image
# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True

# 记录程序开始时间
start_time = time.time()

label_dict = {'相机': 0, '机箱': 1, '键盘': 2, '笔记本': 3, '显示器': 4,
              '鼠标': 5, '收音机': 6, '路由器': 7, '手机': 8, '电话高清': 9}


def read_and_save_data(image_dir,
                       tfrecord_file_train,
                       tfrecord_file_val,
                       tfrecord_file_test):
    # 获取文件列表并随机打乱
    file_list = os.listdir(image_dir)
    random.shuffle(file_list)

    # 计算训练集、验证集和测试集的大小，这里以7:2:1的比例分配
    n_train = int(0.7 * len(file_list))
    n_val = int(0.2 * len(file_list))
    n_test = len(file_list) - n_train - n_val

    # 分割数据集
    train_files = file_list[:n_train]
    val_files = file_list[n_train:n_train + n_val]
    test_files = file_list[-n_test:]

    # 创建 TFRecord 文件写入器
    train_writer = tf.io.TFRecordWriter(tfrecord_file_train)
    val_writer = tf.io.TFRecordWriter(tfrecord_file_val)
    test_writer = tf.io.TFRecordWriter(tfrecord_file_test)

    # 遍历文件夹中的所有 PNG 图像，并将其转换为 TFExample 格式，写入 TFRecord 文件
    for filename in file_list:
        if filename.endswith('.png'):
            image_path = os.path.join(image_dir, filename)
            label = filename.split('_')[0]  # 假设标签信息在文件名的第一个下划线之前
            label_id = label_dict[label]
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                img = img.resize((224, 224))  # 调整图像大小
                image_array = keras.preprocessing.image.img_to_array(img)
                image_bytes = image_array.tobytes()
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes])),
                            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label_id]))
                        }
                    )
                )
                # 根据文件名将数据写入不同的 TFRecord 文件
                if filename in train_files:
                    train_writer.write(example.SerializeToString())
                elif filename in val_files:
                    val_writer.write(example.SerializeToString())
                elif filename in test_files:
                    test_writer.write(example.SerializeToString())

    # 关闭 TFRecord 文件写入器
    train_writer.close()
    val_writer.close()
    test_writer.close()


read_and_save_data('./datasets/camera_datasets_png',
                   './datasets/camera_train.tfrecord',
                   './datasets/camera_val.tfrecord',
                   './datasets/camera_test.tfrecord')

# read_and_save_data('./datasets/chassis_datasets_png',
#                    './datasets/chassis_train.tfrecord',
#                    './datasets/chassis_val.tfrecord',
#                    './datasets/chassis_test.tfrecord')
# 记录程序结束时间
end_time = time.time()

# 计算程序运行时间并输出结果
print("Program execution time:", end_time - start_time, "seconds")
# read_and_save_data('./datasets/camera_datasets', './datasets/camera.tfrecord')完毕
# read_and_save_data('./datasets/chassis_datasets', './datasets/chassis.tfrecord')完毕
# read_and_save_data('./datasets/keyboard_datasets', './datasets/keyboard.tfrecord')完毕
# read_and_save_data('./datasets/laptop_datasets', './datasets/laptop.tfrecord')  # 有问题
# read_and_save_data('./datasets/monitor_datasets', './datasets/monitor.tfrecord')完毕
# read_and_save_data('./datasets/mouse_datasets', './datasets/mouse.tfrecord')  # 有问题
# read_and_save_data('./datasets/radio_datasets', './datasets/radio.tfrecord')完毕
# read_and_save_data('./datasets/router_datasets', './datasets/router.tfrecord')完毕
# read_and_save_data('./datasets/smartphone_datasets', './datasets/smartphone.tfrecord')完毕
# read_and_save_data('./datasets/telephone_datasets', './datasets/telephone.tfrecord')完毕
