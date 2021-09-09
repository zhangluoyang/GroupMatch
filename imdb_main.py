"""
构建imdb数据集
@Author: zhangluoyang
@E-mail: 55058629@qq.com
@Time:  12月 30, 2020
"""
import json
import os
import lmdb
import numpy as np
from PIL import ImageFile
from typing import Dict, List

ImageFile.LOAD_TRUNCATED_IMAGES = True


def label_count_dict(path: str) -> Dict[int, int]:
    """
    用于计算 label -> count 的映射 为测试机
    :param path 数据集
    :return:
    """
    file = open(path, "r", encoding="utf-8")
    label_count: Dict[int, int] = {}
    while True:
        line = file.readline().strip()
        if not line:
            break
        try:
            data = json.loads(line)
            label = data["groupPictureLabelData"]["label"]
            if label not in label_count:
                label_count[label] = 0
            label_count[label] = label_count[label] + 1
        except Exception as e:
            print("出现错误:{0}", e)
    return label_count


def generate_imdb(path: str,
                  data_path: str,
                  train_map_size: int = 2 ** 30,
                  test_map_size: int = 2 ** 30):
    """

    :param path:
    :param data_path: 数据集路径
    :param train_map_size:
    :param test_map_size:
    :return:
    """
    if not os.path.exists(path):
        os.mkdir(path)
    train_path = "{0}/train".format(path)
    test_path = "{0}/test".format(path)
    if not os.path.exists(train_path):
        os.mkdir(train_path)
    if not os.path.exists(test_path):
        os.mkdir(test_path)
    train_keys_path = "{0}/keys.npy".format(train_path)
    test_keys_path = "{0}/keys.npy".format(test_path)
    # 记录每一种类别的index 用于采样
    train_zone_index_path = "{0}/zone_index.json".format(train_path)
    test_zone_index_path = "{0}/zone_index.json".format(test_path)

    train_db_path = "{0}/data.db".format(train_path)
    test_db_path = "{0}/data.db".format(test_path)

    train_connection = lmdb.open(train_db_path,
                                 subdir=False,
                                 map_size=train_map_size,
                                 readonly=False,
                                 meminit=False,
                                 map_async=True)
    train_txn = train_connection.begin(write=True)

    test_connection = lmdb.open(test_db_path,
                                subdir=False,
                                map_size=test_map_size,
                                readonly=False,
                                meminit=False,
                                map_async=True)
    test_txn = test_connection.begin(write=True)

    label_count = label_count_dict(path=data_path)
    train_label_count: Dict[int, int] = {}
    test_label_count: Dict[int, int] = {}
    # 功能区id 映射的样本下标
    train_zone_index_dict: Dict[int, List[int]] = {}
    test_zone_index_dict: Dict[int, List[int]] = {}

    file = open(data_path, "r", encoding="utf-8")
    train_index = 0
    test_index = 0
    while True:
        line = file.readline()
        if not line:
            break
        try:
            data = json.loads(line)
        except Exception as e:
            print("出现错误:{0}".format(e))
            continue
        label = data["groupPictureLabelData"]["label"]
        zoneId = data["groupPictureLabelData"]["zoneId"]

        if zoneId not in (48, 49, 52, 54):
            continue

        count = label_count[label]
        if label not in train_label_count:
            train_label_count[label] = 0
        if label not in test_label_count:
            test_label_count[label] = 0
        # 仅有多余5张图片 才有可能作为测试样本 (测试类最多两个样本)
        if count >= 5 and train_label_count[label] >= 1 and test_label_count[label] < 2:
            if zoneId not in test_zone_index_dict:
                test_zone_index_dict[zoneId] = [test_index]
            else:
                test_zone_index_dict[zoneId].append(test_index)
            test_label_count[label] = test_label_count[label] + 1
            idx: bytes = u"{0}".format(test_index).encode("utf-8")
            value: bytes = u"{0}".format(line).encode("utf-8")
            test_txn.put(idx, value)
            test_index += 1
            print("增加一个测试样本")
            if test_index % 64 == 0:
                test_txn.commit()
                test_txn = test_connection.begin(write=True)
        else:
            if zoneId not in train_zone_index_dict:
                train_zone_index_dict[zoneId] = [train_index]
            else:
                train_zone_index_dict[zoneId].append(train_index)
            train_label_count[label] = train_label_count[label] + 1
            idx: bytes = u"{0}".format(train_index).encode("utf-8")
            value: bytes = u"{0}".format(line).encode("utf-8")
            train_txn.put(idx, value)
            train_index += 1
            print("增加一个训练样本")
            train_label_count[label] = train_label_count[label] + 1
            if train_index % 64 == 0:
                train_txn.commit()
                train_txn = train_connection.begin(write=True)
    train_txn.commit()
    test_txn.commit()
    np.save(train_keys_path, np.array(list(range(0, train_index))))
    np.save(test_keys_path, np.array(list(range(0, test_index))))
    json.dump(train_zone_index_dict, open(train_zone_index_path, "w", encoding="utf-8"))
    json.dump(test_zone_index_dict, open(test_zone_index_path, "w", encoding="utf-8"))


if __name__ == "__main__":
    data_path = r"G:/image_data.txt"
    path = r"G:/group_image"
    generate_imdb(path=path, data_path=data_path, train_map_size=25 * 2 ** 30, test_map_size=5 * 2 ** 30)
