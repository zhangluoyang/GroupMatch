import math
from bitarray import bitarray
import cv2
from typing import Union, Tuple, List
import numpy as np


def cal_cnn_size(input_size, k_size: int, padding: int, stride: int):
    """
    计算卷积神经网络的输出尺寸
    :param input_size:
    :param k_size:
    :param padding:
    :param stride:
    :return:
    """
    return math.floor((input_size + 2 * padding - k_size) / stride) + 1


def cal_de_cnn_size(input_size: int, k_size: int, padding: int, stride: int):
    return stride * (input_size - 1) - 2 * padding + k_size


def make_divisible(x: Union[np.ndarray, int, float], divisible_by: int = 8):
    """
    保证尺寸是8倍的关系
    :param x:
    :param divisible_by:
    :return:
    """
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


def gray_image_dct(image: np.ndarray, dct_h: int = 8, dct_w: int = 8) -> np.ndarray:
    """
    灰度图片的离散余弦变换
    :param image:
    :param dct_h:
    :param dct_w:
    :return:
    """
    h, w = np.shape(image)
    float_image = np.zeros_like(image, np.float32)
    float_image[: h, : w] = image
    dct_transformer = cv2.dct(float_image)[: dct_h, : dct_w].flatten()
    return dct_transformer


def color_image_dct(image: np.ndarray, dct_h: int = 8, dct_w: int = 8) -> np.ndarray:
    """
    彩色图片的离散余弦变换
    :return:
    """
    b, g, r = cv2.split(image)
    b_dct = gray_image_dct(image=b, dct_h=dct_h, dct_w=dct_w)
    g_dct = gray_image_dct(image=g, dct_h=dct_h, dct_w=dct_w)
    r_dct = gray_image_dct(image=r, dct_h=dct_h, dct_w=dct_w)
    return np.concatenate([b_dct, g_dct, r_dct])


def dct_hash(image: np.ndarray, dct_h: int = 8, dct_w: int = 8, return_str: bool = True) -> Union[str, np.ndarray]:
    """
    离散余弦变换 + 感知哈希算法
    :param image:
    :param dct_h:
    :param dct_w:
    :param return_str: 是否返回字符串
    :return:
    """
    h, w = np.shape(image)
    float_image = np.zeros_like(image, np.float32)
    float_image[: h, : w] = image
    dct_transformer = cv2.dct(float_image)[: dct_h, : dct_w].flatten()
    avg = np.mean(dct_transformer)
    if return_str:
        hash_list = ['1' if v >= avg else '0' for v in dct_transformer]
        return "".join(hash_list)
    else:
        hash_list = [1 if v >= avg else 0 for v in dct_transformer]
        return np.array(hash_list, dtype=np.float32)


def _hamming_dist(bitarray1: str, bitarray2: str) -> int:
    """
    计算汉明距离
    :param bitarray1:
    :param bitarray2:
    :return:
    """
    xor_result = bitarray(bitarray1) ^ bitarray(bitarray2)
    return xor_result.count()


def _hamming_numpy(array1: np.ndarray, array2: np.ndarray) -> np.ndarray:
    """
    计算两个矩阵的汉明距离
    :param array1:
    :param array2:
    :return:
    """
    assert np.shape(array1) == np.shape(array2) or len(array1) == 1
    return np.sum(array1 != array2, axis=1)


def _norm_numpy(array: np.ndarray) -> np.ndarray:
    """
    归一化numpy
    :param array:
    :return:
    """
    if isinstance(array, list):
        array = np.array(array)
    # 归一化
    norm = np.linalg.norm(array, axis=1)
    norm = np.reshape(norm, (array.shape[0], 1))
    norm = np.divide(array, norm)
    return norm


def _similarity(array: np.ndarray, norm: np.ndarray) -> np.ndarray:
    """

    :param array: [feature_dim]
    :param norm: [instance_num, feature_dim]
    :return:
    """
    # 需要注意 这里已经归一化了 不需要再此进行归一化处理
    # array_norm = np.linalg.norm(array)
    # array_norm = np.divide(array, array_norm)
    return np.dot(norm, array.T)


def _top_k_index(top_k: int, similarity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """

    :param top_k: 取出top_k的index
    :param similarity: 相似度矩阵
    :return:
    """
    top_k_index = np.argsort(similarity)[::-1][: top_k]
    top_k_metrics = similarity[top_k_index]
    return top_k_index, top_k_metrics


def cal_similarity(a_list: List[float], b_list: List[float], ignore_ids: List[int] = None) -> float:
    """
    计算 N(a, b) / max(len(a), len(b))
    :param a_list
    :param b_list
    :param ignore_ids: 如果有忽略的 先进行忽略
    :return:
    """
    a_set = set(a_list)
    b_set = set(b_list)
    max_len = max(len(a_set), len(b_set))
    a = a_set.intersection(b_list)
    if ignore_ids is not None:
        a_set = set(list(filter(lambda e: e not in ignore_ids, a_list)))
        b_set = set(list(filter(lambda e: e not in ignore_ids, b_list)))
        a = a_set.intersection(b_set)
    return len(a) / max_len


def calculate_euclidean_distance(point: Union[Tuple[float, float], np.ndarray],
                                 points: Union[List[Tuple[float, float]], np.ndarray]) -> np.ndarray:
    """
    计算欧式距离
    :param point 点
    :param points 点列表
    """
    if isinstance(point, tuple):
        point = np.array([point])
    if isinstance(points, list):
        points = np.array(points)
    num, _ = np.shape(points)
    repeat_point = np.repeat(point, repeats=num, axis=0)
    return np.linalg.norm(repeat_point - points, axis=-1)
