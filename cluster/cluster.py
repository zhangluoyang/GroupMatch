"""
colour cluster
@Author: zhangluoyang
@E-mail: 55058629@qq.com
@Time:  12月 16, 2020
"""
import numpy as np
from typing import Union
from sklearn.cluster import KMeans, MiniBatchKMeans


def colour_cluster_center(x: np.ndarray, num_clusters: int, batch_size: Union[int, None] = 1024) -> np.ndarray:
    """
    :param x: colour image dataset [num, height, width, 3]
    :param num_clusters:
    :param batch_size:
    :return:
    """
    # [num, height, width, 3] -> [num * height * width, 3]
    pixels = x.reshape(-1, x.shape[-1])
    if batch_size:
        mini_batch_k_mean_s = MiniBatchKMeans(n_clusters=num_clusters, random_state=0, batch_size=batch_size)
        mini_batch_k_mean_s.fit(pixels)
        return mini_batch_k_mean_s.cluster_centers_
    else:
        k_mean_s = KMeans(n_clusters=num_clusters, random_state=0)
        k_mean_s.fit(pixels)
        return k_mean_s.cluster_centers_
