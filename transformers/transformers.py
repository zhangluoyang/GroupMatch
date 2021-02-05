"""
一些常用的数据变换
@Author: zhangluoyang
@E-mail: 55058629@qq.com
@Time:  12月 17, 2020
"""
import torch
import numpy as np


class Quantize(object):
    """
    像素值离散化  (按照距离聚类中心的标号离散化)
    """

    def __init__(self,
                 centroids: np.ndarray):
        """

        :param centroids: [num_clusters, 3]  3 通道数目
        """
        # [num_clusters, 3] -> [3, num_clusters]
        self.centroids = torch.transpose(torch.tensor(centroids), 0, 1)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """

        :param tensor: [C, H, W]
        :return: [H*W]
        """
        c, h, w = tensor.shape
        # [3, H, W] => [H, W, 3]
        tensor = tensor.permute(1, 2, 0).contiguous()
        # [H * W, 3]
        tensor = tensor.view(-1, c)
        d = self.squared_euclidean_distance(a=tensor)
        x = torch.argmin(d, 1)
        x = x.view(h * w)

        return x

    def squared_euclidean_distance(self, a: torch.Tensor) -> torch.Tensor:
        """
        计算距离
        :param a: [H * W, 3]
        :return: [H * W, num_clusters]
        """
        a2 = torch.sum(torch.square(a), dim=1, keepdims=True)
        b2 = torch.sum(torch.square(self.centroids), dim=0, keepdims=True)
        ab = torch.matmul(a, self.centroids)
        d = a2 - 2 * ab + b2
        return d

    def __repr__(self):
        return self.__class__.__name__ + '()'


class UnQuantize(object):
    """
    像素点反量化
    """

    def __init__(self, centroids: np.ndarray):
        """
        :param centroids: [num_clusters, 3]  3 通道数目
        """
        self.centroids = centroids

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """

        :param image: [h, w]
        :return:
        """
        return self.centroids[image]
