"""
一些常用的hash算法
@Author: zhangluoyang
@E-mail: 55058629@qq.com
@Time:  11月 19, 2020
"""
import numpy as np
from typing import List, Union, Dict


class Instance(object):
    """
    用于存储数据的实例
    """

    def __init__(self, image: np.ndarray, feature: np.ndarray):
        self.image = image
        self.feature = feature


class Bucket(object):
    """
    桶数据
    """

    def __init__(self):
        self.instances: List[Instance] = []
        self._instance_feature: Union[np.ndarray, None] = None

    def add_instance(self, instance: Instance):
        self.instances.append(instance)

    def k_nearst_neighbor(self, query_feature: np.ndarray, k: int = 8) -> List[np.ndarray]:
        """
        找出最近的k个结果
        :param query_feature: 查询的特征 [1, feature_dim]
        :param k:
        :return:
        """
        if self._instance_feature is None:
            self._instance_feature = np.array([instance.feature for instance in self.instances])
        # [1, num]
        similarity = np.dot(query_feature, self._instance_feature.T)
        # top k 的结果
        index_s = np.argsort(-similarity)
        top_k_indexs = index_s[: k]
        return [self.instances[index].image for index in top_k_indexs]


class InMemoryStorage(object):
    """
    存储在内存内部的数据结构
    """

    def __init__(self):
        self.storage: Dict[str, Bucket] = dict()

    def append_val(self, key: str, instance: Instance):
        """

        :param key: hash编码值
        :param instance: 实例数据
        :return:
        """
        self.storage.setdefault(key, Bucket()).add_instance(instance=instance)


class LSHash(object):
    """
    局部敏感哈希算法
    """

    def __init__(self, hash_size: int, input_dim: int, num_hash_tables: int = 4, save_path: str = None):
        """

        :param hash_size:   哈希值大小
        :param input_dim:   输入特征尺寸
        :param num_hash_tables:   哈希表的数目
        :param save_path: 存储的路径
        """

        self.hash_size = hash_size
        self.input_dim = input_dim
        self.num_hash_tables = num_hash_tables
        self.save_path = save_path
        # 用于计算哈希值的平面
        self.uniform_planes = self._init_uniform_planes()
        # 用于存储哈希表的值
        self.hash_tables = self._init_hash_tables()

    def _init_hash_tables(self) -> List[InMemoryStorage]:
        """
        存储hash表
        :return:
        """

        return [InMemoryStorage() for i in range(self.num_hash_tables)]

    def _init_uniform_planes(self) -> List[np.ndarray]:
        """
        初始化 用于计算 hash值
        :return:
        """
        return [np.random.randn(self.hash_size, self.input_dim) for _ in range(self.num_hash_tables)]

    def index(self, instance: Instance):
        """

        :param instance:  输入的数据
        :return:
        """
        for i, table in enumerate(self.hash_tables):
            table.append_val(key=self._hash(plane=self.uniform_planes[i], input_point=instance.feature),
                             instance=instance)

    def _hash(self, plane: np.ndarray, input_point: Union[np.ndarray, List[float]]) -> str:
        """
        依据 planes计算哈希值 并返回
        :param plane: 用于计算哈希值的平面  [hash_size, input_dim]
        :param input_point: 输入点 [1, input_dim]
        :return:
        """
        if isinstance(input_point, list):
            input_point = np.array(input_point)
        # [batch_size, hash_size]
        projections = np.dot(plane, input_point)
        return "".join(['1' if i > 0 else '0' for i in projections])

    def _list_str(self, key: str) -> List[str]:
        """
        返回list str数值
        :return:
        """
        return [k for k in key]
