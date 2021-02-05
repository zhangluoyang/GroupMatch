"""
一些必要的采样算法
@Author: zhangluoyang
@E-mail: 55058629@qq.com
@Time:  1月 21, 2021
"""
import numpy as np
from typing import List, Dict
from torch.utils.data.sampler import Sampler


class RandomCycleIter(object):
    """
    一个自循环的数据迭代器 仅负责生成id
    """

    def __init__(self, data_ids: np.ndarray,
                 test_model: bool = False):
        """

        :param data_ids:  对应的数据id
        """
        if isinstance(data_ids, list):
            data_ids = np.array(data_ids)
        self.data_ids = data_ids
        self.index = 0
        self.length = len(data_ids)
        self.test_model = test_model

    def __iter__(self):
        return self

    def __next__(self):
        self.index += 1
        if self.index == self.length:
            self.index = 0
            if not self.test_model:
                np.random.shuffle(self.data_ids)
        return self.data_ids[self.index]


class RandomCycleIterInstance(object):
    """
    负责对每一个类别进行随机采样 每一个类别单独存储一个 RandomCycleIter 类 返回数据的index
    """

    def __init__(self, random_cycle_iter_list: List[RandomCycleIter]):
        """

        :param random_cycle_iter_list: 存储数据id的类别迭代器列表 下标对应类别下标
        """
        self.random_cycle_iter_list = random_cycle_iter_list

    def __sample__(self, sample_class_index: int,
                   sample_num: int):
        """
        待采样类别的数据下标 index
        :param sample_class_index 带采样的类别下标
        :param sample_num 采样的样本数目
        :return:
        """
        assert sample_class_index < len(self.random_cycle_iter_list)
        random_cycle_iter = self.random_cycle_iter_list[sample_class_index]
        return [next(random_cycle_iter) for _ in range(sample_num)]


class ClassAwareSampler(Sampler):
    """
    每一个样本数目 循环采样 (猜测是保证每一个匹次内样本数目一致的做法吧)
    """

    def __init__(self, class_data_dict: Dict[int, List[int]],
                 each_class_sample_num: int, test_model: bool = False):
        """
        :param class_data_dict: 类别 对应 数据的下标
        :param each_class_sample_num:  每一个类别采样的样本数目
        :param test_model: 是否是测试模式 如果是测试模式 则不进行数据的shuffle
        """
        super(ClassAwareSampler, self).__init__(class_data_dict)
        self.class_data_dict = class_data_dict
        self.each_class_sample_num = each_class_sample_num
        self.class_num = len(class_data_dict)
        self.test_model = test_model
        random_cycle_iter_list = [RandomCycleIter(data_ids=data_ids, test_model=test_model) for class_id, data_ids in
                                  list(class_data_dict.items())]
        self.random_cycle_iter_instance = RandomCycleIterInstance(random_cycle_iter_list=random_cycle_iter_list)

    def __len__(self):
        """
        每一次采样返回的样本数目
        :return:
        """
        return self.each_class_sample_num * len(self.class_data_dict)

    def __iter__(self):
        """
        返回数据的id
        :return:
        """
        batch_data_ids = []
        # 每一个样本都采样 each_class_sample_num 个样本
        for i in range(self.class_num):
            batch_data_ids.extend(self.random_cycle_iter_instance.__sample__(sample_num=self.each_class_sample_num,
                                                                             sample_class_index=i))
        return batch_data_ids
