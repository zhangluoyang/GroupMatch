"""
@Author: zhangluoyang
@E-mail: 55058629@qq.com
@Time:  12月 30, 2020
"""
import json
import cv2
import lmdb
import torch
import random
import numpy as np
import torchvision.transforms as T
import torch.utils.data as data
from typing import Dict, List, Tuple
from utils.s_utils import base64_to_image
from utils.utils import cal_similarity
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class LmdaDataSet(data.Dataset):

    def __init__(self, db_path: str, keys_path: str, transform: T.Compose = None):
        """

        :param db_path: 数据路径
        :param keys_path: key路径
        :param transform: 图像增强的一些操作
        """
        self.keys = np.load(keys_path)
        self.lmdb_connection = lmdb.open(db_path, subdir=False, readonly=True, lock=False,
                                         readahead=False, meminit=False)
        self.transform = transform

    def __len__(self):
        return len(self.keys)


class TripleDataSet(LmdaDataSet):

    def __init__(self, db_path: str, keys_path: str, zone_index_path: str, transform: T.Compose):
        """

        :param db_path:
        :param keys_path:
        :param zone_index_path: 记录每一个功能区的下标
        """
        super(TripleDataSet, self).__init__(db_path=db_path,
                                            keys_path=keys_path,
                                            transform=transform)
        # self.zone_index_path = zone_index_path
        self.zone_index_dict: Dict[int, List[int]] = json.load(open(zone_index_path, "r", encoding="utf-8"))

    def _similarity(self, model_list_a: List[int],
                    style_list_a: List[int],
                    brand_list_a: List[int],
                    series_list_a: List[int],
                    model_list_b: List[int],
                    style_list_b: List[int],
                    brand_list_b: List[int],
                    series_list_b: List[int],
                    ignore_ids: List[int] = None) -> float:
        """
        计算相似度
        :param model_list_a
        :param style_list_a 风格列表
        :param brand_list_a 品牌列表
        :param series_list_a 系列列表
        :param model_list_b
        :param style_list_b 风格列表
        :param brand_list_b 品牌列表
        :param series_list_b 系列列表
        :return:
        """
        model_id_sim = cal_similarity(a_list=model_list_a, b_list=model_list_b, ignore_ids=ignore_ids)
        style_sim = cal_similarity(a_list=style_list_a, b_list=style_list_b, ignore_ids=ignore_ids)
        brand_sim = cal_similarity(a_list=brand_list_a, b_list=brand_list_b, ignore_ids=ignore_ids)
        series_sim = cal_similarity(a_list=series_list_a, b_list=series_list_b, ignore_ids=ignore_ids)
        # if ignore_ids is not None:
        # print(model_id_sim, style_sim, brand_sim, series_sim)
        return 0.5 * model_id_sim + 0.3 * style_sim + 0.1 * brand_sim + 0.1 * series_sim

    def get_value_by_key(self, key: int) -> dict:
        with self.lmdb_connection.begin(write=False) as txn:
            return json.loads(txn.get("{0}".format(key).encode("utf-8")).decode("utf-8"))

    def negative_sample(self, data: dict, key_index: int) -> Tuple[np.ndarray, float]:
        """
        负采样一个样本
        :param data 当前样本
        :param key_index 下标列表 (采样的结果不应该是此值)
        :return: 图像，不相似的程度
        """
        zoneId = data["groupPictureLabelData"]["zoneId"]
        label_index = self.zone_index_dict["{0}".format(zoneId)]
        if len(label_index) == 1:
            return np.zeros((120, 120, 3), dtype=np.uint8), 1.0
        else:
            while True:
                negative_key = random.choice(label_index)
                if negative_key != key_index:
                    negative_data = self.get_value_by_key(key=negative_key)
                    generatePictureLabelDatas = negative_data["generatePictureLabelDatas"]
                    groupPictureLabelData = negative_data["groupPictureLabelData"]
                    negative_picture_datas = generatePictureLabelDatas + [groupPictureLabelData]
                    negative_picture_data = random.choice(negative_picture_datas)
                    image_ = base64_to_image(negative_picture_data["targetBase64Image"])
                    dis_sim = 1 - self._similarity(model_list_a=data["groupPictureLabelData"]["modelIds"],
                                                   style_list_a=data["groupPictureLabelData"]["styleIds"],
                                                   brand_list_a=data["groupPictureLabelData"]["brandIds"],
                                                   series_list_a=data["groupPictureLabelData"]["seriesIds"],
                                                   model_list_b=negative_picture_data["modelIds"],
                                                   style_list_b=negative_picture_data["styleIds"],
                                                   brand_list_b=negative_picture_data["brandIds"],
                                                   series_list_b=negative_picture_data["seriesIds"],
                                                   ignore_ids=[0, -1])
                    return image_, dis_sim

    def positive_sample(self, data: dict) -> Tuple[np.ndarray, float]:
        """
        正采样一个样本 (如果没有，则对图像进行翻转生成一个样本)
        :param data 一则数据
        :return: 图像, 相似度
        """
        if len(data["generatePictureLabelDatas"]) == 0:
            image = base64_to_image(data["groupPictureLabelData"]["targetBase64Image"])
            image_ = cv2.flip(image, 1)
            return image_, 1.0
        else:
            # 存在一定的概率 直接使用图像的翻转值
            if random.random() <= 0.25:
                image = base64_to_image(data["groupPictureLabelData"]["targetBase64Image"])
                image_ = cv2.flip(image, 1)
                return image_, 1.0
            else:
                generate_data = random.choice(data["generatePictureLabelDatas"])
                image_ = base64_to_image(generate_data["targetBase64Image"])
                sim = self._similarity(model_list_a=data["groupPictureLabelData"]["modelIds"],
                                       style_list_a=data["groupPictureLabelData"]["styleIds"],
                                       brand_list_a=data["groupPictureLabelData"]["brandIds"],
                                       series_list_a=data["groupPictureLabelData"]["seriesIds"],
                                       model_list_b=generate_data["modelIds"],
                                       style_list_b=generate_data["styleIds"],
                                       brand_list_b=generate_data["brandIds"],
                                       series_list_b=generate_data["seriesIds"])
                return image_, sim

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                              torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        :param item:
        :return: target_image, positive_image, negative_image, target, positive_weight, negative_weight
        """
        try:
            key = self.keys[item]
            # target 数据
            target_data = self.get_value_by_key(key=key)
            target_image = base64_to_image(base64_code=target_data["groupPictureLabelData"]["targetBase64Image"])
            target = target_data["groupPictureLabelData"]["label"]

            # 正样本及其权重
            positive_image, positive_weight = self.positive_sample(data=target_data)
            # 负样本及其权重
            negative_image, negative_weight = self.negative_sample(data=target_data, key_index=key)

            # 数据增强
            target_image = Image.fromarray(target_image)
            positive_image = Image.fromarray(positive_image)
            negative_image = Image.fromarray(negative_image)

            if self.transform is not None:
                target_image = self.transform(target_image)
                positive_image = self.transform(positive_image)
                negative_image = self.transform(negative_image)
            target = torch.tensor(target, dtype=torch.int64)
            positive_weight = torch.tensor(positive_weight, dtype=torch.float32)
            negative_weight = torch.tensor(negative_weight, dtype=torch.float32)
            return target_image, positive_image, negative_image, target, positive_weight, negative_weight
        except:
            # 发现有一条数据有问题 则直接使用一条随机的数据 直到正确为止
            return self.__getitem__(item=random.randint(a=0, b=self.__len__()))

# if __name__ == "__main__":
#     transform = T.Compose([T.Resize((256, 256), interpolation=Image.BICUBIC),
#                            T.RandomCrop(224),
#                            T.RandomHorizontalFlip(),
#                            T.ToTensor()])
#     data_dir = r"G:/group_image/train"
#     db_path = "{0}/data.db".format(data_dir)
#     keys_path = "{0}/keys.npy".format(data_dir)
#     zone_index_path = "{0}/zone_index.json".format(data_dir)
#     data_set = TripleDataSet(db_path=db_path, keys_path=keys_path, zone_index_path=zone_index_path,
#                              transform=transform)
#     t = data_set.__getitem__(item=2)
