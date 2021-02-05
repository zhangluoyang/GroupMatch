"""
@Author: zhangluoyang
@E-mail: 55058629@qq.com
@Time:  12月 16, 2020
"""
import torch
from skimage import io
import numpy as np
import random
from PIL import Image
from PIL import ImageFile
import torch.utils.data as data
from torchvision import transforms
import torchvision.transforms as T
from typing import Union, Tuple, List, Dict
from utils import file_utils as fs

# 图像太大的时候处理方法
ImageFile.LOAD_TRUNCATED_IMAGES = True

transform_ = T.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    T.ToTensor()
])


class ImageDataFolder(data.Dataset):

    def __init__(self, data_path: str, transform: Union[transforms.Compose, None] = transform_):
        # 类别
        dir_names = fs.get_dir_paths(data_path)
        self.label_names = dir_names
        self.name_to_id_dict: Dict[str, int] = {}
        for name in dir_names:
            self.name_to_id_dict[name] = len(self.name_to_id_dict)
        self.transform = transform
        self.image_paths: List[str] = []
        for name in dir_names:
            dir_name_path = "{0}/{1}".format(data_path, name)
            image_file_paths = fs.get_file_paths(dir_name_path)
            self.image_paths.extend(image_file_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[item]
        image = io.imread(image_path)
        if np.shape(image)[-1] == 4:
            image = image[:, :, :3]
        name = fs.get_parent_path_name(image_path)
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(self.name_to_id_dict[name], dtype=torch.int32)
        return image, label


class ImageDataFolderMulti(data.Dataset):

    def __init__(self, data_path: str, transform: Union[transforms.Compose, None] = transform_):
        # 第一种类别
        first_dir_names = fs.get_dir_paths(data_path)
        self.first_label_names = first_dir_names

        self.first_name_to_id_dict: Dict[str, int] = {}
        for name in first_dir_names:
            self.first_name_to_id_dict[name] = len(self.first_name_to_id_dict)

        # 第二种类别
        second_dir_names = []
        for first_dir_name in first_dir_names:
            for name in fs.get_dir_paths("{0}/{1}".format(data_path, first_dir_name)):
                if name not in second_dir_names:
                    second_dir_names.append(name)

        self.second_name_to_id_dict: Dict[str, int] = {}
        for name in second_dir_names:
            self.second_name_to_id_dict[name] = len(self.second_name_to_id_dict)
        self.transform = transform
        self.image_paths: List[str] = []
        for name in first_dir_names:
            dir_name_path = "{0}/{1}".format(data_path, name)
            image_file_paths = fs.get_file_paths(dir_name_path)
            self.image_paths.extend(image_file_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        image_path = self.image_paths[item]
        image = io.imread(image_path)
        if np.shape(image)[-1] == 4:
            image = image[:, :, :3]
        second_name = fs.get_parent_path_name(image_path)
        first_name = fs.get_grandfather_path_name(image_path)
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        first_label = torch.tensor(self.first_name_to_id_dict[first_name], dtype=torch.int64)
        second_label = torch.tensor(self.second_name_to_id_dict[second_name], dtype=torch.int64)
        return image, first_label, second_label


class ImageDataFileMulti(data.Dataset):
    def __init__(self, file_path: str, transform: Union[transforms.Compose, None] = transform_):
        file = open(file_path, "r", encoding="utf-8")
        lines: List[str] = file.readlines()
        self.lines = [line.strip() for line in lines]
        # 第一种类别
        first_names = []
        second_names = []
        for line in self.lines:
            split_line = line.split("/")
            if split_line[-3] not in first_names:
                first_names.append(split_line[-3])
            if split_line[-2] not in second_names:
                second_names.append(split_line[-2])
        self.first_name_to_id_dict: Dict[str, int] = {}
        for name in first_names:
            self.first_name_to_id_dict[name] = len(self.first_name_to_id_dict)
        self.second_name_to_id_dict: Dict[str, int] = {}
        for name in second_names:
            self.second_name_to_id_dict[name] = len(self.second_name_to_id_dict)
        self.transform = transform

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        try:
            image_path = self.lines[item]
            image = io.imread(image_path)
            shape = np.shape(image)
            if len(shape) == 3:
                if np.shape(image)[-1] == 4:
                    image = image[:, :, :3]
            if len(shape) == 2:
                image: np.ndarray = image[:, :, np.newaxis]
                image = np.concatenate([image.copy(), image.copy(), image.copy()], axis=-1)
            split_line = image_path.split("/")
            first_name = split_line[-3]
            second_name = split_line[-2]
            image = Image.fromarray(image)
            if self.transform is not None:
                image = self.transform(image)
            first_label = torch.tensor(self.first_name_to_id_dict[first_name], dtype=torch.int64)
            second_label = torch.tensor(self.second_name_to_id_dict[second_name], dtype=torch.int64)
            return image, first_label, second_label
        except:
            # 如果当前数据异常 则任意返回一个正确的数据
            return self.__getitem__(item=random.randint(0, self.__len__()))
