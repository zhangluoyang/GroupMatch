"""
分类模块
@Author: zhangluoyang
@E-mail: 55058629@qq.com
@Time:  11月 13, 2020
"""
import torch
import torch.nn as nn
from typing import Dict, List


class BasicClassification(nn.Module):
    """
    基础的分类模块
    """

    def __init__(self, feature_module: nn.Module,
                 input_tensor_names: List[str]):
        """

        :param feature_module: 特征提取器
        :param input_tensor_names: 输出tensor名称
        """
        super(BasicClassification, self).__init__()
        self.feature_module = feature_module
        self.input_tensor_names = input_tensor_names

    def feature_extractor(self, x: torch.Tensor) -> torch.Tensor:
        """
        特征提取器
        :param x:
        :return:
        """
        feature = self.feature_module.forward(x)
        return feature.squeeze()

    def forward(self, tensor_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplemented


class IDClassification(BasicClassification):
    """
    分类器
    """

    def __init__(self, feature_module: nn.Module,
                 in_features: int,
                 class_num: int,
                 bias: bool,
                 input_tensor_names: List[str]):
        """

        :param feature_module: 特征提取模块
        :param in_features: 输入特征维度
        :param class_num: 类别模块
        :param bias:
        :param input_tensor_names: 输入的tensor
        """
        super(IDClassification, self).__init__(feature_module=feature_module, input_tensor_names=input_tensor_names)
        self.class_num = class_num
        self.fc_layer = nn.Linear(in_features=in_features, out_features=self.class_num, bias=bias)

    def forward(self, tensor_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        feature = self.feature_extractor(tensor_dict["input_image"])
        out = self.fc_layer(feature)
        return {"out": out}


class IDClassificationFeature(BasicClassification):
    """
    用于 cross-entropy loss 和 center loss 的损失函数
    """

    def __init__(self, feature_module: nn.Module,
                 in_features: int,
                 class_num: int,
                 bias: bool,
                 input_tensor_names: List[str]):
        super(IDClassificationFeature, self).__init__(feature_module=feature_module,
                                                      input_tensor_names=input_tensor_names)
        self.class_num = class_num
        self.fc_layer = nn.Linear(in_features=in_features, out_features=self.class_num, bias=bias)

    def forward(self, tensor_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        feature = self.feature_extractor(tensor_dict["input_image"])
        out = self.fc_layer(feature)
        return {"out": out, "feature": feature}


class NormalizationIDClassificationFeature(BasicClassification):
    """
    用于 cross-entropy loss 和 triple loss 的损失函数
    """

    def __init__(self, feature_module: nn.Module,
                 in_features: int,
                 class_num: int,
                 bias: bool,
                 input_tensor_names: List[str]):
        super(NormalizationIDClassificationFeature, self).__init__(feature_module=feature_module,
                                                                   input_tensor_names=input_tensor_names)
        self.class_num = class_num
        self.fc_layer = nn.Linear(in_features=in_features, out_features=self.class_num, bias=bias)
        self.bn = nn.BatchNorm1d(in_features)

    def forward(self, tensor_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        feature = self.feature_extractor(tensor_dict["input_image"])
        out = self.fc_layer(self.bn(feature))
        return {"out": out, "feature": feature}


class TripleNormalizationIDClassificationFeature(BasicClassification):
    """
    带有正负样本的特征抽取 (用于后续的 triple loss)
    """

    def __init__(self, feature_module: nn.Module,
                 in_features: int,
                 class_num: int,
                 bias: bool,
                 input_tensor_names: List[str] = None):
        super(TripleNormalizationIDClassificationFeature, self).__init__(feature_module=feature_module,
                                                                         input_tensor_names=input_tensor_names)
        self.class_num = class_num
        self.fc_layer = nn.Linear(in_features=in_features, out_features=self.class_num, bias=bias)
        self.bn = nn.BatchNorm1d(in_features)

    def forward(self, tensor_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        #  负样本特征
        negative_feature = self.feature_extractor(x=tensor_dict["negative_image"])
        #  正样本特征
        positive_feature = self.feature_extractor(x=tensor_dict["positive_image"])
        #  目标样本特征
        feature = self.feature_extractor(x=tensor_dict["target_image"])
        out = self.fc_layer(self.bn(feature))
        return {"out": out,
                "feature": feature,
                "negative_feature": negative_feature,
                "positive_feature": positive_feature}
