"""
常用的损失层
@Author: zhangluoyang
@E-mail: 55058629@qq.com
@Time:  11月 18, 2020
"""
import math
from typing import List, Union, Dict, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F


class BasicLoss(nn.Module):
    """
    损失函数
    """

    def __init__(self, tensor_name: Union[str, None],
                 target_tensor_name: Union[str, None],
                 name: str,
                 eof: float = 1.0):
        """
        :param tensor_name: tensor的名称
        :param target_tensor_name: 目标 tensor的名称
        :param name: 损失的名称
        :param eof: 损失的权重
        """
        super(BasicLoss, self).__init__()
        self.tensor_name = tensor_name
        self.target_tensor_name = target_tensor_name
        self.name = name
        self.eof = eof

    def forward(self, tensor_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplemented


class CrossEntropyLoss(BasicLoss):
    """
    交叉熵代价函数
    """

    def __init__(self, tensor_name: Union[str, None],
                 target_tensor_name: Union[str, None],
                 name: str = "cross_loss",
                 eof: float = 1.0):
        super(CrossEntropyLoss, self).__init__(tensor_name=tensor_name, target_tensor_name=target_tensor_name,
                                               name=name, eof=eof)

    def forward(self, tensor_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.cross_entropy(tensor_dict[self.tensor_name], tensor_dict[self.target_tensor_name])


class LabelSmoothingCrossEntropy(BasicLoss):
    """
    带有标签平滑的损失函数
    y = (1-alpha) * y + (1/c)
    """

    def __init__(self, tensor_name: Union[str, None],
                 target_tensor_name: Union[str, None],
                 name: str = "smooth_label",
                 eof: float = 1.0,
                 alpha: float = 0.1):
        super(LabelSmoothingCrossEntropy, self).__init__(tensor_name=tensor_name,
                                                         target_tensor_name=target_tensor_name,
                                                         name=name, eof=eof)
        self.alpha = alpha

    def forward(self, tensor_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 类别数目
        c = tensor_dict[self.tensor_name].size()[-1]
        log_preds = F.log_softmax(tensor_dict[self.tensor_name], dim=-1)
        loss = -log_preds.sum(dim=-1)
        loss = loss.mean()
        return loss * self.alpha / c + (1 - self.alpha) * \
               F.nll_loss(log_preds, tensor_dict[self.target_tensor_name], reduction="mean")


class CenterLoss(BasicLoss):
    """
        Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    """

    def __init__(self, tensor_name: str,
                 eof: float,
                 class_num: int,
                 target_tensor_name: str,
                 name: str = "center_loss",
                 feat_dim: int = 64):
        """
        :param tensor_name: tensor的名称
        :param eof: 损失的权重
        :param class_num: 类别数目
        :param target_tensor_name 目标tensor名称
        :param feat_dim: 类别中心点
        """
        super(CenterLoss, self).__init__(tensor_name=tensor_name, eof=eof,
                                         name=name, target_tensor_name=target_tensor_name)
        self.class_num = class_num
        self.feat_dim = feat_dim

        self.centers = nn.Parameter(torch.Tensor(class_num, feat_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.centers, a=math.sqrt(5))

    def forward(self, tensor_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """

        :param tensor_dict:
        :return:
        """
        x = tensor_dict[self.tensor_name]
        labels = tensor_dict[self.tensor_name]
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())
        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        return loss * self.eof


class BatchHardTripletLoss(BasicLoss):
    """
    Combination of Multiple Global Descriptors for Image Retrieval
    一个匹次里的样本 使得 target 与 正负样本的距离保持
    """

    def __init__(self, eof: float,
                 target_feature_tensor_name: str,
                 positive_feature_tensor_name: str,
                 positive_weight_tensor_name: str,
                 negative_feature_tensor_name: str,
                 negative_weight_tensor_name: str,
                 name: str = "triple_loss",
                 margin: float = 1.0):
        """
        :param eof
        :param target_feature_tensor_name 目标 tensor 名称
        :param positive_feature_tensor_name 正样本 tensor 名称
        :param positive_weight_tensor_name 正样本权重 tensor 名称
        :param negative_feature_tensor_name 负样本 tensor 名称
        :param negative_weight_tensor_name 负样本权重 tensor 名称
        :param margin:  正负样本的特征间隔
        """
        super(BatchHardTripletLoss, self).__init__(tensor_name=None, eof=eof, target_tensor_name=None, name=name)
        self.target_feature_tensor_name = target_feature_tensor_name
        self.positive_feature_tensor_name = positive_feature_tensor_name
        self.positive_weight_tensor_name = positive_weight_tensor_name
        self.negative_feature_tensor_name = negative_feature_tensor_name
        self.negative_weight_tensor_name = negative_weight_tensor_name
        self.margin = margin

    def forward(self, tensor_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        这里仅规定仅有一个正样本 仅有一个负样本 (尽量要求 positive_num不为1) 这样可以使得模型收敛加快
        这里需要保证 正负样本数目相同
        :return:
        """
        # todo 特征是否需要进行归一化处理 ?
        target_feature = tensor_dict[self.target_feature_tensor_name]
        positive_feature = tensor_dict[self.positive_feature_tensor_name]
        # 正样本与目标样本相似的权重
        positive_weight = tensor_dict[self.positive_weight_tensor_name]
        negative_feature = tensor_dict[self.negative_feature_tensor_name]
        # 负样本与目标样本不相似的权重 等价于 (1-相似度)
        negative_weight = tensor_dict[self.negative_weight_tensor_name]
        # [batch_num, 1, feature_dim]
        target_feature = target_feature.unsqueeze(1)
        # 计算距离正样本的距离
        # [batch_num, positive_num, feature_dim]
        positive_distance = torch.pow((target_feature - positive_feature), exponent=2)
        # [batch_num, positive_num]
        positive_distance = torch.sum(positive_distance, dim=-1)
        positive_distance = positive_weight * positive_distance
        # 计算距离负样本的距离
        # [batch_num, negative_num, feature_dim]
        negative_distance = torch.pow((target_feature - negative_feature), exponent=2)
        # [batch_num, negative_sum]
        negative_distance = torch.sum(negative_distance, dim=-1)
        negative_distance = negative_weight * negative_distance
        # 保证正负样本有一定的间隔
        # [batch_num, num]
        loss = F.relu(positive_distance - negative_distance + self.margin)
        return torch.mean(torch.mean(loss, dim=1), dim=0)


class FocalLoss(BasicLoss):
    """
    Focal loss for Dense Object Detection
    解决1 样本分布不均匀问题
    解决2 简单易学的样本权重少一些 非简单难以学习的样本权重多一些
    """

    def __init__(self,
                 class_num: int,
                 tensor_name: str,
                 target_tensor_name: str,
                 alpha: Union[None, List[float]],
                 name: str = "focal_loss",
                 eof: float = 1.0,
                 gamma: float = 2.0):
        """
        :param class_num: 类别数目
        :param alpha: 控制样本的分布不均匀参数
        :param gamma: 控制简单易学样本的权重参数

        """
        super(FocalLoss, self).__init__(tensor_name=tensor_name, eof=eof, target_tensor_name=target_tensor_name,
                                        name=name)
        if alpha is None:
            # 每一个类别相同
            self.alpha = nn.Parameter(torch.ones(class_num, 1), requires_grad=False)
        else:
            assert class_num == len(alpha)
            self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=False)

        self.class_num = class_num
        self.gamma = gamma

    def forward(self, tensor_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        :return:
        """
        inputs = tensor_dict[self.tensor_name]
        targets = tensor_dict[self.target_tensor_name]
        # [batch_size, class_num]
        class_mask = torch.zeros_like(inputs)
        # [batch_size, 1]
        ids = targets.view(-1, 1)
        # [batch_size, class_num] (one-hot)
        class_mask.scatter_(1, ids.data, 1.0)
        # [batch_size, 1] 每一个样本对应类别的权重
        alpha = self.alpha[targets]
        # [batch_size, class_num]
        p = torch.softmax(inputs, dim=-1)
        # [batch_size, 1]
        prob_s = (p * class_mask).sum(1).view(-1, 1)
        log_p = prob_s.log()
        batch_loss = -alpha * torch.pow(1 - prob_s, self.gamma) * log_p
        return batch_loss.mean()
