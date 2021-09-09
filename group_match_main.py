"""
功能区匹配训练
@Author: zhangluoyang
@E-mail: 55058629@qq.com
@Time:  1月 27, 2021
"""
import torch
from PIL import Image
from typing import List
import torchvision.transforms as T
from metrics.metrics import BasicMetrics, Accuracy
from torch.utils.data import DataLoader
from model.keras.torch_keras import Model
from data.LmdbDataSet import TripleDataSet
from model.model import TripleNormalizationIDClassificationFeature
from model.feature.resnet import res_net_50
from loss.loss_layer import CrossEntropyLoss, BatchHardTripletLoss, BasicLoss, LabelSmoothingCrossEntropy


def run(train_data_dir: str, test_data_dir: str):
    """

    :param train_data_dir: 训练集
    :param test_data_dir: 测试集
    :return:
    """
    # 数据集
    transform = T.Compose([T.RandomHorizontalFlip(),
                           T.RandomCrop((800, 800)),
                           T.Resize((224, 224), interpolation=Image.BICUBIC),
                           T.ToTensor(),
                           T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_db_path = "{0}/data.db".format(train_data_dir)
    train_keys_path = "{0}/keys.npy".format(train_data_dir)
    train_zone_index_path = "{0}/zone_index.json".format(train_data_dir)
    train_data_set = TripleDataSet(db_path=train_db_path, keys_path=train_keys_path,
                                   zone_index_path=train_zone_index_path, transform=transform)

    test_db_path = "{0}/data.db".format(test_data_dir)
    test_keys_path = "{0}/keys.npy".format(test_data_dir)
    test_zone_index_path = "{0}/zone_index.json".format(test_data_dir)
    test_data_set = TripleDataSet(db_path=test_db_path, keys_path=test_keys_path,
                                  zone_index_path=test_zone_index_path, transform=transform)

    # 类别分类样本数目
    class_num = 22
    # 特征提取层的特征维度
    in_features = 2048
    batch_size = 20
    num_workers = 0
    epochs: int = 200
    check_path: str = "./check_point"
    log_step_freq: int = 1
    device: str = "cuda:0"
    # dataSet 输出的tensor 名称
    tensor_names: List[str] = ["target_image",
                               "positive_image",
                               "negative_image",
                               "target",
                               "positive_weight",
                               "negative_weight"]
    # 损失函数 (分类的损失函数适当小一些 类目太多)
    # classification_loss = CrossEntropyLoss(tensor_name="out", target_tensor_name="target", eof=0.2)
    classification_loss = LabelSmoothingCrossEntropy(tensor_name="out", target_tensor_name="target", eof=0.2, alpha=0.1)
    triple_loss = BatchHardTripletLoss(target_feature_tensor_name="feature",
                                       positive_feature_tensor_name="positive_feature",
                                       positive_weight_tensor_name="positive_weight",
                                       negative_feature_tensor_name="negative_feature",
                                       negative_weight_tensor_name="negative_weight",
                                       eof=1.0)
    loss_funcs: List[BasicLoss] = [classification_loss, triple_loss]
    # 评估函数
    metrics_funcs: List[BasicMetrics] = [Accuracy(tensor_names=["out", "target"])]
    # 特征提取器
    feature_net = res_net_50()
    # 模型
    triple_net = TripleNormalizationIDClassificationFeature(feature_module=feature_net,
                                                            in_features=in_features,
                                                            class_num=class_num,
                                                            bias=False)

    # 优化器
    opt_args = {"lr": 0.001, "weight_decay": 0.0001, "momentum": 0.9}
    model = Model(net=triple_net)
    optimizer = torch.optim.SGD(model.parameters(), nesterov=True, **opt_args)

    train_data_loader = DataLoader(train_data_set, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    test_data_loader = DataLoader(test_data_set, shuffle=False, batch_size=batch_size, num_workers=num_workers)
    model.summary(input_shape=(3, 224, 224))
    model.compile(loss_funcs=loss_funcs, metrics_funcs=metrics_funcs, tensor_names=tensor_names,
                  optimizer=optimizer,
                  device=device,
                  early_stop=10)
    model.fit(epochs=epochs,
              dl_train=train_data_loader,
              dl_val=test_data_loader,
              log_step_freq=log_step_freq,
              check_path=check_path)


if __name__ == "__main__":
    train_dir = r"G:/group_image/train"
    test_dir = r"G:/group_image/test"
    run(train_data_dir=train_dir, test_data_dir=test_dir)
