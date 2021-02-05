"""
torch 的 keras 方式调用
@Author: zhangluoyang
@E-mail: 55058629@qq.com
@Time:  12月 17, 2020
"""

import os
import torch
import datetime
import numpy as np
import pandas as pd
from collections import OrderedDict
from prettytable import PrettyTable
from loss.loss_layer import BasicLoss
from typing import Dict, List, Union, Tuple
from metrics.metrics import BasicMetrics
from model.model import BasicClassification

__version__ = "1.5.3"

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

layer_modules = (torch.nn.MultiheadAttention,)


def to_device(tensor_dict: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    """
    将数据放置指定设备上面
    :param tensor_dict:
    :param device:
    :return:
    """
    for key in tensor_dict.keys():
        tensor_dict[key] = tensor_dict[key].to(device)
    return tensor_dict


def to_tensor_dict(tensors: List[torch.Tensor], names: List[str]) -> Dict[str, torch.Tensor]:
    assert len(tensors) == len(names)
    tensor_dict: Dict[str, torch.Tensor] = {}
    for i in range(len(tensors)):
        tensor_dict[names[i]] = tensors[i]
    return tensor_dict


def summary(model, input_shape, input_dtype=torch.FloatTensor, batch_size=-1,
            layer_modules=layer_modules, *args, **kwargs):
    # 这里仅仅打印特征提取层的信息
    model = model.net.feature_module

    def register_hook(module):
        def hook(module, inputs, outputs):

            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            key = "%s-%i" % (class_name, module_idx + 1)

            info = OrderedDict()
            info["id"] = id(module)
            if isinstance(outputs, (list, tuple)):
                try:
                    info["out"] = [batch_size] + list(outputs[0].size())[1:]
                except AttributeError:
                    # pack_padded_seq and pad_packed_seq store feature into data attribute
                    info["out"] = [batch_size] + list(outputs[0].data.size())[1:]
            else:
                info["out"] = [batch_size] + list(outputs.size())[1:]

            info["params_nt"], info["params"] = 0, 0
            for name, param in module.named_parameters():
                info["params"] += param.nelement() * param.requires_grad
                info["params_nt"] += param.nelement() * (not param.requires_grad)

            # if the current module is already-used, mark as "(recursive)"
            # check if this module has params
            if list(module.named_parameters()):
                for v in summary.values():
                    if info["id"] == v["id"]:
                        info["params"] = "(recursive)"

            summary[key] = info

        # ignore Sequential and ModuleList and other containers
        if isinstance(module, layer_modules) or not module._modules:
            hooks.append(module.register_forward_hook(hook))

    hooks = []
    summary = OrderedDict()

    model.apply(register_hook)

    # multiple inputs to the network
    if isinstance(input_shape, tuple):
        input_shape = [input_shape]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *size).type(input_dtype) for size in input_shape]
    # print(type(x[0]))

    try:
        with torch.no_grad():
            model(*x) if not (kwargs or args) else model(*x, *args, **kwargs)
    except Exception:
        # This can be usefull for debugging
        print("Failed to run torchkeras.summary...")
        raise
    finally:
        for hook in hooks:
            hook.remove()

    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # layer, output_shape, params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["out"]),
            "{0:,}".format(summary[layer]["params"] + summary[layer]["params_nt"])
        )
        total_params += (summary[layer]["params"] + summary[layer]["params_nt"])
        total_output += np.prod(summary[layer]["out"])
        trainable_params += summary[layer]["params"]
        print(line_new)

    # assume 4 bytes/number
    total_input_size = abs(np.prod(input_shape) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.6f" % total_input_size)
    print("Forward/backward pass size (MB): %0.6f" % total_output_size)
    print("Params size (MB): %0.6f" % total_params_size)
    print("Estimated Total Size (MB): %0.6f" % total_size)
    print("----------------------------------------------------------------")


class Model(torch.nn.Module):

    # print time bar...
    @staticmethod
    def print_bar():
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n" + "=" * 80 + "%s" % nowtime)

    def __init__(self, net: BasicClassification):
        super(Model, self).__init__()
        self.net = net

        self.history = {}
        self.loss_funcs: List[BasicLoss] = NotImplemented
        self.metrics_funcs: List[BasicMetrics] = NotImplemented
        self.device = NotImplemented
        self.early_stop = NotImplemented
        self.optimizer = NotImplemented
        self.tensor_names: List[str] = NotImplemented

    def forward(self, tensor_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.net:
            return self.net.forward(tensor_dict=tensor_dict)
        else:
            raise NotImplementedError

    def compile(self, loss_funcs: List[BasicLoss],
                metrics_funcs: List[BasicMetrics],
                tensor_names: List[str],
                optimizer=None,
                device=None,
                early_stop: int = 5):
        """

        :param loss_funcs:  损失函数列表
        :param metrics_funcs:  评估函数列表
        :param tensor_names: 数据迭代器每一次生成tensor对应的名称
        :param optimizer:  优化器
        :param device:  设备
        :param early_stop:  提前停止的条件
        :return:
        """
        self.loss_funcs = loss_funcs
        self.metrics_funcs = metrics_funcs
        self.tensor_names = tensor_names
        self.early_stop = early_stop
        self.optimizer = optimizer if optimizer else torch.optim.Adam(self.parameters(), lr=0.001)
        self.device = device if torch.cuda.is_available() else None
        if self.device:
            self.to(self.device)

    def summary(self, input_shape: Union[List[int], Tuple[int, int, int]], input_dtype=torch.FloatTensor,
                batch_size: int = -1):
        summary(self, input_shape, input_dtype, batch_size)

    def train_step(self, tensor_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """

        :param tensor_dict:
        :return:
        """
        self.train()
        self.optimizer.zero_grad()
        predictions = self.forward(tensor_dict)
        output_dict = dict(tensor_dict, **predictions)
        train_metrics = {}

        loss_list: List[torch.Tensor] = []
        for loss_func in self.loss_funcs:
            loss = loss_func(output_dict)
            train_metrics[loss_func.name] = loss.item()
            loss_list.append(loss)
        for metric_func in self.metrics_funcs:
            metrics = metric_func(output_dict)
            train_metrics[metric_func.name] = metrics.item()
        # 所有损失函数之和
        loss_sum = sum(loss_list)
        train_metrics["loss"] = loss_sum.item()
        loss_sum.backward()
        # update parameters
        self.optimizer.step()
        self.optimizer.zero_grad()

        return train_metrics

    @torch.no_grad()
    def evaluate_step(self, tensor_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        predictions = self.forward(tensor_dict)
        output_dict = dict(tensor_dict, **predictions)
        val_metrics = {}
        with torch.no_grad():
            loss_list: List[torch.Tensor] = []
            for loss_func in self.loss_funcs:
                loss = loss_func(output_dict)
                val_metrics[loss_func.name] = loss.item()
                loss_list.append(loss)
            for metric_func in self.metrics_funcs:
                metrics = metric_func(output_dict)
                val_metrics[metric_func.name] = metrics.item()
            # 所有损失函数之和
            loss_sum = sum(loss_list)
            val_metrics["loss"] = loss_sum.item()
        return val_metrics

    def fit(self, epochs, dl_train, dl_val=None, log_step_freq=1, check_path: str = None):

        print("Start Training ...")
        Model.print_bar()

        dl_val = dl_val if dl_val else []

        # 上一步验证集最小的index
        last_min_eval_loss = 1e10
        last_min_eval_index = 0
        for epoch in range(1, epochs + 1):
            # 1，training loop -------------------------------------------------
            train_metrics_sum, step = {}, 0
            for batch_data in dl_train:
                tensor_dict = to_tensor_dict(tensors=batch_data, names=self.tensor_names)
                tensor_dict = to_device(tensor_dict=tensor_dict, device=self.device)
                step = step + 1
                train_metrics = self.train_step(tensor_dict=tensor_dict)
                for name, metric in train_metrics.items():
                    train_metrics_sum[name] = train_metrics_sum.get(name, 0.0) + metric
                if step % log_step_freq == 0:
                    logs = {"step": step}
                    logs.update({k: round(v / step, 3) for k, v in train_metrics_sum.items()})
                    print(logs)
            for name, metric_sum in train_metrics_sum.items():
                self.history[name] = self.history.get(name, []) + [metric_sum / step]

            # 2，validate loop -------------------------------------------------
            val_metrics_sum, step = {}, 0
            for batch_data in dl_val:
                tensor_dict = to_tensor_dict(tensors=batch_data, names=self.tensor_names)
                tensor_dict = to_device(tensor_dict=tensor_dict, device=self.device)
                step = step + 1
                val_metrics = self.evaluate_step(tensor_dict=tensor_dict)
                for name, metric in val_metrics.items():
                    val_metrics_sum[name] = val_metrics_sum.get(name, 0.0) + metric
            for name, metric_sum in val_metrics_sum.items():
                self.history[name] = self.history.get(name, []) + [metric_sum / step]

            if check_path is not None:
                if not os.path.exists(check_path):
                    print("mask dir:{0}".format(check_path))
                    os.makedirs(check_path)
                torch.save(self.net.state_dict(), "{0}/epoch_{1}.pth".format(check_path, epoch))
                remove_path = "{0}/epoch_{1}.pth".format(check_path, epoch - self.early_stop - 1)
                if os.path.exists(remove_path):
                    os.remove(remove_path)
                    print("remove:{0}".format(remove_path))

            if self.early_stop is not None and dl_val is not None:
                epoch_eval_loss = val_metrics_sum["loss"] / step
                print("check early stop")
                print("epoch_eval_loss:{0}, epoch_eval_loss:{1}, last_min_eval_index:{2}, epoch:{3}".format(
                    epoch_eval_loss,
                    last_min_eval_loss,
                    last_min_eval_index,
                    epoch))
                # 连续early_stop 次结果没有提升 则退出
                if epoch_eval_loss > last_min_eval_loss and epoch - last_min_eval_index >= self.early_stop:
                    print("early stop ....")
                    break
                if epoch_eval_loss < last_min_eval_loss:
                    last_min_eval_loss = epoch_eval_loss
                    last_min_eval_index = epoch
            # 3，print logs -------------------------------------------------
            infos = {"epoch": epoch}
            infos.update({k: round(self.history[k][-1], 3) for k in self.history})
            tb = PrettyTable()
            tb.field_names = infos.keys()
            tb.add_row(infos.values())
            print("\n", tb)
            Model.print_bar()

        # 仅保留一个模型文件
        for epoch in range(1, epochs + 1):
            path = "{0}/epoch_{1}.pth".format(check_path, epoch)
            if epoch != last_min_eval_index:
                if os.path.exists(path):
                    os.remove(path)
            else:
                os.rename(path, "{0}/best.pth".format(check_path))
                print("Finished Training...")

        return pd.DataFrame(self.history)
