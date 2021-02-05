"""
torch 常用的评估函数
@Author: zhangluoyang
@E-mail: 55058629@qq.com
@Time:  12月 22, 2020
"""
import torch
from typing import Dict, List
from metrics.utils import check_same_shape
from metrics.utils import convert_to_tensor


class BasicMetrics(object):

    def __init__(self, tensor_names: List[str], name: str):
        self.tensor_names = tensor_names
        self.name = name

    def __call__(self, tensor_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplemented


class Accuracy(BasicMetrics):
    def __init__(self, tensor_names: List[str], name: str = "acc", threshold: float = 0.5):
        super(Accuracy, self).__init__(tensor_names=tensor_names, name=name)
        self.threshold = threshold

    def __call__(self, tensor_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """

        :param tensor_dict
        :return:
        """
        logits = tensor_dict[self.tensor_names[0]]
        y_pred = torch.argmax(logits, dim=1)
        y_true = tensor_dict[self.tensor_names[1]]
        return torch.mean((y_pred == y_true).float())

# class Precision:
#     """
#     Computes precision of the predictions with respect to the true labels.
#     Args:
#         y_true: Tensor of Ground truth values.
#         y_pred: Tensor of Predicted values.
#         epsilon: Fuzz factor to avoid division by zero. default: `1e-10`
#     Returns:
#         Tensor of precision score
#     """
#
#     def __init__(self, epsilon=1e-10):
#         self.epsilon = epsilon
#
#     def __call__(self, y_pred, y_true):
#         true_positives = torch.sum(torch.round(torch.clamp(y_pred * y_true, 0, 1)))
#         predicted_positives = torch.sum(torch.round(torch.clamp(y_pred, 0, 1)))
#         precision = true_positives / (predicted_positives + self.epsilon)
#         return precision
#
#
# class Recall:
#     """
#     Computes recall of the predictions with respect to the true labels.
#     Args:
#         y_true: Tensor of Ground truth values.
#         y_pred: Tensor of Predicted values.
#         epsilon: Fuzz factor to avoid division by zero. default: `1e-10`
#     Returns:
#         Tensor of recall score
#     """
#
#     def __init__(self, epsilon=1e-10):
#         self.epsilon = epsilon
#
#     def __call__(self, y_pred, y_true):
#         true_positives = torch.sum(torch.round(torch.clamp(y_pred * y_true, 0, 1)))
#         actual_positives = torch.sum(torch.round(torch.clamp(y_true, 0, 1)))
#         recall = true_positives / (actual_positives + self.epsilon)
#         return recall
#
#
# class DSC:
#     def __init__(self):
#         self.smooth = 1.
#
#     def __call__(self, tensor1, tensor2):
#         iflat = tensor1.view(-1)
#         tflat = tensor2.view(-1)
#         intersection = (iflat * tflat).sum()
#
#         return 1 - ((2. * intersection + self.smooth) /
#                     (iflat.sum() + tflat.sum() + self.smooth))
#
#
# class HingeMetric:
#     """
#         Arguments
#         ---------
#         pred : torch.Tensor
#         ground_truth : torch.Tensor [-1 or 1]
#         """
#
#     def __call__(self, tensor1, tensor2):
#         if 0.0 in torch.unique(tensor2):
#             tensor2[tensor2 == 0.0] = -1.0
#         hinge_loss = 1 - torch.mul(tensor1, tensor2)
#         hinge_loss[hinge_loss < 0] = 0
#         return torch.mean(hinge_loss)
#
#
# class SquareHingeMetric:
#     def __call__(self, tensor1, tensor2):
#         """
#         Arguments
#         ---------
#         pred : torch.Tensor
#         ground_truth : torch.Tensor [-1 or 1]
#         """
#         if 0. in torch.unique(tensor2):
#             tensor2[tensor2 == 0.] = -1.
#         hinge_loss = 1 - torch.mul(tensor1, tensor2)
#         hinge_loss[hinge_loss < 0] = 0
#         return torch.mean(hinge_loss ** 2)
#
#
# class KLDivergence:
#     """
#     Computes Kullback-Leibler divergence metric between `y_true` and `y_pred`.
#     Args:
#         y_true: Tensor of Ground truth values. shape = [batch_size, d0, .., dN]
#         y_pred: Tensor of Predicted values. shape = [batch_size, d0, .., dN]
#         epsilon: Fuzz factor to avoid division by zero. default: `1e-10`
#     Returns:
#         Tensor of Kullback-Leibler divergence metric
#     """
#
#     def __init__(self, epsilon: float = 1e-10):
#         self.epsilon = epsilon
#
#     def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
#         y_pred = convert_to_tensor(y_pred)
#         y_true = convert_to_tensor(y_true)
#
#         check_same_shape(y_pred, y_true)
#
#         y_pred = torch.clamp(y_pred, self.epsilon, 1)
#         y_true = torch.clamp(y_true, self.epsilon, 1)
#         kld = torch.sum(y_true * torch.log(y_true / y_pred), axis=-1)
#         return torch.mean(kld)
#
#
# class F1Score:
#     """
#     Computes F1-score between `y_true` and `y_pred`.
#     Args:
#         y_true: Tensor of Ground truth values.
#         y_pred: Tensor of Predicted values.
#         epsilon: Fuzz factor to avoid division by zero. default: `1e-10`
#     Returns:
#         Tensor of F1-score
#     """
#
#     def __init__(self, epsilon=1e-10):
#         self.epsilon = epsilon
#         self.precision = Precision()
#         self.recall = Recall()
#
#     def __call__(self, y_pred, y_true):
#         precision = self.precision(y_pred, y_true)
#         recall = self.recall(y_pred, y_true)
#         return 2 * ((precision * recall) / (precision + recall + self.epsilon))
#
#
# class Huber:
#     """
#     Computes the huber loss between `y_true` and `y_pred`.
#     Args:
#         y_true: Tensor of Ground truth values. shape = [batch_size, d0, .., dN]
#         y_pred: Tensor of Predicted values. shape = [batch_size, d0, .., dN]
#         delta: A float, the point where the Huber loss function changes from a
#                 quadratic to linear. default: `1.0`
#     Returns:
#         Tensor of Huber loss
#     """
#
#     def __call__(
#             self, y_pred: torch.Tensor, y_true: torch.Tensor, delta: float = 1.0
#     ) -> torch.Tensor:
#         y_pred = convert_to_tensor(y_pred)
#         y_true = convert_to_tensor(y_true)
#
#         check_same_shape(y_pred, y_true)
#
#         abs_error = torch.abs(y_pred - y_true)
#         quadratic = torch.clamp(abs_error, max=delta)
#         linear = abs_error - quadratic
#         loss = 0.5 * quadratic.pow(2) + delta * linear
#         return loss.mean()
#
#
# class LogCoshError:
#     """
#     Computes Logarithm of the hyperbolic cosine of the prediction error.
#     Args:
#         y_true: Tensor of Ground truth values. shape = [batch_size, d0, .., dN]
#         y_pred: Tensor of Predicted values. shape = [batch_size, d0, .., dN]
#     Returns:
#         Tensor of Logcosh error
#     """
#
#     def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
#         y_pred = convert_to_tensor(y_pred)
#         y_true = convert_to_tensor(y_true)
#
#         check_same_shape(y_pred, y_true)
#
#         diff = y_pred - y_true
#         return torch.mean(torch.log((torch.exp(diff) + torch.exp(-1.0 * diff)) / 2.0))
#
#
# class MeanAbsoluteError:
#     def __call__(self, tensor1, tensor2):
#         """
#         Arguments
#         ---------
#         x : torch.Tensor
#         y : torch.Tensor
#         """
#         return torch.mean(torch.abs(tensor1 - tensor2))
#
#
# class MeanIoU:
#     def __init__(self):
#         self.epsilon = 1e-10
#
#     def __call__(self, tensor1, tensor2):
#         # if single dimension
#         if len(tensor1.shape) == 1 and len(tensor2.shape) == 1:
#             inter = torch.sum(torch.squeeze(tensor1 * tensor2))
#             union = torch.sum(torch.squeeze(tensor1 + tensor2)) - inter
#         else:
#             inter = torch.sum(
#                 torch.sum(torch.squeeze(tensor1 * tensor2, axis=3), axis=2), axis=1
#             )
#             union = (
#                     torch.sum(
#                         torch.sum(torch.squeeze(tensor1 + tensor2, axis=3), axis=2), axis=1
#                     )
#                     - inter
#             )
#         return torch.mean((inter + self.epsilon) / (union + self.epsilon))
#
#
# class MeanSquaredError:
#     def __call__(self, tensor1, tensor2):
#         """
#         Arguments
#         ---------
#         x : torch.Tensor
#         y : torch.Tensor
#         """
#         return torch.mean((tensor1 - tensor2) ** 2)
#
#
# class MeanSquaredLogarithmicError:
#     """
#     Computes the mean squared logarithmic error between `y_true` and `y_pred`.
#     Args:
#         y_true: Tensor of Ground truth values. shape = [batch_size, d0, .., dN]
#         y_pred: Tensor of Predicted values. shape = [batch_size, d0, .., dN]
#     Returns:
#         Tensor of mean squared logarithmic error
#     """
#
#     def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
#         y_pred = convert_to_tensor(y_pred)
#         y_true = convert_to_tensor(y_true)
#
#         check_same_shape(y_pred, y_true)
#
#         squared_log = torch.pow(torch.log1p(y_pred) - torch.log1p(y_true), 2)
#
#         return torch.mean(squared_log)
#
#
# class RootMeanSquaredError:
#     def __call__(self, tensor1, tensor2):
#         """
#         Returns the root mean squared error (RMSE) of two tensors.
#         Arguments
#         ---------
#         x : torch.Tensor
#         y : torch.Tensor
#         """
#         return torch.sqrt(torch.mean((tensor1 - tensor2) ** 2))
#
#
# class RSquared:
#     def corrcoef(self, tensor1, tensor2):
#         """
#         Arguments
#         ---------
#         x : torch.Tensor
#         y : torch.Tensor
#         """
#         xm = tensor1.sub(torch.mean(tensor1))
#         ym = tensor2.sub(torch.mean(tensor2))
#         r_num = xm.dot(ym)
#         r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
#         r_val = r_num / r_den
#         return r_val
#
#     def __call__(self, tensor1, tensor2):
#         """
#         Arguments
#         ---------
#         x : torch.Tensor
#         y : torch.Tensor
#         """
#         return (self.corrcoef(tensor1, tensor2)) ** 2
