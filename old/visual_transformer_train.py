# """
# @Author: zhangluoyang
# @E-mail: 55058629@qq.com
# @Time:  12月 22, 2020
# """
# import torch
# from PIL import Image
# import torch.nn as nn
# from torchvision import datasets
# import torchvision.transforms as T
# from torch.utils.data import DataLoader
# from model.deprecation.visual_transformer import vit_base_patch16_224
# from metrics.metrics import Accuracy
# from model.keras.torch_keras import KerasModel
#
# if __name__ == "__main__":
#     data_path = r"./data_set"
#     batch_size = 40
#     transform = T.Compose([T.Resize((256, 256), interpolation=Image.BICUBIC),
#                            T.RandomCrop(224),
#                            T.RandomHorizontalFlip(),
#                            T.ToTensor()])
#     train_data_set = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
#     test_data_set = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
#     train_data_loader = DataLoader(train_data_set, shuffle=True, batch_size=batch_size, num_workers=0)
#     test_data_loader = DataLoader(test_data_set, shuffle=True, batch_size=batch_size, num_workers=0)
#     visual_transformer = vit_base_patch16_224(pretrained=True, num_classes=10)
#     criterion = nn.CrossEntropyLoss()
#     metrics_dict = {"acc": Accuracy()}
#     model = KerasModel(visual_transformer)
#     model.summary(input_shape=(3, 224, 224))
#     opt_args = {"lr": 0.01, "weight_decay": 0.0001, "momentum": 0.9}
#     optimizer = torch.optim.SGD(model.parameters(), momentum=opt_args["momentum"], nesterov=True, **opt_args)
#     # scheduler = Scheduler(optimizer=optimizer, ) todo
#     model.compile(loss_func=criterion, metrics_dict=metrics_dict, device="cuda:2", optimizer=optimizer)
#     model.fit(epochs=1000, dl_train=train_data_loader, dl_val=test_data_loader, log_step_freq=10)
