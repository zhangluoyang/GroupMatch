"""
@Author: zhangluoyang
@E-mail: 55058629@qq.com
@Time:  12月 24, 2020
"""
import torch
from PIL import Image
from model.keras.torch_keras import Model
import torchvision.transforms as T
from torch.utils.data import DataLoader
from model.deprecation.mobilenet import MobileNetV3
from data.LmdbDataSet import MultiDataSetImagePath

if __name__ == "__main__":
    batch_size = 64
    epochs = 100
    transform = T.Compose([T.Resize((256, 256), interpolation=Image.BICUBIC),
                           T.RandomCrop(224),
                           T.ToTensor()])
    # train_data_set = MultiDataSetImagePath(r"/mnt/zhangluoyang/taobao/train", transform)
    train_data_set = MultiDataSetImagePath(r"F:/taobao/train", transform)
    train_data_loader = DataLoader(train_data_set, shuffle=True, batch_size=batch_size, num_workers=8, pin_memory=True)
    # test_data_set = MultiDataSetImagePath(r"/mnt/zhangluoyang/taobao/test", transform)
    test_data_set = MultiDataSetImagePath(r"F:/taobao/test", transform)
    test_data_loader = DataLoader(test_data_set, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True)

    class_num_01 = len(train_data_set.first_name_to_id_dict)
    class_num_02 = len(train_data_set.second_name_to_id_dict)
    mobile_net = MobileNetV3(n_class=[class_num_01, class_num_02])
    model = Model(mobile_net)
    model.summary(input_shape=(3, 224, 224))
    opt_args = {"lr": 0.01, "weight_decay": 0.0001, "momentum": 0.9}
    optimizer = torch.optim.SGD(model.parameters(), nesterov=True, **opt_args)
    # model.compile(loss_func=[nn.CrossEntropyLoss(), nn.CrossEntropyLoss()],
    #               index=[0, 1],
    #               cofs=[1.0, 0.0],
    #               metrics_dict={"acc": Accuracy()},
    #               device="cuda:2",
    #               optimizer=optimizer)
    # model.fit(epochs=epochs, dl_train=train_data_loader, dl_val=test_data_loader, log_step_freq=1,
    #           check_path="./checkpoint/{0}".format(time.time()))
