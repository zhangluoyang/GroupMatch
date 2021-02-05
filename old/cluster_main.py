"""
像素点聚类
@Author: zhangluoyang
@E-mail: 55058629@qq.com
@Time:  12月 17, 2020
"""
import numpy as np
from torch.utils.data import DataLoader
from data.DataSet import ImageDataFolder
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans

if __name__ == "__main__":
    data_path = r"F:/data/all/test"
    save_path = r"./centroids.npy"
    data_set = ImageDataFolder(data_path=data_path)
    batch_size = 8
    num_clusters = 512
    max_iter = 8
    mini_batch_k_mean_s = MiniBatchKMeans(n_clusters=num_clusters, random_state=0)
    for i in range(max_iter):
        data_loader = DataLoader(data_set, shuffle=True, batch_size=batch_size, num_workers=0)
        print("cluster index:{0}".format(i))
        for images, labels in tqdm(tqdm(data_loader)):
            images = images.tolist()
            images = np.transpose(a=images, axes=(0, 2, 3, 1))
            pixels = images.reshape(-1, images.shape[-1])
            mini_batch_k_mean_s.fit(pixels)
            mini_batch_k_mean_s.partial_fit(pixels)
    np.save(save_path, mini_batch_k_mean_s.cluster_centers_)
