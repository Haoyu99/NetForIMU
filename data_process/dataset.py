from os import path as osp
import numpy as np
import random
from torch.utils.data import Dataset

from data_process.data_utils import load_cached_sequences


class RIDIDataset(Dataset):
    """
    CNN 网络模型
    输入200组数据 对应一个输出
    """
    def __init__(self, seq_type, root_dir, data_list, cache_path=None, step_size=10, window_size=200,
                 random_shift=0, transform=None, **kwargs):
        super(RIDIDataset, self).__init__()
        self.feature_dim = seq_type.feature_dim
        self.target_dim = seq_type.target_dim
        self.aux_dim = seq_type.aux_dim
        self.window_size = window_size
        self.step_size = step_size
        self.random_shift = random_shift
        self.transform = transform
        self.interval = kwargs.get('interval', window_size)

        self.data_path = [osp.join(root_dir, data) for data in data_list]
        self.index_map = []
        self.ts, self.orientations, self.gt_pos = [], [], []
        # 从缓存中加载数据
        self.features, self.targets, aux = load_cached_sequences(
            seq_type, root_dir, data_list, cache_path, interval=self.interval, **kwargs)
        # 把来自不同文件的数据加载到同一个list当中
        for i in range(len(data_list)):
            self.ts.append(aux[i][:, 0])
            self.orientations.append(aux[i][:, 1:5])
            self.gt_pos.append(aux[i][:, -2:])
            # 将target中的数据 切片[i,j] i代表文件id， j代表切片的索引值 每step_size切一次
            self.index_map += [[i, j] for j in range(0, self.targets[i].shape[0], step_size)]
        print('lode success')
        if kwargs.get('shuffle', True):
            random.shuffle(self.index_map)

    def __getitem__(self, item):
        seq_id, frame_id = self.index_map[item][0], self.index_map[item][1]
        # 默认 random_shift = 0
        if self.random_shift > 0:
            frame_id += random.randrange(-self.random_shift, self.random_shift)
            frame_id = max(self.window_size, min(frame_id, self.targets[seq_id].shape[0] - 1))
        # 就是说每200个feature 生成一个target
        feat = self.features[seq_id][frame_id:frame_id + self.window_size]
        targ = self.targets[seq_id][frame_id]

        if self.transform is not None:
            feat, targ = self.transform(feat, targ)

        return feat.astype(np.float32).T, targ.astype(np.float32), seq_id, frame_id

    def __len__(self):
        return len(self.index_map)
    def get_test_seq(self, i):
        return self.features[i].astype(np.float32), self.targets[i].astype(np.float32)




class SequenceToSequenceDataset(Dataset):
    """
    序列模型
    """
    def __init__(self, seq_type, root_dir, data_list, cache_path=None, step_size=100, window_size=200,
                 random_shift=0, transform=None, **kwargs):
        super(SequenceToSequenceDataset, self).__init__()
        self.seq_type = seq_type
        self.feature_dim = seq_type.feature_dim
        self.target_dim = seq_type.target_dim
        self.aux_dim = seq_type.aux_dim
        self.window_size = window_size
        self.step_size = step_size
        self.random_shift = random_shift
        self.transform = transform

        self.data_path = [osp.join(root_dir, data) for data in data_list]
        self.index_map = []

        self.features, self.targets, aux = load_cached_sequences(
            seq_type, root_dir, data_list, cache_path, interval=self.window_size, **kwargs)

        # Optionally smooth the sequence 平滑序列
        # feat_sigma = kwargs.get('feature_sigma,', -1)
        # targ_sigma = kwargs.get('target_sigma,', -1)
        # # 如果sigma大于0，则对其进行高斯滤波
        # if feat_sigma > 0:
        #     self.features = [gaussian_filter1d(feat, sigma=feat_sigma, axis=0) for feat in self.features]
        # if targ_sigma > 0:
        #     self.targets = [gaussian_filter1d(targ, sigma=targ_sigma, axis=0) for targ in self.targets]

        max_norm = kwargs.get('max_velocity_norm', 3.0)
        self.ts, self.orientations, self.gt_pos, self.local_v = [], [], [], []
        for i in range(len(data_list)):
            self.features[i] = self.features[i][:-1]
            self.targets[i] = self.targets[i]
            # aux
            self.ts.append(aux[i][:-1, :1])
            self.orientations.append(aux[i][:-1, 1:5])
            self.gt_pos.append(aux[i][:-1, -2:])
            # 计算一个合加速度
            velocity = np.linalg.norm(self.targets[i], axis=1)  # Remove outlier ground truth data
            # 移除异常点
            bad_data = velocity > max_norm
            # bad_data 是一个为维度和velocity一样的 布尔矩阵 大于max_norm的地方是ture 其余地方为false
            # j 从（400，数据总数，步长100）
            for j in range(window_size + random_shift, self.targets[i].shape[0], step_size):
                # any 又有一个为真 就为真  如果没有坏点 则放入索引map中
                if not bad_data[j - window_size - random_shift:j + random_shift].any():
                    self.index_map.append([i, j])

        if kwargs.get('shuffle', True):
            random.shuffle(self.index_map)

    def __getitem__(self, item):
        # output format: input, target, seq_id, frame_id
        seq_id, frame_id = self.index_map[item][0], self.index_map[item][1]
        if self.random_shift > 0:
            frame_id += random.randrange(-self.random_shift, self.random_shift)
            frame_id = max(self.window_size, min(frame_id, self.targets[seq_id].shape[0] - 1))
        # feat 的维度 ： 6 * 400
        feat = np.copy(self.features[seq_id][frame_id - self.window_size:frame_id])
        # target 的维度 ： 6 * 400
        targ = np.copy(self.targets[seq_id][frame_id - self.window_size:frame_id])


        if self.transform is not None:
            feat, targ = self.transform(feat, targ)

        return feat.astype(np.float32), targ.astype(np.float32), seq_id, frame_id

    def __len__(self):
        return len(self.index_map)

    def get_test_seq(self, i):
        return self.features[i].astype(np.float32)[np.newaxis,], self.targets[i].astype(np.float32)