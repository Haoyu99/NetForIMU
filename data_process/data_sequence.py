from abc import ABC, abstractmethod
from os import path as osp
import numpy as np
import pandas
from quaternion import quaternion

# 数据sequence从csv中加载 转换为对应的list
class DataSequence(ABC):
    """
    An abstract interface for compiled sequence.
    一个抽象类 用于处理数据
    """

    def __init__(self, **kwargs):
        super(DataSequence, self).__init__()

    @abstractmethod
    def load(self, path):
        pass

    @abstractmethod
    def get_feature(self):
        pass

    @abstractmethod
    def get_target(self):
        pass

    @abstractmethod
    def get_aux(self):
        pass


class RIDIRawDataSequence(DataSequence):
    """
    DataSet: RIDI数据集
    Features : 三轴的加速度，三轴的陀螺仪
    target: 时间窗位移
    """
    feature_dim = 6
    target_dim = 2
    # aux 数据是 时间 四元数 以及真实位置
    aux_dim = 7


    def __init__(self, data_path, **kwargs):
        super().__init__(**kwargs)
        self.ts, self.features, self.targets, self.orientations, self.gt_pos = None, None, None, None, None
        # w是一个窗口值
        self.w = kwargs.get('interval', 1)
        self.info = {}

        if data_path is not None:
            self.load(data_path)

    def load(self, path):
        """
        从指定的path中加载csv文件
        :param path:
        :return:
        """
        if path[-1] == '/':
            path = path[:-1]

        self.info['path'] = osp.split(path)[-1]
        print(path)

        if osp.exists(osp.join(path, 'processed/data.csv')):
            imu_all = pandas.read_csv(osp.join(path, 'processed/data.csv'))
        else:
            print('fail to load, data.csv is not exist')

        ts = imu_all[['time']].values / 1e09  # 时间值变为以秒为单位
        gyro = imu_all[['gyro_x', 'gyro_y', 'gyro_z']].values
        acce = imu_all[['acce_x', 'acce_y', 'acce_z']].values
        pos = imu_all[['pos_x', 'pos_y']].values
        quat = quaternion.from_float_array(imu_all[['rv_w', 'rv_x', 'rv_y', 'rv_z']].values)
        self.ts = ts
        self.features = np.concatenate([gyro, acce], axis=1)
        # 计算每个时间窗的的位移
        self.targets = pos[self.w:, :] - pos[:-self.w, :]
        self.gt_pos = pos
        self.orientations = quaternion.as_float_array(quat)
        print(self.ts.shape, self.features.shape, self.targets.shape, self.gt_pos.shape, self.orientations.shape,
              self.w)

    def get_feature(self):
        return self.features

    def get_target(self):
        return self.targets

    def get_aux(self):
        return np.concatenate([self.ts, self.orientations, self.gt_pos], axis=1)


class RIDIGlobalDataSequence(DataSequence):
    """
       DataSet: RIDI数据集
       Features : 经过旋转的 三轴的加速度，三轴的陀螺仪
       target: 时间窗位移
       """
    feature_dim = 6
    target_dim = 2
    # aux 数据是 时间 四元数 以及真实位置
    aux_dim = 7

    def __init__(self, data_path, **kwargs):
        super().__init__(**kwargs)
        self.ts, self.features, self.targets, self.orientations, self.gt_pos = None, None, None, None, None
        # w是一个窗口值
        self.w = kwargs.get('interval', 1)
        self.info = {}

        if data_path is not None:
            self.load(data_path)

    def load(self, path):
        """
        从指定的path中加载csv文件
        :param path:
        :return:
        """
        if path[-1] == '/':
            path = path[:-1]

        self.info['path'] = osp.split(path)[-1]
        print(path)

        if osp.exists(osp.join(path, 'processed/data.csv')):
            imu_all = pandas.read_csv(osp.join(path, 'processed/data.csv'))
        else:
            print('fail to load, data.csv is not exist')

        ts = imu_all[['time']].values / 1e09  # 时间值变为以秒为单位
        gyro = imu_all[['gyro_x', 'gyro_y', 'gyro_z']].values
        acce = imu_all[['acce_x', 'acce_y', 'acce_z']].values
        pos = imu_all[['pos_x', 'pos_y']].values

        # Use game rotation vector as device orientation.
        init_tango_ori = quaternion.quaternion(*imu_all[['ori_w', 'ori_x', 'ori_y', 'ori_z']].values[0])
        game_rv = quaternion.from_float_array(imu_all[['rv_w', 'rv_x', 'rv_y', 'rv_z']].values)
        #
        init_rotor = init_tango_ori * game_rv[0].conj()
        ori = init_rotor * game_rv
        #
        nz = np.zeros(ts.shape)
        gyro_q = quaternion.from_float_array(np.concatenate([nz, gyro], axis=1))
        acce_q = quaternion.from_float_array(np.concatenate([nz, acce], axis=1))

        gyro_glob = quaternion.as_float_array(ori * gyro_q * ori.conj())[:, 1:]
        acce_glob = quaternion.as_float_array(ori * acce_q * ori.conj())[:, 1:]

        self.ts = ts
        self.features = np.concatenate([gyro_glob, acce_glob], axis=1)
        # 计算每个时间窗的的位移
        self.targets = pos[self.w:, :] - pos[:-self.w, :]
        self.gt_pos = pos
        self.orientations = quaternion.as_float_array(game_rv)
        print(self.ts.shape, self.features.shape, self.targets.shape, self.gt_pos.shape, self.orientations.shape,
              self.w)

    def get_feature(self):
        return self.features

    def get_target(self):
        return self.targets

    def get_aux(self):
        return np.concatenate([self.ts, self.orientations, self.gt_pos], axis=1)