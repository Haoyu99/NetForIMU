import math
from data_process.dataset import RIDIDataset
from utils import RandomHoriRotate


def get_dataset(args, **kwargs):
    """
    根据不同的需求加载数据
    :param args:
    :param kwargs:
    :return:
    """
    mode = kwargs.get('mode', 'train')
    random_shift, shuffle, transforms, grv_only = 0, False, None, False
    # 加载训练数据
    if mode == 'train':
        shuffle = True
        random_shift = args.step_size // 2
        transforms = RandomHoriRotate(math.pi * 2)
        with open(args.train_list) as f:
            data_list = [s.strip().split(',' or ' ')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']
            print(data_list)
    # 加载验证集数据
    elif mode == 'val':
        shuffle = True
        with open(args.val_list) as f:
            data_list = [s.strip().split(',' or ' ')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']
            print(data_list)
    # 加载测试数据
    else:
        shuffle = False
        with open(args.test_list) as f:
            data_list = [s.strip().split(',' or ' ')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']
            print(data_list)
    dataset = RIDIDataset(args.seq_type, args.data_dir, data_list, args.cache_path, args.step_size,
                                        args.window_size,
                                        random_shift=random_shift, transform=transforms, shuffle=shuffle,
                                        grv_only=grv_only, **kwargs)
    return dataset
