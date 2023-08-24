import json
import os
import time
from os import path as osp
from pathlib import Path
from shutil import copyfile

import numpy as np
import torch
from scipy.interpolate import interp1d
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from data_process.data_sequence import RIDIRawDataSequence, RIDIGlobalDataSequence

from get_dataset import get_dataset

# 定义训练中需要用到的参数
from get_model import get_CNN_model


class GetArgs(dict):
    def __init__(self, *args, **kwargs):
        super(GetArgs, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            value = GetArgs(value)
        return value


info = {'type': 'lstm_bi',
        # 数据源
        'data_dir': '/home/jiamingjie/zhanghaoyu/data',
        # 训练数据list
        'train_list': '/home/jiamingjie/zhanghaoyu/data/train_handle.txt',
        # 验证数据list
        'val_list': '/home/jiamingjie/zhanghaoyu/data/val_handle.txt',
        # 测试数据list
        'test_list': '/home/jiamingjie/zhanghaoyu/data/test_list2.txt',
        # 数据生成的缓存地址 第一次加载从csv-> hd5 之后加载缓存
        'cache_path': '/home/jiamingjie/zhanghaoyu/data//cache',
        # 测试时加载的模型地址
        'model_path': '/home/jiamingjie/zhanghaoyu/datacache/handle_out/checkpoints/checkpoint_best.pt',
        'feature_sigma': 0.001,
        'target_sigma': 0.0,
        # 200个数据算一个切片 6*200 输入
        'window_size': 200,
        # 打乱顺序的范围
        'step_size': 10,
        'batch_size': 64,
        'num_workers': 1,
        # 输出地址
        'out_dir': '/home/jiamingjie/zhanghaoyu/datacache/handle_out/',
        'device': 'cuda:1',
        'dataset': 'ridi',
        'layers': 3,
        'layer_size': 200,
        'epochs': 300,
        'save_interval': 20,
        'lr': 0.01,
        'mode': 'train',
        'continue_from': None,
        'fast_test': False,
        'show_plot': True,
        'seq_type': RIDIGlobalDataSequence,
        }
args = GetArgs(info)


# 写出配置信息
def write_config(args, **kwargs):
    if args.out_dir:
        with open(osp.join(args.out_dir, 'config.json'), 'w') as f:
            values = vars(args)
            values['file'] = "pytorch_global_position"
            if kwargs:
                values['kwargs'] = kwargs
            json.dump(values, f, sort_keys=True)


def get_loss_function():
    criterion = torch.nn.MSELoss()
    return criterion


def format_string(*argv, sep=' '):
    result = ''
    for val in argv:
        if isinstance(val, (tuple, list, np.ndarray)):
            for v in val:
                result += format_string(v, sep=sep) + sep
        else:
            result += str(val) + sep
    return result[:-1]


def train(args, **kwargs):
    # 加载训练数据和验证集数据
    start_t = time.time()
    train_dataset = get_dataset(args, mode='train', **kwargs)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,
                              drop_last=True, pin_memory=True)
    end_t = time.time()
    print('训练集获取成功. 用时: {:.3f}s'.format(end_t - start_t))
    val_dataset, val_loader = None, None
    if args.val_list is not None:
        val_dataset = get_dataset(args, mode='val', **kwargs)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        print('验证集加载成功')
    # 训练数据的设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print('训练设备' + device.type)
    summary_writer = None
    # 输出文件的地址
    if args.out_dir:
        if not osp.isdir(args.out_dir):
            os.makedirs(args.out_dir)
        if not osp.isdir(osp.join(args.out_dir, 'checkpoints')):
            os.makedirs(osp.join(args.out_dir, 'checkpoints'))
        if not osp.isdir(osp.join(args.out_dir, 'logs')):
            os.makedirs(osp.join(args.out_dir, 'logs'))
        copyfile(args.train_list, osp.join(args.out_dir, "train_list.txt"))
        if args.val_list is not None:
            copyfile(args.val_list, osp.join(args.out_dir, "validation_list.txt"))
        write_config(args, **kwargs)

    print('训练集Dataset切片个数: {}'.format(len(train_dataset)))
    train_mini_batches = len(train_loader)
    # 验证集
    if val_dataset:
        print('验证集Dataset切片个数: {}'.format(len(val_dataset)))
        val_mini_batches = len(val_loader)

    network = get_CNN_model("resnet101").to(device)
    # criterion = get_loss_function()
    # loss 计算方式 使用均方根误差
    criterion = torch.nn.MSELoss()
    # 使用Adam优化器
    optimizer = torch.optim.Adam(network.parameters(), args.lr)
    # lr自动更新
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1, verbose=True, eps=1e-12)

    # 输出txt 日志
    log_file = None
    if args.out_dir:
        log_file = osp.join(args.out_dir, 'logs', 'log.txt')
        if osp.exists(log_file):
            if args.continue_from is None:
                os.remove(log_file)
            else:
                copyfile(log_file, osp.join(args.out_dir, 'logs', 'log_old.txt'))

    start_epoch = 0
    if args.out_dir is not None and osp.exists(osp.join(args.out_dir, 'logs')):
        summary_writer = SummaryWriter(osp.join(args.out_dir, 'logs'))

    step = 0
    best_val_loss = np.inf
    train_errs = np.zeros(args.epochs)

    print("Starting from epoch {}".format(start_epoch))
    train_losses_all, val_losses_all = [], []

    # Get the initial loss.  获取初始的loss
    init_train_targ, init_train_pred = run_test(network, train_loader, device, eval_mode=False)
    init_train_loss = np.mean((init_train_targ - init_train_pred) ** 2, axis=0)
    train_losses_all.append(np.mean(init_train_loss))
    print('-------------------------')
    print('Init: average loss: {}/{:.6f}'.format(init_train_loss, train_losses_all[-1]))
    if summary_writer is not None:
        add_summary(summary_writer, init_train_loss, 0, 'train')

    if val_loader is not None:
        init_val_targ, init_val_pred = run_test(network, val_loader, device)
        init_val_loss = np.mean(init_val_targ - init_val_pred ** 2, axis=0)
        val_losses_all.append(np.mean(init_val_loss))
        print('Validation loss: {}/{:.6f}'.format(init_val_loss, val_losses_all[-1]))
        if summary_writer is not None:
            add_summary(summary_writer, init_val_loss, 0, 'val')

    try:
        # 开始正式训练
        for epoch in range(start_epoch, args.epochs):
            log_line = ''
            network.train()
            # MSEAverageMeter的作用是计算均值
            # train_vel = MSEAverageMeter(3, [2], _output_channel)
            train_outs, train_targets = [], []
            start_t = time.time()
            for bid, batch in enumerate(train_loader):
                # bid 指切片号 batch中包含数据
                # 每次获取的feat的规格是[batch_size * Windows_size * input_dim]
                feat, targ, _, _ = batch
                feat, targ = feat.to(device), targ.to(device)
                # 梯度清0
                optimizer.zero_grad()
                # 进行预测
                predicted = network(feat)
                train_outs.append(predicted.cpu().detach().numpy())
                train_targets.append(targ.cpu().detach().numpy())
                # train_vel.add(predicted.cpu().detach().numpy(), targ.cpu().detach().numpy())
                # 计算损失值，通常使用交叉熵、均方误差等损失函数，这里是均方根
                loss = criterion(predicted, targ)
                loss = torch.mean(loss)
                # 对损失值进行反向传播，计算参数的梯度
                loss.backward()
                # 使用优化器（optimizer）更新模型的参数
                optimizer.step()
                step += 1
            # 计算每个batch的loss
            train_outs = np.concatenate(train_outs, axis=0)
            train_targets = np.concatenate(train_targets, axis=0)
            train_losses = np.average((train_outs - train_targets) ** 2, axis=0)
            end_t = time.time()
            print('-------------------------')
            print('Epoch {}, time usage: {:.3f}s, average loss: {:.6f}, lr : {}'.format(
                epoch, end_t - start_t, np.average(train_losses), optimizer.param_groups[0]['lr']))
            train_losses_all.append(np.average(train_losses))

            if summary_writer is not None:
                add_summary(summary_writer, train_losses, epoch + 1, 'train')
                summary_writer.add_scalar('optimizer/lr', optimizer.param_groups[0]['lr'], epoch)
            saved_model = False
            # 验证集
            if val_loader is not None:
                network.eval()
                val_outs, val_targets = run_test(network, val_loader, device)
                val_losses = np.average((val_outs - val_targets) ** 2, axis=0)
                avg_loss = np.average(val_losses)
                # print('Validation loss: {}/{:.6f}'.format(val_losses, avg_loss))
                print('Validation loss: {:.6f}'.format(avg_loss))
                scheduler.step(avg_loss)
                if summary_writer is not None:
                    add_summary(summary_writer, val_losses, epoch + 1, 'val')
                val_losses_all.append(avg_loss)
                if avg_loss < best_val_loss:
                    best_val_loss = avg_loss
                    if args.out_dir and osp.isdir(args.out_dir):
                        model_path = osp.join(args.out_dir, 'checkpoints', 'checkpoint_best.pt')
                        torch.save({'model_state_dict': network.state_dict(),
                                        'epoch': epoch,
                                        'optimizer_state_dict': optimizer.state_dict()}, model_path)
                        print('Model saved to ', model_path)

            if args.out_dir and not saved_model and (epoch + 1) % args.save_interval == 0:  # save even with validation
                model_path = osp.join(args.out_dir, 'checkpoints', 'icheckpoint_%d.pt' % epoch)
                torch.save({'model_state_dict': network.state_dict(),
                            'epoch': epoch,
                            'loss': train_errs[epoch],
                            'optimizer_state_dict': optimizer.state_dict()}, model_path)
                print('Model saved to ', model_path)
    except KeyboardInterrupt:
        print('-' * 60)
        print('Early terminate')

    print('Training completed')
    if args.out_dir:
        model_path = osp.join(args.out_dir, 'checkpoints', 'checkpoint_latest.pt')
        torch.save({'model_state_dict': network.state_dict(),
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict()}, model_path)



def run_test(network, data_loader, device, eval_mode=True):
    targets_all = []
    preds_all = []
    if eval_mode:
        network.eval()
    # 使用初始化的网络先跑一次 记录预测结果
    for bid, (feat, targ, _, _) in enumerate(data_loader):
        pred = network(feat.to(device)).cpu().detach().numpy()
        targets_all.append(targ.detach().numpy())
        preds_all.append(pred)
    targets_all = np.concatenate(targets_all, axis=0)
    preds_all = np.concatenate(preds_all, axis=0)
    return targets_all, preds_all


def add_summary(writer, loss, step, mode):
    names = '{0}_loss/loss_x,{0}_loss/loss_y,{0}_loss/loss_z,{0}_loss/loss_sin,{0}_loss/loss_cos'.format(
        mode).split(',')
    for i in range(loss.shape[0]):
        writer.add_scalar(names[i], loss[i], step)
    writer.add_scalar('{}_loss/avg'.format(mode), np.mean(loss), step)


# 获取根据速度获取全局的位置
def recon_traj_with_preds_global(dataset, preds, seq_id=0, type='preds', **kwargs):

    if type == 'gt':
        pos = dataset.gt_pos[seq_id]
    else:
        start_pos = dataset.gt_pos[seq_id][0]
        preds[0] = start_pos
        pos = np.cumsum(preds,axis=0)
        # ts = dataset.ts[seq_id]
        # # Compute the global velocity from local velocity.
        # dts = np.mean(ts[ind[1:]] - ts[ind[:-1]])
        # pos = preds * dts
        # pos[0, :] = dataset.gt_pos[seq_id][0, :]
        # pos = np.cumsum(pos, axis=0)
    # veloc = preds
    # ori = dataset.orientations[seq_id]

    return pos

def recon_traj_with_preds(dataset, preds, seq_id=0, **kwargs):
    """
    Reconstruct trajectory with predicted global velocities.
    """
    ts = dataset.ts[seq_id]
    ind = np.array([i[1] for i in dataset.index_map if i[0] == seq_id], dtype=np.int)
    dts = np.mean(ts[ind[1:]] - ts[ind[:-1]])
    pos = np.zeros([preds.shape[0] + 2, 2])
    pos[0] = dataset.gt_pos[seq_id][0, :2]
    pos[1:-1] = np.cumsum(preds[:, :2] * dts, axis=0) + pos[0]
    pos[-1] = pos[-2]
    ts_ext = np.concatenate([[ts[0] - 1e-06], ts[ind], [ts[-1] + 1e-06]], axis=0)
    pos = interp1d(ts_ext, pos, axis=0)(ts)
    return pos


# 测试
def test(args, **kwargs):
    global device, _output_channel
    import matplotlib.pyplot as plt
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        checkpoint = torch.load(args.model_path, map_location=lambda storage, location: storage)
    else:
        checkpoint = torch.load(args.model_path, map_location=args.device)
    network = get_CNN_model()(args, **kwargs)
    print(network)
    network.load_state_dict(checkpoint.get('model_state_dict'))
    network.eval().to(device)
    print('Model {} loaded to device {}.'.format(args.model_path, device))
    preds_seq, targets_seq, losses_seq, ate_all, rte_all = [], [], [], [], []
    traj_lens = []
    seq_dataset = get_dataset(args, mode='test', **kwargs)
    seq_loader = DataLoader(seq_dataset, 1024, num_workers=args.num_workers, shuffle=False)
    ind = np.array([i[1] for i in seq_dataset.index_map if i[0] == 0], dtype=np.int)
    targets, preds = run_test(network, seq_loader, device, True)
    print(targets.shape)
    print(preds.shape)
    losses = np.mean((targets - preds) ** 2, axis=0)
    preds_seq.append(preds)
    targets_seq.append(targets)
    losses_seq.append(losses)
    pos_pred = recon_traj_with_preds(seq_dataset, preds)[:, :2]
    print(pos_pred.shape)
    pos_gt = seq_dataset.gt_pos[0][:, :2]

    traj_lens.append(np.sum(np.linalg.norm(pos_gt[1:] - pos_gt[:-1], axis=1)))
    # ate, rte = compute_ate_rte(pos_pred, pos_gt, pred_per_min)
    # ate_all.append(ate)
    # rte_all.append(rte)
    # pos_cum_error = np.linalg.norm(pos_pred - pos_gt, axis=1)

    # print('Sequence {}, loss {} / {}, ate {:.6f}, rte {:.6f}'.format(data, losses, np.mean(losses), ate, rte))

    train_dataset = get_dataset(args, mode='test', **kwargs)
    print(train_dataset.index_map)
    feat, tar = train_dataset.get_test_seq(0)
    # print(feat.shape)
    # print(tar.shape)
    # feat = torch.Tensor(feat).to(device)
    # preds = np.squeeze(network(feat).cpu().detach().numpy())
    # new_data = []
    # for i in range(0, preds.shape[0], 200):
    #     new_data.append(preds[i:i + 200].mean(axis=0))
    # new_data = np.array(new_data)
    # print(new_data.shape)


    # ind = np.arange(tar.shape[0])
    # pos_pred = recon_traj_with_preds_global(train_dataset, new_data,  type='pred', seq_id=0)
    pos_gt = recon_traj_with_preds_global(train_dataset, tar, seq_id=0,type='gt')
    print(pos_pred)
    #
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
    ax1.plot(pos_gt[:, 0], pos_gt[:, 1],label = 'gt')
    ax1.plot(pos_pred[:, 0], pos_pred[:, 1],label = 'pred')
    ax1.legend()
    plt.show()



if __name__ == '__main__':
    train(args,use_scheduler = True)
    # test(args)
