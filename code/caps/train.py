import argparse
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 当前 caps 文件夹
PARENT_DIR = os.path.dirname(BASE_DIR)                 # 上一级 code 文件夹
sys.path.append(BASE_DIR)
sys.path.append(PARENT_DIR)

import pandas
from torch.optim.lr_scheduler import StepLR
import pickle
import random
import sys
from typing import List
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils
from torch import Tensor
from torch.utils.data import DataLoader
from model import SetTransformer
from feature_env import FeatureEvaluator, base_path
from train_utils import AvgrageMeter, pairwise_accuracy, hamming_distance, FSDataset_set_tf
from record import SelectionRecord
from utils.logger import info, error
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--task_name', type=str, choices=['spectf', 'svmguide3', 'german_credit', 'uci_credit_card','spam_base','megawatt1','ionosphere',
                                                      'mice_protein', 'coil-20', 'minist_fashion', 'urbansound8k',
                                                      'openml_589', 'openml_616', 'IQdataset'], default='spam_base')

parser.add_argument('--gpu', type=int, default=0, help='used gpu')
parser.add_argument('--fe', type=str, choices=['+', '', '-'], default='-')
parser.add_argument('--gen_num', type=int, default=25)
parser.add_argument('--l2_reg', type=float, default=0)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--set_tf_arch', type=str, choices=['SAB','ISAB'],default='ISAB')
parser.add_argument('--set_tf_hidden_size', type=int,default=128)
# parser.add_argument('--grad_bound', type=float, default=6.0)
args = parser.parse_args()

# 创建一个专门用来计算神经网络有多大、包含多少个参数的小工具。
def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

# 把带有占位符（eos）的浓缩特征索引，清理掉废料后，还原成了最原始、最直白的一长串 0 和 1（选或不选）
def choice_to_onehot(choice, eos):
    new_choice = []
    eos_batches = choice.data.eq(eos)
    eos_batches = ~eos_batches.cpu()

    for i in range(eos_batches.shape[0]):
        onehot = torch.zeros(eos).long()
        feat_seq = choice[i][eos_batches[i]]
        onehot[feat_seq] = 1
        new_choice.append(onehot)
    return torch.stack(new_choice, dim=0)


def set_tf_train(train_queue, model, optimizer,eos):
    ce = AvgrageMeter()
    acc_ = AvgrageMeter()
    f1_ = AvgrageMeter()
    model.train()
    for step, sample in enumerate(train_queue):
        encoder_input = sample['input'].cuda(model.gpu).float()
        optimizer.zero_grad()

        eos_batches = encoder_input.data.eq(eos)
        eos_batches = ~eos_batches.cpu().view(-1)
        label = encoder_input.view(-1)[eos_batches].long()

        feat_emd, output = model(encoder_input)
        valid_output = output[eos_batches]
        loss = F.cross_entropy(valid_output, label) # ce loss
        loss.backward()
        optimizer.step()


        pred = torch.max(valid_output, -1)[1].view(-1)
        acc = accuracy_score(label.cpu().numpy(), pred.cpu().numpy())
        f1 = f1_score(label.cpu().numpy(), pred.cpu().numpy(), average='macro')

        n = encoder_input.size(0)
        ce.update(loss.data, n)
        ce.update(loss.data, n)
        f1_.update(f1, n)
        acc_.update(acc, n)
    return ce.avg, acc_.avg, f1_.avg

def set_tf_valid(queue, model, eos):
    ce = AvgrageMeter()
    acc_ = AvgrageMeter()
    f1_ = AvgrageMeter()
    with torch.no_grad():
        model.eval()
        for step, sample in enumerate(queue):
            encoder_input = sample['input'].cuda(model.gpu).float()


            eos_batches = encoder_input.data.eq(eos)
            eos_batches = ~eos_batches.cpu().view(-1)
            label = encoder_input.view(-1)[eos_batches].long()

            feat_emd, output = model(encoder_input)
            valid_output = output[eos_batches]
            loss = F.cross_entropy(valid_output, encoder_input.view(-1)[eos_batches].long())

            pred = torch.max(valid_output, -1)[1].view(-1)
            acc = accuracy_score(label.cpu().numpy(), pred.cpu().numpy())
            f1 = f1_score(label.cpu().numpy(), pred.cpu().numpy(), average='macro')

            n = encoder_input.size(0)
            ce.update(loss.data, n)
            # ce.update(loss.data, n)
            f1_.update(f1, n)
            acc_.update(acc, n)
            #
        return ce.avg, acc_.avg, f1_.avg




def main():
    if not torch.cuda.is_available():
        info('No GPU found!')
        sys.exit(1)
    # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpu)
    # 全局随机种子（Seed）对齐
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # 配置 cuDNN（NVIDIA 深度神经网络加速库）的底层计算模式。
    cudnn.enabled = True
    cudnn.benchmark = False
    cudnn.deterministic = True
    device = int(args.gpu)
    # 将解析到的所有命令行参数（配置项）打印/记录到日志中，方便日后查看某次实验是在什么参数设置下跑出来的。
    info(f"Args = {args}")

    pkl_dir = f'{base_path}/history/{args.task_name}'
    pkl_path = f'{pkl_dir}/new_fe.pkl'
    
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            fe: FeatureEvaluator = pickle.load(f)
    else:
        info(f"new_fe.pkl 丢失，正在为 {args.task_name} 自动生成...")
        os.makedirs(pkl_dir, exist_ok=True)
        fe = FeatureEvaluator(args.task_name)
        
        # ================= 新增：自动收集初始热身数据 =================
        info("【数据初始化】发现历史记录为空，开始自动随机做题以积累历史数据... (请耐心等待)")
        fe.report_ds() # 先把所有特征都选上的情况测一遍，作为基准
        
        # 随机探索 200 次组合，作为训练 Set Transformer 的初始原材料
        for i in range(200): 
            # 随机生成一个 0和1 组成的列表，代表选或不选某个特征
            mask = torch.randint(0, 2, (fe.ds_size,)).float()
            # 防止一个特征都不选导致报错，强行选一个
            if mask.sum() == 0: 
                mask[random.randint(0, int(fe.ds_size)-1)] = 1.0 
                
            # 丢给下游分类器去打分，存入记录 (rp=False 不打印每一步防止刷屏)
            fe.report_performance(mask, store=True, rp=False)
            
            if (i+1) % 50 == 0:
                info(f"已自动评估并记录 {i+1}/200 个特征组合...")
        # ==============================================================
        
        with open(pkl_path, 'wb') as f:
            pickle.dump(fe, f)
        info("自动生成与初始数据收集彻底完毕！")

    # model = SetTransformer_for_large_ds(fe.ds_size,fe.ds_size,fe.ds_size,args)
    model = SetTransformer(fe.ds_size,fe.ds_size,fe.ds_size,args)
    info(f"param size = {count_parameters_in_MB(model)}MB")
    model = model.cuda(device)

    choice, labels = fe.get_record(args.gen_num, eos=fe.ds_size)
    valid_choice, valid_labels = fe.get_record(0, eos=fe.ds_size)

    info('Training Encoder-Decoder ==> Set Transformer')

    min_val = min(labels)
    max_val = max(labels)
    train_encoder_target = [(i - min_val) / (max_val - min_val) for i in labels]
    valid_encoder_target = [(i - min_val) / (max_val - min_val) for i in valid_labels]

    train_dataset = FSDataset_set_tf(choice, train_encoder_target, train=True, sos_id=fe.ds_size, eos_id=fe.ds_size)
    valid_dataset = FSDataset_set_tf(valid_choice, valid_encoder_target, train=False, sos_id=fe.ds_size, eos_id=fe.ds_size)
    train_queue = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    valid_queue = torch.utils.data.DataLoader(
        valid_dataset, batch_size=len(valid_dataset), shuffle=False, pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)
    losses = []

    model_save_path = f'{base_path}/history/{fe.task_name}/set_tf/'
    os.makedirs(model_save_path, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # loss = set_tf_train(train_queue, model, optimizer,eos=fe.ds_size)
        loss, f1, acc = set_tf_train(train_queue, model, optimizer,eos=fe.ds_size)
        losses.append(loss.cpu().detach().numpy())
        if epoch % 10 == 0 or epoch == 1:
            info("epoch {:04d} train loss {:.6f} accuracy {:.6f} f1_score {:.6f}".format(epoch, loss, acc, f1))
        if epoch % 100 == 0 or epoch == 1:
            valid_train_loss, valid_train_f1, valid_train_acc = set_tf_valid(valid_queue, model,eos=fe.ds_size)
            info("Evaluation on train data")
            info("epoch {:04d} valid_train loss {:.6f} accuracy {:.6f} f1_score {:.6f}".format(epoch, valid_train_loss, valid_train_acc, valid_train_f1))
            valid_loss, valid_f1, valid_acc = set_tf_valid(valid_queue, model,eos=fe.ds_size)
            info("Evaluation on valid data")
            info("epoch {:04d} valid loss {:.6f} accuracy {:.6f} f1_score {:.6f}".format(epoch, valid_loss, valid_acc, valid_f1))
            torch.save(model.state_dict(), os.path.join(model_save_path, f'set_tf_epoch{epoch}_{valid_loss}.model_dict'))

    plt.plot(range(1, args.epochs + 1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')

    save_dir = f'{base_path}/history/{fe.task_name}/set_tf/'
    os.makedirs(save_dir, exist_ok=True)

    current_time = datetime.now().strftime('%Y%m%d%H%M%S')
    save_name = f"{current_time}_{args.task_name}_{fe.ds_size}_lr{args.lr}_{valid_loss}.png"
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path)
    plt.close()
    print(f'{fe.task_name} done!')


if __name__ == '__main__':
    main()


