import argparse
import os
import sys

# 1. 稳妥的绝对路径注入（防止找不到 feature_env）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(PARENT_DIR)

import pandas
import pickle
import random
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils
from torch import Tensor
from torch.utils.data import DataLoader

# 2. 正确导入本地模块（这里明确定义了 SetTransformer 和 PPO）
from model import SetTransformer, PPO
from feature_env import FeatureEvaluator, base_path
from train_utils import FSDataset_set_tf
from record import SelectionRecord
from utils.logger import info, error

# 3. 修复作者改错的文件名
from train import main as train_set_TF
from train import args as train_set_TF_args

# --- 下面保留原有的 parser = argparse.ArgumentParser() 不动 ---
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--new_gen', type=int, default=200)
parser.add_argument('--task_name', type=str, choices=['spectf', 'svmguide3', 'german_credit', 'spam_base',
                                                      'ionosphere', 'megawatt1', 'uci_credit_card', 'openml_618',
                                                      'openml_589', 'openml_616', 'openml_607', 'openml_620',
                                                      'openml_637',
                                                      'openml_586', 'uci_credit_card', 'higgs', 'ap_omentum_ovary','activity'
                                                      , 'mice_protein', 'coil-20', 'isolet', 'minist', 'minist_fashion'], default='german_credit')

parser.add_argument('--gpu', type=int, default=0, help='used gpu')
parser.add_argument('--fe', type=str, choices=['+', '', '-'], default='-')
parser.add_argument('--top_k', type=int, default=100)
parser.add_argument('--gen_num', type=int, default=25)
parser.add_argument('--max_step_size', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=512)

parser.add_argument('--lr_actor', type=float, default=0.0003)
parser.add_argument('--lr_critic', type=float, default=0.001)
parser.add_argument('--eps_clip', type=float, default=0.2)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--search_step', type=int, default=1000)
parser.add_argument('--ppo_hidden_size', type=int, default=128)
# 迭代次数
parser.add_argument('--epoch', type=int, default=1)
parser.add_argument('--set_tf_arch', type=str, choices=['SAB','ISAB'],default='ISAB')
parser.add_argument('--set_tf_hidden_size', type=int,default=128)

parser.add_argument('--reward_tradeoff', type=float, default=0.1)
args = parser.parse_args()


baseline_name = [
    'kbest',
    'mrmr',
    'lasso',
    'rfe',
    'lassonet',
    'gfs',
    'sarlfs',
    'marlfs',
    'rra',
    'mcdm',
    'gains'
]

def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

def choice_to_onehot(choice, eos):
    new_choice = []
    eos_batches = choice.data.eq(eos)
    eos_batches = ~eos_batches.cpu()

    for i in range(eos_batches.shape[0]):
        onehot = torch.zeros(eos)
        feat_seq = choice[i][eos_batches[i]]
        onehot[feat_seq] = 1
        new_choice.append(onehot)
    return torch.stack(new_choice, dim=0)

def output_to_state(output, valid_input, eos):
    eos_batches = valid_input.data.eq(eos)
    output[eos_batches] = eos
    new_state = choice_to_onehot(output, eos)
    return new_state

def get_reward(state, fe, epoch, epoches):
    reward_list = torch.empty(state.size(0), 1)
    print(f'{epoch}/{epoches} trajectroy collecting....')
    for i in range(state.shape[0]):
        # 【修改点1：彻底杜绝泄题】把 flag='test' 改为 flag='train'
        reward = fe.report_performance(state[i].numpy(),rp=False, flag='train')
        reward_list[i] = reward
    print(f'{epoch}/{epoches}trajectroy collected!!!')
    return reward_list

def ppo_search(queue,ppo,set_tf,feat_len, epoches, fe, reward_weight):
    ppo.train()
    set_tf.eval()
    for epoch in range(epoches):
        for i, sample in enumerate(queue):
            encoder_input = sample['input']
            performance = sample['target']

            # valid feature index sequence to one-hot vector
            state = choice_to_onehot(encoder_input,feat_len)
            # select action with policy
            action = ppo.select_action(state) # size: [batch_size, 1, hidden_embedding_size]

            with torch.no_grad():
                feat_emb, _ = set_tf(encoder_input.cuda(set_tf.gpu).float())
                new_feat_emb = action + feat_emb.squeeze()
                new_output = set_tf.infer(new_feat_emb.unsqueeze(1))

            new_state = output_to_state(new_output, encoder_input, feat_len)
            new_state_reward = get_reward(new_state, fe, epoch, epoches)
            reward = reward_weight * (new_state_reward - performance) + (1 - reward_weight) * performance

            ppo.buffer.rewards.append(reward)

        ppo.update()

        new_selection = []
        new_choice = []
        predict_step_size = 0

        while len(new_selection) < args.new_gen:
            predict_step_size += 1
            info('Generate new architectures with step size {:d}'.format(predict_step_size))
            new_record = generate_new_records(queue, ppo, set_tf, fe.ds_size)
            for choice in new_record:
                if choice.sum() <= 0:
                    error('insufficient selection')
                    continue
                record = SelectionRecord(choice.numpy(), -1)
                if record not in fe.records.r_list and record not in new_selection:
                    new_selection.append(record)
                    new_choice.append(choice)
                if len(new_selection) >= args.new_gen:
                    break
            info(f'{len(new_selection)} new choice generated now', )
            if predict_step_size > args.max_step_size:
                break
        info(f'build {len(new_selection)} new choice !!!')

        new_choice_pt = torch.stack(new_choice)

        # 【修改点2：规矩挑人】只允许在 80% 的训练集里挑最好的特征
        best_selection_train = None
        best_optimal_train = -1000
        for s in new_selection:
            train_data = fe.generate_data(s.operation, 'train')
            train_result = fe.get_performance(train_data)
            if train_result > best_optimal_train:
                best_selection_train = s.operation
                best_optimal_train = train_result
                info(f'found best on train : {best_optimal_train}')

        # ================== 【修改点3：真正的闭卷大考（全任务类型支持版）】 ==================
        save_path = f'{base_path}/history/{fe.task_name}/ppo/{args.ppo_hidden_size}_{args.eps_clip}_{args.reward_tradeoff}_final'
        os.makedirs(save_path, exist_ok=True)
        opt_path_test = os.path.join(save_path, 'best-ppo-results.hdf')
        report_txt_path = os.path.join(save_path, '真正泛化成绩单.txt')

        # 1. 剪裁出只包含黄金特征的 80% 训练卷 和 20% 测试卷
        train_data = fe.generate_data(best_selection_train, 'train')
        test_data  = fe.generate_data(best_selection_train, 'test')

        X_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
        X_test,  y_test  = test_data.iloc[:, :-1],  test_data.iloc[:, -1]

        # 2. 打开成绩单准备写入，自动识别任务类型
        with open(report_txt_path, 'w', encoding='utf-8') as f:
            f.write(f"【{args.task_name}】 ({fe.task_type} 任务) 真正泛化战报：\n")
            f.write(f"----------------------------------------\n")

            # ================== 二分类任务 (Classification) ==================
            if fe.task_type == 'cls':
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
                
                clf = RandomForestClassifier(random_state=0, n_jobs=-1)
                clf.fit(X_train, y_train)
                y_predict = clf.predict(X_test)

                real_pre = precision_score(y_test, y_predict, average='weighted', zero_division=0)
                real_rec = recall_score(y_test, y_predict, average='weighted', zero_division=0)
                real_f1  = f1_score(y_test, y_predict, average='weighted', zero_division=0)
                try: real_auc = roc_auc_score(y_test, y_predict, average='weighted')
                except ValueError: real_auc = 0.0

                f.write(f"Precision : {real_pre * 100:.2f}%\n")
                f.write(f"Recall    : {real_rec * 100:.2f}%\n")
                f.write(f"F1-Score  : {real_f1 * 100:.2f}%\n")
                f.write(f"ROC/AUC   : {real_auc * 100:.2f}%\n")

            # ================== 多分类任务 (Multi-Classification) ==================
            elif fe.task_type == 'mcls':
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.multiclass import OneVsRestClassifier
                from sklearn.metrics import precision_score, recall_score, f1_score
                
                # 多分类必须套一层 OneVsRest 壳子
                clf = OneVsRestClassifier(RandomForestClassifier(random_state=0, n_jobs=-1))
                clf.fit(X_train, y_train)
                y_predict = clf.predict(X_test)

                real_pre  = precision_score(y_test, y_predict, average='macro', zero_division=0)
                real_rec  = recall_score(y_test, y_predict, average='macro', zero_division=0)
                real_mif1 = f1_score(y_test, y_predict, average='micro', zero_division=0)
                real_maf1 = f1_score(y_test, y_predict, average='macro', zero_division=0)

                f.write(f"Precision : {real_pre * 100:.2f}%\n")
                f.write(f"Recall    : {real_rec * 100:.2f}%\n")
                f.write(f"Micro-F1  : {real_mif1 * 100:.2f}%\n")
                f.write(f"Macro-F1  : {real_maf1 * 100:.2f}%\n")

            # ================== 回归任务 (Regression) ==================
            elif fe.task_type == 'reg':
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.metrics import mean_absolute_error, mean_squared_error
                from utils.tools import relative_absolute_error
                
                # 回归任务不能用分类器，必须用回归器
                reg = RandomForestRegressor(random_state=0, n_jobs=-1)
                reg.fit(X_train, y_train)
                y_predict = reg.predict(X_test)

                # 按作者原版规则，算的是 1 - 误差（所以数值越高越好）
                real_mae  = 1 - mean_absolute_error(y_test, y_predict)
                real_mse  = 1 - mean_squared_error(y_test, y_predict, squared=True)
                real_rae  = 1 - relative_absolute_error(y_test, y_predict)
                real_rmse = 1 - mean_squared_error(y_test, y_predict, squared=False)

                f.write(f"1-MAE     : {real_mae:.4f}\n")
                f.write(f"1-MSE     : {real_mse:.4f}\n")
                f.write(f"1-RAE     : {real_rae:.4f}\n")
                f.write(f"1-RMSE    : {real_rmse:.4f}\n")

            # ================== 检测任务 (Detection) ==================
            elif fe.task_type == 'det':
                from sklearn.neighbors import KNeighborsClassifier
                from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, recall_score
                
                # 检测任务作者底层指定了 KNN 模型
                knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
                knn.fit(X_train, y_train)
                y_predict = knn.predict(X_test)

                real_map = average_precision_score(y_test, y_predict)
                real_f1  = f1_score(y_test, y_predict, average='weighted', zero_division=0)
                try: real_ras = roc_auc_score(y_test, y_predict)
                except ValueError: real_ras = 0.0
                real_rec = recall_score(y_test, y_predict, average='weighted', zero_division=0)

                f.write(f"MAP Score : {real_map * 100:.2f}%\n")
                f.write(f"F1-Score  : {real_f1 * 100:.2f}%\n")
                f.write(f"ROC/AUC   : {real_ras * 100:.2f}%\n")
                f.write(f"Recall    : {real_rec * 100:.2f}%\n")

        # 扫尾工作，保存文件
        choice_path = os.path.join(save_path, 'ppo_generated_choice.pt')
        fe.generate_data(best_selection_train, 'train').to_hdf(opt_path_test, key='train')
        fe.generate_data(best_selection_train, 'test').to_hdf(opt_path_test, key='test')

        ppo_save_path = os.path.join(save_path, 'ppo.model_dict')
        ppo.save(ppo_save_path)
        print("--------------------------------------------------------------------------------------------")
        print("saving model at : " + ppo_save_path)
        print("--------------------------------------------------------------------------------------------")
        torch.save(new_choice_pt, choice_path)
        info(f'save generated choice to {choice_path}')
    return None

def generate_new_records(queue,ppo, set_tf, feat_len):
    with torch.no_grad():
        # inference
        for i, sample in enumerate(queue):
            encoder_input = sample['input']

            # valid feature index sequence to one-hot vector
            state = choice_to_onehot(encoder_input, feat_len)
            # select action with policy
            action = ppo.select_action(state)  # size: [batch_size, 1, hidden_embedding_size]
            feat_emb, _ = set_tf(encoder_input.cuda(set_tf.gpu).float())
            new_feat_emb = action + feat_emb.squeeze()
            new_output = set_tf.infer(new_feat_emb.unsqueeze(1))
            new_records = output_to_state(new_output, encoder_input, feat_len)
    return new_records

def select_top_k(choice: Tensor, labels: Tensor, k:int) -> tuple[Tensor, Tensor]:
    values, indices = torch.topk(labels, k, dim=0)
    return choice[indices.squeeze()], labels[indices.squeeze()]

def main():
    if not torch.cuda.is_available():
        info('No GPU found!')
        sys.exit(1)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.enabled = True
    cudnn.benchmark = False
    cudnn.deterministic = True
    device = int(args.gpu)
    info(f"Args = {args}")
    
    with open(f'{base_path}/history/{args.task_name}/new_fe.pkl', 'rb') as f:
        fe: FeatureEvaluator = pickle.load(f)
        
    set_tf = SetTransformer(fe.ds_size, fe.ds_size, fe.ds_size, args)
    
    # === 自动寻找最新模型代码 开始 ===
    import glob
    model_dir = f'{base_path}/history/{args.task_name}/set_tf/'
    model_files = glob.glob(os.path.join(model_dir, '*.model_dict'))
    if not model_files:
        error(f"在 {model_dir} 下找不到训练好的模型文件，请先运行 train.py！")
        sys.exit(1)
    latest_model = max(model_files, key=os.path.getctime)
    info(f"找到并加载最新模型: {latest_model}")
    set_tf.load_state_dict(torch.load(latest_model))
    # === 自动寻找最新模型代码 结束 ===
    set_tf = set_tf.cuda(device)

    ppo_agent = PPO(fe.ds_size, args.set_tf_hidden_size, args.ppo_hidden_size, args.lr_actor, args.lr_critic, args.gamma, args.search_step, args.eps_clip)
    ppo_agent = ppo_agent.cuda(device)

    valid_choice, valid_labels = fe.get_record(0, eos=fe.ds_size)

    top_selection, top_performance = select_top_k(valid_choice, valid_labels, args.top_k)

    infer_dataset = FSDataset_set_tf(top_selection, top_performance, sos_id=fe.ds_size, eos_id=fe.ds_size)
    infer_queue = DataLoader(infer_dataset, batch_size=len(infer_dataset), shuffle=False,
                             pin_memory=True)

    flag = ppo_search(infer_queue, ppo_agent, set_tf, fe.ds_size, args.epoch, fe, args.reward_tradeoff)
    return flag

if __name__ == '__main__':
    train_set_TF_args.task_name = args.task_name
    main()