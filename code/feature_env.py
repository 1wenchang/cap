"""
feature env
interactive with the actor critic for the state and state after action
"""
import os
from collections import namedtuple
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from record import RecordList
from utils.logger import error, info
from utils.tools import test_task_new, downstream_task_new, downstream_task_by_method_std
import pickle
import os
# 自动定位当前 feature_env.py 所在的 code 文件夹
_current_dir = os.path.dirname(os.path.abspath(__file__))
# 自动退一层，找到 CAPS-main 根目录
_project_root = os.path.dirname(_current_dir)
# 绝对锁定 data 文件夹的位置
base_path = os.path.join(_project_root, 'data').replace('\\', '/')

TASK_DICT = {'airfoil': 'reg', 'amazon_employee': 'cls', 'ap_omentum_ovary': 'cls',
             'bike_share': 'reg', 'german_credit': 'cls', 'higgs': 'cls',
             'housing_boston': 'reg', 'ionosphere': 'cls', 'lymphography': 'cls',
             'messidor_features': 'cls', 'openml_620': 'reg', 'pima_indian': 'cls',
             'spam_base': 'cls', 'spectf': 'cls', 'svmguide3': 'cls',
             'uci_credit_card': 'cls', 'wine_red': 'cls', 'wine_white': 'cls',
             'openml_586': 'reg', 'openml_589': 'reg', 'openml_607': 'reg',
             'openml_616': 'reg', 'openml_618': 'reg', 'openml_637': 'reg',
             'smtp': 'det', 'thyroid': 'det', 'yeast': 'det', 'wbc': 'det', 'mammography': 'det', 'arrhythmia': 'cls',
             'nomao': 'cls', 'megawatt1': 'cls', 'activity':'mcls', 'mice_protein':'mcls', 'coil-20':'mcls', 'isolet':'mcls', 'minist':'mcls',
             'minist_fashion':'mcls','IQdataset':'reg','urbansound8k':'mcls'
             }

MEASUREMENT = {
    'cls': ['precision', 'recall', 'f1_score', 'roc_auc'],
    'reg': ['mae', 'mse', 'rae', 'rmse'],
    'det': ['map', 'f1_score', 'ras', 'recall'],
    'mcls' : ['precision', 'recall', 'mif1', 'maf1']
}

model_performance = {
    'mcls':namedtuple('ModelPerformance', MEASUREMENT['mcls']),
    'cls': namedtuple('ModelPerformance', MEASUREMENT['cls']),
    'reg': namedtuple('ModelPerformance', MEASUREMENT['reg']),
    'det': namedtuple('ModelPerformance', MEASUREMENT['det'])
}

class Evaluator(object):
    def __init__(self, task, task_type=None, dataset=None):
        self.original_report = None
        self.records = RecordList()
        self.task_name = task
        if task_type is None:
            self.task_type = TASK_DICT[self.task_name]
        else:
            self.task_type = task_type
        ##############################################################################
        # if dataset is None:
        #     data_path = os.path.join(base_path+'/history/', self.task_name + '.hdf')
        #     original = pd.read_hdf(data_path)
        # else:
        #     original = dataset
        ##############################################################################
        data_path = os.path.join(base_path + '/history/', self.task_name + '.hdf')
        # ========== 破窗提取法开始 ==========
        try:
            original = pd.read_hdf(data_path)
        except Exception as e:
            import h5py
            import pandas as pd
            import numpy as np
            print(f"\n[破窗系统] Pandas 读取失败，启动 h5py 强行提取 {self.task_name}.hdf ...")
            
            with h5py.File(data_path, 'r') as f:
                matrices = []
                # 深入压缩包，把所有纯数字矩阵全抓出来
                def extract(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        arr = obj[()]
                        # 只要是数字就拿出来
                        if np.issubdtype(arr.dtype, np.number):
                            if arr.ndim == 1:
                                arr = arr.reshape(-1, 1)
                            matrices.append(arr)
                f.visititems(extract)
                
                if not matrices:
                    raise Exception(f"在 {data_path} 里找不到任何数值数据！作者可能给错文件了。")
                
                # 过滤并拼接：只把行数相同的矩阵拼在一起（防止作者存了一些乱七八糟的单行配置）
                std_len = max([m.shape[0] for m in matrices])
                valid_data = [m for m in matrices if m.shape[0] == std_len]
                
                # 拼成 Pandas 需要的 DataFrame
                original = pd.DataFrame(np.hstack(valid_data))
            print("[破窗系统] 底层数据矩阵提取并拼装成功！\n")
        # ========== 破窗提取法结束 ==========

        col = np.arange(original.shape[1])
        self.col_names = original.columns
        original.columns = col
        y = original.iloc[:, -1]
        x = original.iloc[:, :-1]
        if task == 'ap_omentum_ovary':
            y[y == 'Ovary'] = 1
            y[y == 'Omentum'] = 0
            y = y.astype(float)
            original = pd.concat([pd.DataFrame(x), pd.DataFrame(y)], axis=1)
        self.original = original.fillna(value=0)
        y = self.original.iloc[:, -1]
        x = self.original.iloc[:, :-1]
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                            random_state=0, shuffle=True)

        self.train = pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train)], axis=1)
        self.test = pd.concat([pd.DataFrame(X_test), pd.DataFrame(y_test)], axis=1)
        info('initialize the train and test dataset')
        self._check_path()

    def __len__(self):
        return len(self.records)

    def generate_data(self, operation, flag):
        pass

    def get_performance(self, data=None):
        if data is None:
            data = self.original
        return downstream_task_new(data, self.task_type)

    def report_ds(self):
        pass

    def _store_history(self, choice, performance):
        self.records.append(choice, performance)

    def _flush_history(self, choices, performances, is_permuted, num, padding):
        if is_permuted:
            flag_1 = 'augmented'
        else:
            flag_1 = 'original'
        if padding:
            flag_2 = 'padded'
        else:
            flag_2 = 'not_padded'
        torch.save(choices, f'{base_path}/history/{self.task_name}/choice.{flag_1}.{flag_2}.{num}.pt')
        info(f'save the choice to {base_path}/history/{self.task_name}/choice.pt')
        torch.save(performances, f'{base_path}/history/{self.task_name}/performance.{flag_1}.{flag_2}.{num}.pt')
        info(f'save the performance to {base_path}/history/{self.task_name}/performance.pt')

    def _check_path(self):
        if not os.path.exists(f'{base_path}/history/{self.task_name}'):
            os.mkdir(f'{base_path}/history/{self.task_name}')

    def save(self, num=25, padding=True, padding_value=-1):
        if num > 0:
            is_permuted = True
        else:
            is_permuted = False
        info('save the records...')
        choices, performances = \
            self.records.generate(num=num, padding=padding, padding_value=padding_value)
        self._flush_history(choices, performances, is_permuted, num, padding)

    def get_record(self, num=0, eos=-1):
        results = []
        labels = []
        for record in self.records.r_list:
            result, label = record.get_permutated(num, True, eos)
            results.append(result)
            labels.append(label)
        return torch.cat(results, 0), torch.cat(labels, 0)

    def get_triple_record(self, num=0, eos=-1, mode='ht'):
        h_results = []
        labels = []
        t_results = []
        h_seed = []
        labels_seed = []
        for record in self.records.r_list:
            if mode.__contains__('h'):
                h, label = record.get_permutated(num, True, eos)
            else:
                h, label = record.repeat(num, True, eos)
            if mode.__contains__('t'):
                t, _ = record.get_permutated(num, True, eos)
            else:
                t, _ = record.repeat(num, True, eos)
            h_results.append(h)
            t_results.append(t)
            labels.append(label)
            h_seed.append(h_results[0])
            labels_seed.append(labels[0])
        return torch.cat(h_results, 0), torch.cat(labels, 0), torch.cat(t_results), \
               torch.cat(h_seed), torch.cat(labels_seed),

    def report_performance(self, choice, store=True, rp=True, flag=''):
        opt_ds = self.generate_data(choice, flag)
        a, b, c, d = test_task_new(opt_ds, task=self.task_type)
        report = model_performance[self.task_type](a, b, c, d)
        if flag == 'test':
            store = False
        if self.original_report is None:
            a, b, c, d = test_task_new(self.test, task=self.task_type)
            self.original_report = (a, b, c, d)
        else:
            a, b, c, d = self.original_report
        original_report = model_performance[self.task_type](a, b, c, d)
        if self.task_type == 'reg':
            final_result = report.rae
            if rp:
                info('1-MAE on original is: {:.4f}, 1-MAE on generated is: {:.4f}'.
                     format(original_report.mae, report.mae))
                info('1-MSE on original is: {:.4f}, 1-MSE on generated is: {:.4f}'.
                     format(original_report.mse, report.mse))
                info('1-RAE on original is: {:.4f}, 1-RAE on generated is: {:.4f}'.
                     format(original_report.rae, report.rae))
                info('1-RMSE on original is: {:.4f}, 1-RMSE on generated is: {:.4f}'.
                     format(original_report.rmse, report.rmse))
        elif self.task_type == 'cls':
            final_result = report.f1_score
            if rp:
                info('Pre on original is: {:.4f}, Pre on generated is: {:.4f}'.
                     format(original_report.precision, report.precision))
                info('Rec on original is: {:.4f}, Rec on generated is: {:.4f}'.
                     format(original_report.recall, report.recall))
                info('F-1 on original is: {:.4f}, F-1 on generated is: {:.4f}'.
                     format(original_report.f1_score, report.f1_score))
                info('ROC/AUC on original is: {:.4f}, ROC/AUC on generated is: {:.4f}'.
                     format(original_report.roc_auc, report.roc_auc))
        elif self.task_type == 'det':
            final_result = report.ras
            if rp:
                info(
                    'Average Precision Score on original is: {:.4f}, Average Precision Score on generated is: {:.4f}'
                    .format(original_report.map, report.map))
                info(
                    'F1 Score on original is: {:.4f}, F1 Score on generated is: {:.4f}'
                    .format(original_report.f1_score, report.f1_score))
                info(
                    'ROC AUC Score on original is: {:.4f}, ROC AUC Score on generated is: {:.4f}'
                    .format(original_report.ras, report.ras))
                info(
                    'Recall on original is: {:.4f}, Recall Score on generated is: {:.4f}'
                    .format(original_report.recall, report.recall))
        elif self.task_type == 'mcls':
            final_result = report.mif1
            if rp:
                info('Pre on original is: {:.4f}, Pre on generated is: {:.4f}'.
                     format(original_report.precision, report.precision))
                info('Rec on original is: {:.4f}, Rec on generated is: {:.4f}'.
                     format(original_report.recall, report.recall))
                info('Micro-F1 on original is: {:.4f}, Micro-F1 on generated is: {:.4f}'.
                     format(original_report.mif1, report.mif1))
                info('Macro-F1 on original is: {:.4f}, Macro-F1 on generated is: {:.4f}'.
                     format(original_report.maf1, report.maf1))
        else:
            error('wrong task name!!!!!')
            assert False
        if store:
            self._store_history(choice, final_result)
        return final_result


class FeatureEvaluator(Evaluator):
    def __init__(self, task, task_type=None, dataset=None):
        super().__init__(task, task_type, dataset)
        self.ds_size = self.original.shape[1] - 1

    def generate_data(self, choice, flag=''):
        if choice.shape[0] != self.ds_size:
            error('wrong shape of choice')
            assert False
        if flag == 'test':
            ds = self.test
        elif flag == 'train':
            ds = self.train
        else:
            ds = self.original
        X = ds.iloc[:, :-1]
        indice = torch.arange(0, self.ds_size)[choice == 1].numpy()
        X = X.iloc[:, indice].astype(np.float64)
        y = ds.iloc[:, -1].astype(np.float64)
        Dg = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1)
        return Dg

    def _full_mask(self):
        return torch.FloatTensor([1] * self.ds_size)

    def report_ds(self):
        per = self.get_performance()
        info(f'current dataset : {self.task_name}')
        info(f'the size of shape is : {self.original.shape[1]}')
        info(f'original performance is : {per}')
        self._store_history(self._full_mask(), per)

import time
# if __name__ == '__main__':
    # task_name = 'spectf'
    # fe = FeatureEvaluator(task_name)
    # # with open('test.pkl', 'wb') as f:
    # #     pickle.dump(fe,  f)
    # #
    # # with open('test.pkl', 'rb') as a:
    # #     new_fe: FeatureEvaluator = pickle.load(a)
    #
    # start_time = time.time()
    # end_time = time.time()
    # info(f'training on overall eval cost : {end_time - start_time}s')
    # for method in ['RF', 'XGB', 'SVM', 'KNN', 'Ridge', 'DT', 'LASSO']:
    #     info(method)
    #     start_time = time.time()
    #     p, std = downstream_task_by_method_std(fe.original, fe.task_type, method)
    #     end_time = time.time()
    #     info(f'training on {method} eval cost : {end_time - start_time}s')
    # fe.report_performance(torch.FloatTensor([0, 1, 1, 0, 0]))
    # fe.save()
    # print(1)
