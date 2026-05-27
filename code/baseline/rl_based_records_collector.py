import os
import sys

# === 自动锁定绝对路径（极其稳健） ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 当前 baseline 文件夹
PARENT_DIR = os.path.dirname(BASE_DIR)                 # 上一级 code 文件夹
sys.path.append(PARENT_DIR)
# ==================================

from feature_env import FeatureEvaluator, base_path
# from baseline.model import *
# from utils.logger import info
# import pickle

# import warnings

# warnings.filterwarnings('ignore')

# baseline_name = {
#     'gfs': gen_gfs,
#     'kbest': gen_kbest,
#     'mrmr': gen_mrmr,
#     'lasso': gen_lasso,
#     'rfe': gen_rfe,
#     'lassonet': gen_lassonet,
#     'sarlfs': gen_sarlfs,  # feature_env, N_STATES, N_ACTIONS, EPISODE=-1, EXPLORE_STEPS=30
#     # 'marlfs': gen_marlfs,  # feature_env, N_STATES, N_ACTIONS, EPISODE=-1, EXPLORE_STEPS=30
#     'rra': gen_rra,
#     'mcdm': gen_mcdm
# }
from baseline.model.MARLFS import gen_marlfs  # 只精准导入这一个！
from utils.logger import info
import pickle

import warnings
warnings.filterwarnings('ignore')

# 那个大字典 baseline_name 已经不需要了，直接删掉！

def gen_auto_feature_selection(fe_, task_name_):
    fe_.train.to_hdf(f'{base_path}/history/{task_name_}.hdf', key='raw_train')
    fe_.test.to_hdf(f'{base_path}/history/{task_name_}.hdf', key='raw_test')
    max_accuracy, optimal_set, k = gen_marlfs(fe_, N_ACTIONS=2, N_STATES=64, EXPLORE_STEPS=100)
    best_train = fe_.generate_data(optimal_set, 'train')
    best_test = fe_.generate_data(optimal_set, 'test')
    best_train.to_hdf(f'{base_path}/history/{task_name_}.hdf', key='marlfs_train')
    best_test.to_hdf(f'{base_path}/history/{task_name_}.hdf', key='marlfs_test')
    info('working on {} task, best k = {}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'.format(task_name_,k))
    # for name_, func in baseline_name.items():
    #     p_, optimal_set = func(fe_, k)
    #     best_train = fe_.generate_data(optimal_set,  flag='train')
    #     best_test = fe_.generate_data(optimal_set, flag='test')
    #     best_train.to_hdf(f'{base_path}/history/{task_name_}.hdf', key=f'{name_}_train')
    #     best_test.to_hdf(f'{base_path}/history/{task_name_}.hdf', key=f'{name_}_test')
    return k


def process(task_name_):
    fea_eval = FeatureEvaluator(task_name_)
    gen_auto_feature_selection(fea_eval, task_name_)
    with open(f'{base_path}/history/{task_name_}/new_fe.pkl', 'wb') as f:
        pickle.dump(fea_eval,  f)



if __name__ == '__main__':

    task_list = ['spectf']
    for name in task_list:
        task_name = name
        process(task_name)


