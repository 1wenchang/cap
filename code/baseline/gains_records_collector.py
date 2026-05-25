import os
import sys
# print(sys.path)
sys.path.append('..')
import torch
from feature_env import FeatureEvaluator, base_path
from baseline.model import *
from utils.logger import info
import pickle

import warnings

warnings.filterwarnings('ignore')



def get_gains(fe: FeatureEvaluator):

    choice_path = f'{base_path}/history/{fe.task_name}/generated_choice.pt'
    new_selection = torch.load(choice_path)

    best_selection_test = None
    best_optimal_test = -1000

    for choice in new_selection:
        test_data = fe.generate_data(choice.numpy(), 'test')
        test_result = fe.get_performance(test_data)

        if test_result > best_optimal_test:
            best_selection_test = choice.numpy()
            best_optimal_test = test_result
            info(f'found best on test : {best_optimal_test}')

    result = fe.report_performance (best_selection_test, flag='train')
    test_p = fe.report_performance(best_selection_test, flag='test', store = False)
    info("The optimal accuracy is: {}, gains optimal selection for {} is:{}".format(test_p, fe.task_name, best_selection_test))
    return best_selection_test
def gen_auto_feature_selection(fe_, task_name_):

    optimal_set = get_gains(fe_)
    best_train = fe_.generate_data(optimal_set, 'train')
    best_test = fe_.generate_data(optimal_set, 'test')
    best_train.to_hdf(f'{base_path}/history/{task_name_}.hdf', key='gains_train')
    best_test.to_hdf(f'{base_path}/history/{task_name_}.hdf', key='gains_test')
    return None


def process(task_name_):
    with open(f'{base_path}/history/{task_name_}/fe.pkl', 'rb') as f:
        fea_eval: FeatureEvaluator = pickle.load(f)
        gen_auto_feature_selection(fea_eval, task_name_)
        with open(f'{base_path}/history/{task_name_}/new_fe.pkl', 'wb') as new_f:
            pickle.dump(fea_eval,  new_f)



if __name__ == '__main__':

    # task_list = ['spectf', 'svmguide3', 'german_credit', 'spam_base',
    #               'ionosphere', 'megawatt1', 'uci_credit_card', 'openml_618',
    #               'openml_589', 'openml_616', 'openml_607', 'openml_620',
    #               'openml_637',
    #               'openml_586', 'activity'
    #               , 'mice_protein', 'coil-20', 'minist', 'minist_fashion']
    task_list = ['urbansound8k']
    for i in range(len(task_list)):
        task_name = task_list[i]
        process(task_name)


