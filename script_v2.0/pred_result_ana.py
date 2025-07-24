
# -*- coding: utf-8 -*-
from sklearn import metrics
import numpy as np
import pandas as pd
import argparse

def auc_and_roc_relative(data_file, output_file ):
    import matplotlib.pyplot as plt
    all_pred = []
    all_y = []
    for line in open(data_file):
        items = line.strip().split(',')
        all_pred.append(float(items[0]))
        all_y.append(float(items[-1]))
    all_pred = np.array(all_pred)
    all_y = np.array(all_y)
    auc_metric = metrics.roc_auc_score(all_y, all_pred)
    suggest_threshold_x, recall_y, precision_y = [], [], []
    for suggest_threshold in np.arange(0, 1, step=0.1):
        all_pred_bin = np.where(all_pred > suggest_threshold, 1, 0)
        recall, precision = metrics.recall_score(all_y, all_pred_bin), metrics.precision_score(all_y, all_pred_bin)
        f1_score = metrics.f1_score(all_y, all_pred_bin)
        suggest_threshold_x.append(suggest_threshold)
        recall_y.append(1-recall)
        precision_y.append(1-precision)
        # break
    
    plt.plot(suggest_threshold_x, recall_y, label='漏检率')
    plt.plot(suggest_threshold_x, precision_y, label='假阳性')
    plt.legend(loc='best')
    plt.savefig(output_file)

data_file = 'result/exp_7k_0626_dfm_snp100_mask_subset/pred_all_test'


def confusion_matrix_detail(data_file, header):
    pos_th = 0.95
    neg_th = 0.05
    dlj_rjf = set(['RJF50','RJF51','RJF52','RJF53','RJF54','RJF55','RJF56'])

    def pred_adj(pos_score):
        flag = -1
        pos_score = float(pos_score)
        if pos_score > pos_th: 
            flag = 1 
        elif pos_score < neg_th: 
            flag = 0 
        else: flag = 2
        return flag

    col_names = header.split(',')    
    data = pd.read_csv(data_file, header=None, names=col_names)

    # data['y'] = data['sample_id'].map(lambda x: 0 if x.startswith('DC') else 1)
    data['paper_label'] = data['sample_id'].map(lambda x: 1 if len(x.split('_')) > 1 else 0)
    data['sample_geo'] = data['sample_id'].map(lambda x: 'dlj' if x in dlj_rjf else 'other')
    data['pred_adj'] =  data['pos_score'].map(pred_adj)

    def stat_1(data, label):
        data_pos = data[data['y'] == 1]
        data_neg = data[data['y'] == 0]

        rjf_pred_cnt_in_pos = len(data_pos[ data_pos['pred_adj'] == 1])
        dc_pred_cnt_in_pos  = len(data_pos[ data_pos['pred_adj'] == 0])

        rjf_pred_cnt_in_neg = len(data_neg[ data_neg['pred_adj'] == 1 ])
        dc_pred_cnt_in_neg = len(data_neg[ data_neg['pred_adj'] == 0 ])
        print('------{}-----'.format(label))
        print('in pos: pred_rjf {}, pred_dc {}, all {}'.format(rjf_pred_cnt_in_pos, dc_pred_cnt_in_pos, len(data_pos)))
        print('in neg: pred_rjf {}, pred_dc {}, all {}'.format(rjf_pred_cnt_in_neg, dc_pred_cnt_in_neg, len(data_neg)))
        print('------{}-----'.format(label))


    stat_1(data, 'total')
    stat_1(data[ data['group_info'] == 'Gallus_gallus_spadiceus'], 'GGS')
    stat_1(data[ data['paper_label'] == 1], 'paper sample')
    stat_1(data[ data['paper_label'] != 1], 'our sample')


    stat_1(data[ data['sample_geo'] == 'dlj'], 'our sample - dlj sample')


    stat_1(data[ (data['paper_label'] != 1) & ( data['sample_geo'] != 'dlj' )  ] , 'our sample - not dlj sample') 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='rfj_result_analysis'
    )

    parser.add_argument('--method', type=str, default='confusion_matrix_detail')
    parser.add_argument('--logger_file', type=str, default='./logger/analysis.log')
    parser.add_argument('--logging_level', type=int, default=12)
    parser.add_argument('--result_file', type=str, default='')
    parser.add_argument('--result_header', type=str, default='pos_score,pred,y,sample_id,group_info,mit_info')
    parser.add_argument('--output_file', type=str, default='')

    args = parser.parse_args()

    confusion_metric = lambda args: confusion_matrix_detail(args.result_file, args.result_header)
    false_pos_and_recall = lambda args: auc_and_roc_relative(args.result_file, args.output_file)

    method_dict = {
        'confusion_matrix_detail' : confusion_metric
        ,'false_pos_and_recall' : false_pos_and_recall
    }

    func =  method_dict.get(args.method)
    func(args)