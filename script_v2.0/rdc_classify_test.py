

import torch
from torch.utils.data import DataLoader
import pickle
import numpy as np
from sklearn import metrics
import time 
import logging
import argparse
import numpy as np
import os

from model_def import  build_model_class
from data_utils import CustomDataset
from utils import build_snp_token_from_str


demo_str = 'alt:a,chrom:4,gt:0/0,pos:38898973,ref:g'

def build_extra_remove_snp(custom_args):
    if custom_args.mask_token:
        mask_token = pickle.load(open(custom_args.mask_token, 'rb'))
        snp_token_str_remove = filter(lambda x : len(x.split(',')) == 5, list(mask_token['snp_token_str_rm']) ) 
        snp_token_str_remove = [x for x in snp_token_str_remove]
        snp_token_remove = [build_snp_token_from_str(snp_str) for snp_str in snp_token_str_remove]
        snp_remove = set(snp.get_snp() for snp in snp_token_remove)
    else:
        snp_remove = set()
    return snp_remove


def save_choosed_snp(custom_args, dataset_vocabulary):
    all_snp_token_str =  filter(lambda x : len(x.split(',')) == 5, list(set(dataset_vocabulary)) ) 
    snp_token_all = [build_snp_token_from_str(snp_str) for snp_str in all_snp_token_str]
    snp_remove = build_extra_remove_snp(custom_args)
    
    snp_token_valid = list( filter(lambda snp_token: snp_token.get_snp() not in snp_remove, snp_token_all) )
    snp_valid = set(snp_token.get_snp() for snp_token in snp_token_valid)
    with open(custom_args.snp_mark_file, 'w') as fw:
        for line in snp_valid:
            fw.write(line+'\n')
        
def main_proc(custom_args):
    logging.info('------a new classify test--------')
    POS_IDX = 1
    
    token_vec = os.path.join(custom_args.checkpoint, 'token_vec.pickle')
    dataset = CustomDataset(custom_args.data, vocabulary_file=token_vec, test_flag=True)

    if custom_args.snp_mark_file is not None:
        save_choosed_snp(custom_args, dataset.vectorizer.vocabulary_ )

    if custom_args.mock_y > 0:
        data_cnt = dataset.x.shape[0]
        dataset.y = np.concatenate([ np.array([[1] ,[1], [1]]) , np.zeros([data_cnt-3, 1])])
    
    data_loader = DataLoader(dataset, batch_size=2, drop_last=True)

    logging.info('restore saved model')

    checkpoint_file = os.path.join(custom_args.checkpoint, 'model.pt')
    checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
    model_restore = checkpoint['model_def']
    model_class = build_model_class(checkpoint['model_def_name'])
    model_restore = model_class()
    model_restore.load_state_dict(checkpoint['model_param'])
    model_restore.eval()

    all_pred, all_y = [], []

    with torch.no_grad():
        for num, (x, y) in enumerate(data_loader) :
            pred = model_restore(x)
            all_pred.append(pred[:,POS_IDX].reshape(-1, 1))
            all_y.append(y.reshape(-1, 1))
    
    all_pred = torch.concat(all_pred)
    all_y = torch.concat(all_y)
    suggest_threshold = 0.95
    all_pred_bin = torch.where(all_pred > suggest_threshold, torch.tensor(1, dtype=torch.long), torch.tensor(0, dtype=torch.long))

    if custom_args.test_result_file is not None:
        with open(custom_args.test_result_file, 'w') as fw:
            l1 = []
            for pred0, y0, sample_id, breed_info, mit_info in zip(all_pred.reshape(-1).numpy()
                                            , all_y.reshape(-1).numpy() 
                                            , dataset.sample_ids
                                            , dataset.breed_info 
                                            , dataset.mit_info
                                            ):
                pred_bin0 = 1 if pred0 > suggest_threshold else 2
                pred_bin0 = 0 if pred0 < 1-suggest_threshold else pred_bin0
                l1.append((pred0, pred_bin0, y0, sample_id, breed_info, mit_info))
            l1.sort(key = lambda x: x[0], reverse=True)
            fw.write('pred_score,pred_label,ori_label,sample_id,breed_info,mit_info\n')
            for pred0, pred_bin0, y0, sample_id, breed_info, mit_info in l1:
                fw.write('{},{},{},{},{},{}\n'.format(pred0, pred_bin0, y0, sample_id, breed_info, mit_info))
    pos_cnt = sum(all_y)
    auc_metric = metrics.roc_auc_score(all_y, all_pred)
    recall, precision = metrics.recall_score(all_y, all_pred_bin), metrics.precision_score(all_y, all_pred_bin)
    f1_score = metrics.f1_score(all_y, all_pred_bin)
    logging.info('test_size->{:d}, pos_cnt->{:d},  auc->{:.4f}, suggest_threshold -> {:.4f}, recall->{:.4f}, precision->{:.4f}, f1_score->{:.4f}'.format(
        len(all_pred), pos_cnt.item(), auc_metric, suggest_threshold, recall, precision, f1_score))
    
    th_l, recall_l, precision_l = [], [], []
    for th in np.arange(0.01, 1, step=0.01):
        all_pred_bin = torch.where(all_pred > th
                                   , torch.tensor(1, dtype=torch.long)
                                   , torch.tensor(0, dtype=torch.long))
        
        recall, precision = metrics.recall_score(all_y, all_pred_bin), metrics.precision_score(all_y, all_pred_bin)
        th_l.append(th)
        recall_l.append(recall)
        precision_l.append(precision)
        if precision > 0.95:
            break
    
    logging.info('find th {}, precision {}, max-recall {}'.format(th, precision_l[-2:], recall_l[-2:]))


def emb_pca(custom_args):
    dataset = CustomDataset(custom_args.data, vocabulary_file=custom_args.token_vec_file
                            , test_flag=True
                            , mit_flag = custom_args.mit_flag)
    model_class = build_model_class(custom_args.model_flag)
    x = dataset.vectorizer.transform(dataset.corpus)
    tokens = [ _.indices for _ in x]
    
    logging.info('restore saved model')
    _load_state = torch.load(custom_args.model_path, map_location=torch.device('cpu'))
    emb_table = _load_state[custom_args.emb_weight_key]
    result = []
    for one_sample, smaple_id, breed in zip(tokens, dataset.sample_ids, dataset.breed_info):
        emb_list = []
        for one_token in one_sample:
            if one_token == 0: continue
            emb = emb_table[one_token].unsqueeze(0)
            emb_list.append(emb)
        emb_all = torch.concat(emb_list, dim=0)
        emb = torch.sum(emb_all, dim=0)/len(emb_list)
        emb = emb.numpy()
        result.append( (smaple_id, emb, one_sample, breed))
    with open(os.path.join(args.model_dir, 'sample_emb'), 'wb') as fw:
        pickle.dump(result, fw)
    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='dg_test'
    )
    current_time_str = time.strftime("%Y%m%d_%H%M", time.localtime())
    parser.add_argument('--method', type=str, default= "test")
    parser.add_argument('--checkpoint', type=str, required=True )
    parser.add_argument('--mask_token', type=str, default="")
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--snp_mark_file', type=str, default=None)
    parser.add_argument('--logger_file', type=str, default='test_result.log')
    parser.add_argument('--logging_level', type=int, default=12)
    parser.add_argument('--test_result_file', type=str, default=None)
    parser.add_argument('--mock_y', type=int, default= 0)
    parser.add_argument('--emb_weight_key', type=str, default= "embedding.embedding.weight")


    args = parser.parse_args()
    

    print('in test proc')
    print(args)
    logging.basicConfig(filename=args.logger_file, level=args.logging_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    method_dict = {
        'test' : main_proc
        ,'pca' : emb_pca
    }

    method_func = method_dict.get(args.method, lambda x: print('unknow method'))
    method_func(args)
    

