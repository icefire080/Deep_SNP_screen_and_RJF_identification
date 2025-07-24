# -*- coding: utf-8 -*- 

import torch
import pickle
import numpy as np
import time 
import logging
import argparse
import pprint
import random
import numpy as np
import os
from functools import cmp_to_key
import copy
import yaml
from collections import namedtuple

##self defined package
from utils import timer_simple, find_all_snp,  sort_np_matrix_row_by_col_val
from mask_diff_utils import core_diff_mask_weight, get_diff_mask_weight_base
from model_train import train_a_model
from model_def import  build_model_class
from data_utils import CustomDataset, SimpleDataset




def extrac_pos_sample_and_specific_sample(x, y, specific_sample_idx):
    idx = y.squeeze().nonzero()[0]
    all_idx = np.concatenate([idx, specific_sample_idx])
    all_idx = all_idx.astype(np.int64)
    x_pos = x[all_idx, :]
    y_pos = y[all_idx, :]
    board = len(idx)
    return x_pos, y_pos, all_idx

def snp_set_to_snp_token_idx(mask_token_set, snp_2_str, vocabulary):
    mask_snp_token_str_set = list()
    for snp in mask_token_set:
        mask_snp_token_str_set.extend( snp_2_str.get(snp, []) )
    # e.g. chr:1,pos:22 -> chr:1,pos:22,gt:1/0 ; chr:1,pos:22,gt:1/1
    mask_snp_token_str_set = list(set(mask_snp_token_str_set))
    snp_token_idx = [(vocabulary.get(x, 0), x) for x in mask_snp_token_str_set]
    # e.g. sparse-id
    return snp_token_idx

@timer_simple
def find_snp_by_mask_list(custom_args, stop_words = []):
    all_stop_snp = []
    snp_search_conf = custom_args.conf['snp_search']
    seach_globa_step = 0
    for i in range(snp_search_conf['max_find_epoch']):
        stop_snp, local_stop_words, pre_result, reserved_snp_cnt, seach_globa_step = find_max_sub_snp_list(custom_args, stop_words, seach_globa_step)
        stop_words.extend(local_stop_words)
        all_stop_snp.extend(stop_snp)
        #stop_words e.g. chr:1,gt:1/0,pos:12,ref:a,alt,g
        #all_stop_snp e.g. chr:1,pos:12
        pickle.dump({'idx_rm': [12], 'snp_rm':all_stop_snp, 'snp_token_str_rm':stop_words}
                    , open(snp_search_conf['result_remove_snp'] + '.rm_info.epoch{}'.format(i)
                           , 'wb'))
        logging.info('finish choose_epoch {}, seach_globa_step {}, stop_snp_cnt {}, stop_snp_E.g. {}, reserved_snp_cnt {}'.format(
            i, seach_globa_step, len(all_stop_snp), all_stop_snp[:10], reserved_snp_cnt))
        if reserved_snp_cnt < snp_search_conf.get('target_snp_cnt'):
            logging.info('reserved {} snps, early stop'.format(reserved_snp_cnt))
            break
        


def find_max_sub_snp_list(custom_args, stop_words, search_global_step):
    train_conf = custom_args.conf['snp_search_train_param']
    data_conf = custom_args.conf['data']
    seach_conf = custom_args.conf['snp_search']
    logging.info(pprint.pformat(train_conf))

    ## init dataset with stop_words stop-words not in dataset.vectorizer.vocabulary_
    dataset = CustomDataset(data_conf['train_data']
                            , filter_breed = data_conf['snp_select_model_filter_breed']
                            , mit_flag= data_conf['mit_flag']
                            , stop_words_list = stop_words)
    
    ## generate all snp from vocabularyalready removed unuse snp
    ## all-snp and snp map 2 gene type
    all_snp, snp_2_all_gt = find_all_snp(dataset.vectorizer.vocabulary_.keys())
    random.shuffle(all_snp)
    all_snp_cnt = len(all_snp)

    device = torch.device(train_conf['cuda'] if torch.cuda.is_available() and train_conf['cpu'] is False  else "cpu")
    model_class = build_model_class(train_conf['model_flag'])
    model = train_a_model(device, model_class, dataset, train_conf)
    Result_ele = namedtuple('result_ele', ('candidate_snp', 'mask_token_set', 'diff_score', 'score_detail' ))
    result = [] #e.g. [(candidate_snp [ tree root snp_token ]:int, mask_token_set:["chrom:1,pos:105328396", "chrom:2,pos:105328396"],  diff_score:float, score_detail:str), (,)]

    ##get base score for base diff in-silico 
    pred_all_base, y_all_base = get_diff_mask_weight_base(model, SimpleDataset(dataset.x, dataset.y), device=device)
    pred_base = torch.concat([pred_all_base, y_all_base], dim=1).numpy() # [pred, y/label]
    all_sample = len(pred_base)
    pos_cnt = len(pred_base[:,1].nonzero()[0])
    idx = np.arange(0, all_sample).reshape((-1, 1))
    pred_base = np.concatenate( [pred_base, idx], axis=1) ##add ori idx then [pred, y/label, idx]
    
    ## sample some valid 10^3 -> 10^2
    pred_base_sort = sort_np_matrix_row_by_col_val(pred_base, [1, 0])
    neg_valid_sample_cnt = 15
    neg_board_sample = pred_base_sort[
        np.arange(all_sample-pos_cnt-1, all_sample-pos_cnt - neg_valid_sample_cnt, -1)
        , -1]

    valid_x, valid_y , all_idx = extrac_pos_sample_and_specific_sample(dataset.x, dataset.y, neg_board_sample)

    # logging.info('pred_base_sort {}, neg_board_sample {}, all_idx {}'.format(pred_base_sort[1507:1550], neg_board_sample, all_idx))

    valid_pred_raw = pred_base[all_idx,0]

    
    ## Traverse all SNPs (all_snp), one by one as the root node of the search
    for candidate_snp_idx in range(len(all_snp)):

        ## subsequence of this search
        ## start from candidate_snp_idx 1,2,3
        sub_snp_list = copy.deepcopy(all_snp)[candidate_snp_idx:]
        ## init the result in this search
        mask_token_set = set()
        mask_token_set.add(sub_snp_list[0])
        
        for cursor in range(1, len(sub_snp_list)):
            search_global_step += 1

            one_snp = sub_snp_list[cursor]
            mask_token_set.add(one_snp) 
            ## snp to snp_gt to vocob index
            mask_token_idx_list = snp_set_to_snp_token_idx(mask_token_set, snp_2_all_gt, dataset.vectorizer.vocabulary_)
            diff_score, score_detail = core_diff_mask_weight(valid_pred_raw, mask_token_idx_list, model, SimpleDataset(valid_x, valid_y), device=device, search_param=seach_conf)
            if diff_score > 0.5: mask_token_set.remove(one_snp)

            # early stop one
            if len(mask_token_set) > seach_conf.get('find_step') or diff_score > 1: 
                logging.info('[early stop inner] mask_token_set {}, diff_score {}, score_detail {}'.format(len(mask_token_set), diff_score, score_detail))
                break

        
        logging.info('for left token {}, global_idx {}, set_len {}, find mask_token_set {}'.format(sub_snp_list[0], candidate_snp_idx, len(mask_token_set), list(mask_token_set)[:6]))

        
        if diff_score > 1:
            logging.info('early stop final base too weak {}'.format(score_detail))
            break

        if len(mask_token_set) == 1:
            a2 = snp_set_to_snp_token_idx(mask_token_set, snp_2_all_gt, dataset.vectorizer.vocabulary_)
            diff_score, score_detail = core_diff_mask_weight(valid_pred_raw, a2, model, SimpleDataset(valid_x, valid_y), device=device, search_param=seach_conf)
            if diff_score > 0.5: continue

        result_ele = Result_ele(candidate_snp_idx, mask_token_set, diff_score, score_detail)
        result.append(result_ele)
        if len(mask_token_set) > seach_conf.get('find_step'): 
            logging.info('early stop final')
            break

#e.g. result [(candidate_snp [ tree root snp_token ]:int, mask_token_set:["chrom:1,pos:105328396", "chrom:2,pos:105328396"],  diff_score:float, score_detail:str), (,)]
    if len(result) == 0: return [], [], [], all_snp_cnt, search_global_step
    #base model too weak can not implement in-silico experiment
    
    result.sort(key=lambda x: len(x[1]), reverse=True) ##sort by mask_Set token

    
    def build_choose_reason(result):
        detail_str_f = lambda result_ele: '{},len {}, diff_score {}, score_detail {}'.format(
            result_ele.candidate_snp, len(result_ele.mask_token_set) , result_ele.diff_score , result_ele.score_detail) 
        top_one = detail_str_f(result[0])
        last_one = detail_str_f(result[-1])
        reason = 'in {} result len {}, result_0 {}, result_-1 {}'.format(
        find_max_sub_snp_list, len(result), top_one, last_one)
        return reason
    
    logging.info(build_choose_reason(result))

    final_mask = result[0].mask_token_set
    final_mask_snp_token_str = []
    for snp in final_mask:
        final_mask_snp_token_str.extend( snp_2_all_gt.get(snp, []) )

    reserved_snp_cnt = 0
    for x in filter(lambda x: x not in set(final_mask), all_snp):
        reserved_snp_cnt += 1
    return final_mask, final_mask_snp_token_str, result, reserved_snp_cnt, search_global_step
    # e.g. final_mask = set("chrom:1,pos:105328396", "chrom:2,pos:105328396"),  final_mask_snp_token_str = ["chrom:1,pos:1205,ref:a,alt:g,gt:0/1", "chrom:1,pos:1205,ref:a,alt:g,gt:0/1"]


def main_proc(custom_args):
    utils_conf = custom_args.conf['utils']
    logging.basicConfig(filename=utils_conf['logger_file'], level=utils_conf['logging_level'], format='%(module)s/%(funcName)s-%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logging.info('-----begin a finding----')

    if custom_args.method == 'mask_weight':
        find_snp_by_mask_list(custom_args)
    elif custom_args.method == 'mask_weight_with_init':
        stop_words_pickle_file = custom_args.conf['data']['init_stop_snps']
        stop_words_info = pickle.load(open(stop_words_pickle_file + '.rm_info', 'rb'))
        find_snp_by_mask_list(custom_args, stop_words=stop_words_info['snp_token_str_rm'])
    else:
        logging.info('no this method')
        
    logging.info('-----end a choose process----')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='rfj_mask_sublist')

    parser.add_argument('--method', type=str, default='mask_weight')
    parser.add_argument('--conf', type=str, default='dg_rdg/config_test.yml')
    
    args = parser.parse_args()
    with open(args.conf, 'r') as _file:
        conf = yaml.safe_load(_file)
    print(conf)
    args.conf = conf
    model_dir = conf['utils']['model_dir']
    if not os.path.exists(model_dir): os.makedirs(model_dir)
    main_proc(args)