
import logging
import time 
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader
from sklearn import metrics
from enum import Enum
import random 
from functools import cmp_to_key
from collections import namedtuple


def timer_simple(func):
    def func_wrap(*args, **kwargs):
        time_start = time.time()
        result = func(*args, **kwargs)
        time_end = time.time()
        logging.info('{} func cost {}'.format(func.__name__, time_end-time_start))
        return result
    return func_wrap


def build_snp_token_from_str(inp):
    _tmp = lambda x: x.split(':')
    r = [ _tmp(x) for x in inp.split(',')]
    r = {kv[0]:kv[1] for kv in r}
    obj = Snp_token(r['chrom'], r['pos'], r['alt'], r['ref'], r['gt'])
    return obj


class Snp_token():
    def __init__(self, chrom, pos, alt, ref, gt) -> None:
        self.chrom = chrom 
        self.pos = int(pos) 
        self.alt = alt
        self.ref = ref 
        self.gt = gt
        self.relative_token_str_pair = []
        self.relative_token_ids = set()
        self.weight = 0

    def __str__(self):
        l = []
        for k,v in zip(['alt', 'chrom', 'gt', 'pos', 'ref'], [self.alt, self.chrom, self.gt, self.pos, self.ref]):
            l.append('{}:{}'.format(k, v))
        return ','.join(l)

    def __lt__(self, other):
        if self.chrom == other.chrom:
            return self.pos < other.pos
        else:
            return self.chrom < other.chrom
    
    def __repr__(self) -> str:
        return str(self)

    def get_snp(self):
        return 'chrom:{},pos:{}'.format(self.chrom, self.pos)


def rand_choose_snp_from_corpus(corpus):
    all_snp_token = [build_snp_token_from_str(snp_token_str) for snp_token_str in corpus]
    all_snp = set(snp_token.get_snp()  for snp_token in all_snp_token)


def sparse_regular(emb_weight, weight=1):
    emb_table = emb_weight
    emb_table_1 = torch.norm(emb_table, dim=1)
    # nonzero = torch.where(emb_table_1 > 0.0002, torch.tensor(1, dtype=torch.float32), torch.tensor(0, dtype=torch.float32))
    nonzero_cnt = sum(emb_table_1)
    regular_loss = nonzero_cnt * weight
    return regular_loss

def remove_snps_by_emb(emb_table_np, reserve_cnt=300, init_snp_emb_param=None, snp_2_idx={}, random_remove = 0):
    logging.info('reserve {} snp'.format(reserve_cnt))
    snp_weight = np.linalg.norm(emb_table_np, axis=1)
    if init_snp_emb_param:
        snp_weight_init = np.linalg.norm(init_snp_emb_param, axis=1)

    ##reverse map
    idx_2_snp = {}
    for snp_token_str, idx in snp_2_idx.items():
        if snp_token_str.startswith('mit'): continue
        idx_2_snp[idx] = snp_token_str
    
    snp_all_weight = defaultdict(float)
    snp_all_freq = defaultdict(int)
    for idx, w in enumerate(snp_weight):
        if idx not in idx_2_snp: continue
        snp_token_str = idx_2_snp[idx]
        snp_token = build_snp_token_from_str(snp_token_str)
        snp_all_weight[snp_token.get_snp()] += w
        snp_all_freq[snp_token.get_snp()] += 1 

    snp_all_weight_list = [(k, v) for k, v in snp_all_weight.items()]
    snp_all_weight_list.sort(key = lambda x:x[1])
    logging.info('avg snp weight {}, threshold weight, {}'.format(
        np.average([ v for k, v in snp_all_weight_list ])
        ,snp_all_weight_list[-reserve_cnt]
        ))
    logging.info('all_snp_has_weight cnt->{}, reserve_cnt-{}'.format(len(snp_all_weight_list), reserve_cnt))
    
    snp_all_weight_remove = snp_all_weight_list[1: len(snp_all_weight_list) - reserve_cnt]
    snp_all_weight_remove_reverse_weight = snp_all_weight_list[-len(snp_all_weight_list) + reserve_cnt: -1]
    np.random.shuffle(snp_all_weight_list)
    snp_all_weight_remove_rnd = snp_all_weight_list[1: len(snp_all_weight_list) - reserve_cnt]
    

    if random_remove == 1:
        logging.info('use rnd weight choose SNP')
        snp_all_weight_remove = snp_all_weight_remove_rnd
    elif random_remove == 2:
        logging.info('use reverse weight choose SNP')
        snp_all_weight_remove = snp_all_weight_remove_reverse_weight
    else:
        pass

    logging.info('actual remove snp info start {}, end {}, avg-weight {}, max-weight {}'.format(
        snp_all_weight_remove[0], snp_all_weight_remove[-1]
        ,np.average([ x[1] for x in snp_all_weight_remove])
        ,np.max([ x[1] for x in snp_all_weight_remove])
    ))

    snp_all_weight_remove_set = set(x[0] for x in snp_all_weight_remove)

    emb_idx_remove, snp_token_str_remove = [], []
    for idx, w in enumerate(snp_weight):
        if idx not in idx_2_snp:
            continue
        snp_token_str = idx_2_snp[idx]
        snp_token = build_snp_token_from_str(snp_token_str)
        if snp_token.get_snp() in snp_all_weight_remove_set:
            emb_idx_remove.append(idx)
            snp_token_str_remove.append(snp_token_str)

    return emb_idx_remove, snp_token_str_remove, snp_all_weight_remove_set  

def find_all_snp(snp_token_strs):
    snp_2_tokens = defaultdict(list)
    snp_set = set()
    filter_func1 = lambda x: not x.startswith('mit')
    l1 = filter(filter_func1, snp_token_strs)
    l2 = set(map(lambda x: build_snp_token_from_str(x), l1))
    for snp_token in l2:
        snp = snp_token.get_snp()
        snp_2_tokens[snp].append(str(snp_token))
        snp_set.add(snp)
    snp_list = list(snp_set)

    Snp = namedtuple('Snp', ('chr', 'pos'))
    def parse_snp(x):
        d = dict(parse_kv_str(x, ',', ':'))
        one_snp = Snp(d['chrom'], d['pos'])
        return one_snp
    snp_list_2 = [parse_snp(x) for x in snp_list]
    def cmp(snp_a, snp_b):
        if snp_a.chr != snp_b.chr:
            a, b = snp_a.chr, snp_b.chr
        else:
            a, b = snp_a.pos, snp_b.pos
        return 1 if a > b else -1 if a < b else 0
    snp_list_2.sort(key=cmp_to_key(cmp)) ## sort by y/label, pred
    snp_list = [ 'chrom:{},pos:{}'.format(one.chr, one.pos) for one in snp_list_2]    
    return snp_list, snp_2_tokens

def remove_snps_by_mask(model, valid_dataset, device=None, snp_2_idx={}, custom_args = None):
    logging.info('in method {}'.format(remove_snps_by_mask))
    model.eval()
    snp_info = {}

    for snp_token_str, idx in snp_2_idx.items():
        if snp_token_str.startswith('mit'): continue
        snp_token = build_snp_token_from_str(snp_token_str)
        if snp_token.get_snp() in snp_info:
            snp_token = snp_info[snp_token.get_snp()]
        snp_token.relative_token_str_pair.append((idx, snp_token_str))
        snp_info[snp_token.get_snp()] = snp_token
    # chrom:1,pos:45896  ,  chrom:1,pos:10153256
    snp_info_list = []  
    for num, (snp, snp_token) in enumerate(snp_info.items()):
        if num > 100 and custom_args.test_flag == 1:
            weight = 0.2
        else:
            weight = _core_mask_weight(snp_token.relative_token_str_pair , model, valid_dataset, device=device, custom_args=custom_args )
        snp_token.weight = weight
        snp_info_list.append(snp_token)
        if num % 50 == 0:
            logging.info('calculate total {}'.format(num))

    snp_info_list.sort(key=lambda x:x.weight)
    all_snp_token_cnt = len(snp_info_list)
    logging.info('all_snp_token_cnt {}'.format(all_snp_token_cnt))
    emb_idx_remove, snp_token_str_remove, snp_all_weight_remove_set = [], [], set()
    for snp_token in snp_info_list[ :all_snp_token_cnt - custom_args.reserve_cnt]:
        emb_idx_remove.extend(x[0] for x in  snp_token.relative_token_str_pair)
        snp_token_str_remove.extend(x[1] for x in  snp_token.relative_token_str_pair)
        snp_all_weight_remove_set.add(snp_token.get_snp())
    return emb_idx_remove, snp_all_weight_remove_set, snp_token_str_remove

# def _core_mask_weight(token_id_str_pairs, model, valid_dataset, device=None, custom_args = None):
#     all_pred, all_pred_mask, all_mask_valid, all_y = _core_mask_pred_diff(token_id_str_pairs, model, valid_dataset, device=device, custom_args=custom_args)
#     auc =  metrics.roc_auc_score(all_y, all_pred)
#     auc_mask =  metrics.roc_auc_score(all_y, all_pred_mask)
#     all_hit = torch.sum(all_mask_valid).item()
#     weight = auc - auc_mask
#     logging.debug('token_id & str{},  weight {}'.format(token_id_str_pairs, weight))
#     return weight




    

def get_snp_emb_table(model, param_name = 'emb.weight'):
    for name, param in model.named_parameters():
            if name == param_name:
                return param
            else:
                continue
    return None

def mask_to_zero(inp : torch.tensor, mask_set: torch.tensor):
    inp = torch.unsqueeze(inp, dim=1)
    mask_1 = torch.reshape(mask_set, (1, -1, 1))
    mask_1 = torch.tile(mask_1, (1, 1, inp.shape[-1]))
    mask = (mask_1 - inp).bool()
    mask = torch.all(mask, dim=-2, keepdim=True)
    inp_masked = torch.masked_fill(inp, ~mask, 0)
    inp_masked = torch.squeeze(inp_masked)
    return inp_masked

def sort_np_matrix_row_by_col_val(matrx, cols):
    l1 = [x for x in matrx] ## matrxi to list [ (1*n), (1*n) ]

    def cmp(row_a, row_b):
        for col in cols:
            if row_a[col] != row_b[col]:
                a, b = row_a[col], row_b[col]
                break
            elif col == cols[-1]:
                a, b = row_a[col], row_b[col]
        return 1 if a > b else -1 if a < b else 0

    l1.sort(key=cmp_to_key(cmp)) ## sort by y/label, pred

    l1 = [np.reshape(x, [1, -1])  for x in l1 ] ##rebuild 1 * n
    pred_base_sort = np.concatenate(l1, axis=0)
    return pred_base_sort

def parse_kv_str(s, sep, kv_sep):
    result = []
    items = s.split(sep)
    for item in items:
        kvs = item.split(kv_sep)
        k = kvs[0]
        v = kvs[1]
        result.append((k, v))
    return result