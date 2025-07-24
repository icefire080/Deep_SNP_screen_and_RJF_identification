import logging
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn import metrics
from enum import Enum
import random 


def build_mask_from_token_ids(x, mask_token_id_list):
    mask = torch.logical_not(torch.ones_like(x)) 
    for mask_token_id in mask_token_id_list:
        a = torch.ones_like(x) * mask_token_id
        mask_tmp = torch.eq(x, a)
        mask = torch.logical_or(mask, mask_tmp)
    return mask


class Pred_method(Enum):
    ONLY_PRED_ORI = 1
    ONLY_PRED_MASK = 2
    PRED_MASK_ORI = 3


def _core_mask_pred_diff(token_id_str_pairs, model, valid_dataset, device=None, ori_or_mask=Pred_method.PRED_MASK_ORI):
    model.eval()
    all_pred, all_y, all_pred_mask, all_mask_valid = [], [], [], []
    test_loader = DataLoader(valid_dataset, batch_size=11)
    mask_token_id_list = [mask_token_id for mask_token_id, mask_token_str in token_id_str_pairs]
    with torch.no_grad():
        for num, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            
            if ori_or_mask in (Pred_method.ONLY_PRED_ORI, Pred_method.PRED_MASK_ORI):
                pred = model(x)
                all_pred.append(pred[:,1].reshape(-1, 1))

            if ori_or_mask in (Pred_method.ONLY_PRED_MASK, Pred_method.PRED_MASK_ORI):
                mask = build_mask_from_token_ids(x, mask_token_id_list)
                x1 = x.masked_fill(mask, 0)
                mask_valid = torch.sum(mask, dim=1)
                pred_mask = model(x1)
                all_pred_mask.append(pred_mask[:,1].reshape(-1, 1))
                all_mask_valid.append(mask_valid.reshape(-1, 1))

            all_y.append(y.reshape(-1, 1))

    cpu_device = torch.device('cpu')
    if len(all_pred) > 0:
        all_pred = torch.concat(all_pred).to(cpu_device)
    if len(all_pred_mask) > 0:
        all_pred_mask = torch.concat(all_pred_mask).to(cpu_device)
    if len(all_mask_valid) > 0:
        all_mask_valid = torch.concat(all_mask_valid).to(cpu_device)
    all_y = torch.concat(all_y).to(cpu_device)
    return all_pred, all_pred_mask, all_mask_valid, all_y


def get_diff_mask_weight_base(model, valid_dataset, device=None):
    all_pred, _, _, all_y = _core_mask_pred_diff([], model, valid_dataset, device=device, ori_or_mask=Pred_method.ONLY_PRED_ORI )
    logging.info('this model base cnt {}, all_pred {}, min {}'.format(all_pred.shape ,torch.mean(all_pred)
                , torch.min(all_pred)) )
    return all_pred, all_y



def core_diff_mask_weight(all_pred, token_id_str_pairs, model, valid_dataset, device=None, search_param = None):
    th = search_param.get('mask_diff_th')
    auc_th = search_param.get('mask_diff_base_model_auc_th')
    if search_param.get('test_flag') : th = 0.4562
    all_pred = torch.tensor(all_pred).reshape((-1,1))
    _, all_pred_mask, _, all_y = _core_mask_pred_diff(token_id_str_pairs, model, valid_dataset, device=device, ori_or_mask= Pred_method.ONLY_PRED_MASK)
    pair_pred = torch.concat([all_pred, all_pred_mask, all_y], dim=1).numpy()
    
    ORI, MASK, LABEL = 0, 1, 2
    pos_sample_cnt = int(sum(pair_pred[: , LABEL]))
    auc = metrics.roc_auc_score(pair_pred[:, LABEL], pair_pred[:, ORI])
    auc_mask = metrics.roc_auc_score(pair_pred[:,LABEL], pair_pred[:, MASK])
    auc_all = [auc, auc_mask]
    
    board_neg_sample = pair_pred[pos_sample_cnt:]
    board_pos_sample = pair_pred[:pos_sample_cnt]
    pos_lg_th_cnt = np.sum(board_pos_sample > th, axis=0)

    sort_idx = np.argsort(board_pos_sample[:,0])
    board_pos_sample = board_pos_sample[sort_idx]
    board_pos_sample = board_pos_sample[:4]
    board_pos_sample_avg = np.average(board_pos_sample, axis=0)
    neg_sample_avg = np.average(board_neg_sample, axis=0)
    
    

    # score = 0 if min(all_pred_mask - th) > 0 else 1

    cond_0 = pos_lg_th_cnt[ORI] -pos_lg_th_cnt[MASK] >  1 \
        or board_pos_sample_avg[ORI]/board_pos_sample_avg[MASK] > 1.2 \
        or auc_all[ORI] - auc_all[MASK] > 2e-3 \
        or neg_sample_avg[MASK] / neg_sample_avg[ORI] > 1.2
    
    # or all_pred_mask_lg_th_cnt < int(pos_sample*0.95) \

    score = 1 if cond_0 else 0
    score = 2 if auc < auc_th else score
    score = random.randint(0,1) if search_param.get('test_flag', False)  else score

    score_detail = '----in method {}, for mask token {}, len {}, diff score is {}, all_pred.shape {}, all_pred {:.3}, min {:.3}, mask_all_pred {:.3}, min {:.3}, th {}, all_pred_lg_th_cnt {}, all_pred_mask_lg_th_cnt {}, pos_board {}, neg_sample {}, auc {:.4}, mask_auc {:.4}, board_pos_sample_avg {}, neg_sample_avg {}  ----'.format(
                        core_diff_mask_weight
                        , [x[0] for x in token_id_str_pairs][:4]
                        , len(token_id_str_pairs), score
                        , all_pred.shape 
                        , torch.mean(all_pred)
                        , torch.min(all_pred), torch.mean(all_pred_mask), torch.min(all_pred_mask)
                        , th, pos_lg_th_cnt[ORI], pos_lg_th_cnt[MASK]
                        , board_pos_sample, board_neg_sample, auc, auc_mask,board_pos_sample_avg, neg_sample_avg)
    if np.random.randn() > 2.5:
        logging.info(score_detail)
    
    return score, score_detail