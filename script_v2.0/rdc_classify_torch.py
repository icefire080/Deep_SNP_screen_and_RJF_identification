import torch
from torch import nn 
from torch.utils.data import DataLoader
import pickle
import numpy as np
from sklearn.model_selection import KFold
from sklearn import metrics
import time 
import logging
import argparse
import pprint
import numpy as np
import os
from utils import build_snp_token_from_str, timer_simple, get_snp_emb_table, sparse_regular, mask_to_zero, remove_snps_by_emb, remove_snps_by_mask, find_all_snp, core_diff_mask_weight, get_diff_mask_weight_base
from model_def import Snp_transform, build_model_class
from data_utils import CustomDataset, SimpleDataset
import optuna
from functools import partial, cmp_to_key
import optuna
import copy


# from torchsummary import summary
# from torch_demo.attention import ScaledDotProductAttention



@timer_simple
def do_train(data_loader, model, loss_fn, optimizer, print_batches = 30, regular_weight=1e-5
             , device=None, test_flag=0):
    logging.debug('dataset_batches->{}'.format(len(data_loader)))
    for num_batches, (x, y) in enumerate(data_loader):
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        pred = torch.squeeze(pred)
        y = torch.squeeze(y)
        loss = loss_fn(pred, y) 

        regular_loss = 0
        if regular_weight > 0:
            emb_param = get_snp_emb_table(model)
            if emb_param is not None:
                regular_loss = sparse_regular(emb_param, weight=1)
        total_loss = loss + regular_weight*regular_loss
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        class_1 = torch.count_nonzero(y).item()
        if num_batches%print_batches == 0:
            logging.info('after num_batches->{}, regular_loss->{:4f} loss -> {:4f}, class_1 -> {}'.format(
                num_batches, regular_loss, loss, class_1))

        if test_flag > 0 and num_batches > 3:
            break

def do_test(data_loader, model, loss_fn, mask = set(), device=torch.device('cpu')):
    logging.debug('do test device {}'.format(device))
    mask_flag = False
    model.eval()
    total_loss, correct = 0, 0
    size = len(data_loader.dataset)
    # logging.info('te4st_size -> {}'.format(size))

    all_pred, all_y = [], []
    with torch.no_grad():
        for num, (x, y) in enumerate(data_loader) :
            x = x.to(device)
            y = y.to(device)
            # logging.debug('in test x {}'.format(x.shape))
            if mask:
                mask_set = torch.from_numpy(np.array([i for i in mask]))
                x = mask_to_zero(x, mask_set)
                mask_flag = True
            pred = model(x)
            y = y.reshape(-1)
            loss = loss_fn(pred, y)
            pred = pred.to('cpu')
            y = y.to('cpu')
            # logging.debug('batch no {}'.format(num))
            # logging.debug('pred {}'.format(pred))
            total_loss += loss
            _correct =  (pred.argmax(1) == y).type(torch.float).sum().item()
            correct += _correct
            all_pred.append(pred[:,1].reshape(-1, 1))
            all_y.append(y.reshape(-1, 1))
    
    all_pred = torch.concat(all_pred)
    all_y = torch.concat(all_y)
    logging.info('test pred sample {}'.format(all_pred[10:30,:].squeeze().numpy()))
    pos_cnt = sum(all_y)
    auc_metric = metrics.roc_auc_score(all_y, all_pred)
    suggest_threshold = 0.5
    all_pred_bin = torch.where(all_pred > suggest_threshold, torch.tensor(1, dtype=torch.long), torch.tensor(0, dtype=torch.long))
    recall, precision = metrics.recall_score(all_y, all_pred_bin, zero_division=0), metrics.precision_score(all_y, all_pred_bin, zero_division=0)
    f1_score = metrics.f1_score(all_y, all_pred_bin, zero_division=0)
    if mask_flag: logging.info('use mask token to pred')
    logging.info('test_size->{:d}, pos_cnt->{:d}, avg_loss -> {:.4f}, avg_correct -> {:.4f}, auc->{:.4f}, suggest_threshold -> {:.4f}, recall->{:.4f}, precision->{:.4f}, f1_score->{:.4f}'.format(
        len(all_pred), pos_cnt.item(), loss/num, correct/size, auc_metric, suggest_threshold, recall, precision, f1_score))
    return auc_metric

def train_and_find_snp(custom_args):
    logging.info('**** method {} *****'.format(train_and_find_snp))
    logging.info('--------------------------------')

    model_class = build_model_class(custom_args.model_flag)    
    device = torch.device(custom_args.cuda if torch.cuda.is_available() else "cpu")
    
    snp_model = model_class()
    snp_model = snp_model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(snp_model.parameters(), lr=custom_args.lr)
    dataset = CustomDataset(custom_args.train_data
                            , filter_breed = custom_args.snp_select_model_filter_breed
                            , mit_flag=custom_args.mit_flag)
    train_loader = DataLoader(dataset=dataset, batch_size=custom_args.batch_size)
    
    start_time = time.time()
    snp_model.train()
    for n_epoch in range(custom_args.epoch):
        logging.info('begin epoch -> {}'.format(n_epoch))
        do_train(train_loader, snp_model, loss_fn, optimizer
                    , regular_weight=custom_args.regular_weight
                    , device=device
                    , test_flag=custom_args.test_flag) 
    end_time = time.time()
    logging.info('use_all_data, train_size->{}, train process cost time -> {:.5f}'.format(len(dataset)  , end_time - start_time))

    trained_params = snp_model.state_dict()
    logging.info('find lower weight token and save')
    snp_token_str_emb_table = trained_params.get('emb.weight').cpu().numpy()
        
    idx_rm, snp_token_str_rm, snp_rm = remove_snps_by_emb(snp_token_str_emb_table
                                                            , reserve_cnt=custom_args.reserve_cnt
                                                            , init_snp_emb_param=None
                                                            , snp_2_idx = dataset.vectorizer.vocabulary_
                                                            , random_remove=custom_args.random_remove_snp
                                                            )
    pickle.dump({'idx_rm': idx_rm, 'snp_rm':snp_rm, 'snp_token_str_rm':snp_token_str_rm}, open(model_path + '.rm_info', 'wb'))
    return snp_token_str_rm

def retrain_with_stop_words(custom_args, snp_token_str_rm):
    device = torch.device(custom_args.cuda if torch.cuda.is_available() else "cpu")

    logging.info('add stop_words to retrain')
    logging.info('--------------------------------')
    dataset_with_stop_words = CustomDataset(custom_args.train_data
                                            , stop_words_list=snp_token_str_rm
                                            , mit_flag = custom_args.mit_flag)
    
    model_class = build_model_class(custom_args.model_flag)
    
    snp_model_2 = model_class()
    snp_model_2 = snp_model_2.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(snp_model_2.parameters(), lr=custom_args.lr)
    train_loader = DataLoader(
                dataset=dataset_with_stop_words,
                batch_size=custom_args.batch_size,
                drop_last=True
            )
    start_time = time.time()
    snp_model_2.train()
    for n_epoch in range(custom_args.epoch):
        logging.info('begin epoch -> {}'.format(n_epoch))
        do_train(train_loader, snp_model_2, loss_fn, optimizer
                    , regular_weight=0.0
                    , device=device
                    , test_flag=custom_args.test_flag) 
    end_time = time.time()
    snp_model_2 = snp_model_2.to(torch.device('cpu'))
    logging.info('save final model and vect')
    torch.save(snp_model_2.state_dict(), custom_args.model_path)
    dataset_with_stop_words.save_vocabulary(custom_args.token_vec_file)

def train_a_model(device, model_class, dataset, custom_args):
    snp_model_2 = model_class()
    snp_model_2 = snp_model_2.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(snp_model_2.parameters(), lr=custom_args.lr)
    train_loader = DataLoader(
                dataset=dataset,
                batch_size=custom_args.batch_size,
                drop_last=True
            )
    start_time = time.time()
    snp_model_2.train()
    for n_epoch in range(custom_args.epoch):
        logging.info('begin epoch -> {} in method {}'.format(n_epoch, train_a_model))
        do_train(train_loader, snp_model_2, loss_fn, optimizer
                    , regular_weight=0.0
                    , device=device
                    , test_flag=custom_args.test_flag) 
    end_time = time.time()
    return snp_model_2


def train_and_find_snp_method_2(custom_args):
    logging.info('in method {}'.format(train_and_find_snp_method_2))
    dataset = CustomDataset(custom_args.train_data, filter_breed = custom_args.snp_select_model_filter_breed, mit_flag= custom_args.mit_flag)
    train_loader = DataLoader(dataset, batch_size=custom_args.batch_size)
    device = torch.device(custom_args.cuda if torch.cuda.is_available() else "cpu")
    model_class = build_model_class(custom_args.model_flag)
    logging.info('init model {}'.format(model_class))
    model = model_class()
    logging.info('move model to device {}'.format(device))
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=custom_args.lr)
    model.train()
    for n_epoch in range(custom_args.epoch):
        logging.info('begin epoch -> {}'.format(n_epoch))
        do_train(train_loader, model, loss_fn, optimizer
                    , regular_weight=custom_args.regular_weight
                    , device=device
                    , test_flag=custom_args.test_flag) 
    # model = model.to(torch.device('cpu'))
    idx_rm, snp_rm, snp_token_str_rm = remove_snps_by_mask(model, dataset, snp_2_idx = dataset.vectorizer.vocabulary_, device = device, custom_args=custom_args )
    pickle.dump({'idx_rm': idx_rm, 'snp_rm':snp_rm, 'snp_token_str_rm':snp_token_str_rm}, open(model_path + '.rm_info', 'wb'))
    return snp_token_str_rm

def extrac_pos(x, y, neg_sample_idx, pos_sample_idx):
    idx = y.squeeze().nonzero()[0]
    all_idx = np.concatenate([idx, neg_sample_idx])
    all_idx = all_idx.astype(np.int64)
    x_pos = x[all_idx, :]
    y_pos = y[all_idx, :]
    board = len(idx)
    return x_pos, y_pos, board, all_idx

def mask_snp_token_parse(mask_token_set, snp_2_str, vocabulary):
    mask_snp_token_str_set = list()
    for snp in mask_token_set:
        mask_snp_token_str_set.extend( snp_2_str.get(snp, []) )
    mask_snp_token_str_set = list(set(mask_snp_token_str_set))
    a2 = [(vocabulary.get(x, 0), x) for x in mask_snp_token_str_set]
    return a2

def train_and_find_snp_method_3(custom_args, stop_words = []):
    logging.info('in method {}'.format(train_and_find_snp_method_3))
    all_stop_snp = []
    for i in range(custom_args.max_find_epoch):
        logging.info('begin epoch {}, stop_snp_cnt {}, stop_snp {}'.format(i, len(all_stop_snp), all_stop_snp[:10]))
        stop_snp, local_stop_words, pre_result = find_max_sub_snp_list(custom_args, stop_words )
        stop_words.extend(local_stop_words)
        all_stop_snp.extend(stop_snp)
        pickle.dump({'idx_rm': [12], 'snp_rm':all_stop_snp, 'snp_token_str_rm':stop_words}, open(model_path + '.rm_info.epoch{}'.format(i), 'wb'))

    # pickle.dump({'idx_rm': [12], 'snp_rm':all_stop_snp, 'snp_token_str_rm':stop_words}, open(model_path + '.rm_info', 'wb'))


def find_max_sub_snp_list(custom_args, stop_words):
    dataset = CustomDataset(custom_args.train_data
                            , filter_breed = custom_args.snp_select_model_filter_breed
                            , mit_flag= custom_args.mit_flag
                            , stop_words_list=stop_words)
    
    all_snp, snp_2_str = find_all_snp(dataset.vectorizer.vocabulary_.keys())
    device = torch.device(custom_args.cuda if torch.cuda.is_available() else "cpu")
    model_class = build_model_class(custom_args.model_flag)
    model = train_a_model(device, model_class, dataset, custom_args)
    cal_cnt = 0
    result = []
    pred_all_base, y_all_base = get_diff_mask_weight_base([(1,1)], model, SimpleDataset(dataset.x, dataset.y), device=device, custom_args=custom_args)
    pred_base = torch.concat([pred_all_base, y_all_base], dim=1).numpy()
    all_sample = len(pred_base)
    pos_cnt = len(pred_base[:,1].nonzero()[0])

    idx = np.arange(0, all_sample).reshape((-1, 1))
    pred_base = np.concatenate( [pred_base, idx], axis=1)
    l1 = [x for x in pred_base]
    def cmp(x, y):
        if x[1] != y[1]:
            a, b = x[1], y[1]
        else:
            a, b = x[0], y[0]
        return 1 if a > b else -1 if a < b else 0

    l1.sort(key=cmp_to_key(cmp))
    l1 = [np.reshape(x, [1, -1])  for x in l1 ]
    pred_base_sort = np.concatenate(l1, axis=0)
    
    neg_borad = np.arange(all_sample-pos_cnt-1, all_sample-pos_cnt-6, -1)
    neg_board_sample_idx = pred_base_sort[neg_borad, -1]
    pos_sample_idx = pred_base_sort[all_sample-pos_cnt:, -1]

    valid_x, valid_y , board, all_idx = extrac_pos(dataset.x, dataset.y, neg_board_sample_idx, pos_sample_idx)
    valid_pred_raw = pred_base[all_idx,0]

    for candidate_snp in range(len(all_snp)):
        sub_snp_list = copy.deepcopy(all_snp)[candidate_snp:]
        mask_token_set = set()
        mask_token_set.add(sub_snp_list[0])
        for window in range(1, len(sub_snp_list)):
            one = sub_snp_list[window]
            mask_token_set.add(one) 
            a2 = mask_snp_token_parse(mask_token_set, snp_2_str, dataset.vectorizer.vocabulary_)
            diff_score, score_detail = core_diff_mask_weight(valid_pred_raw, a2, model, SimpleDataset(valid_x, valid_y), device=device, custom_args=custom_args)
            cal_cnt += 1
            if diff_score > 0.5: mask_token_set.remove(one)
            if len(mask_token_set) > custom_args.find_step: 
                logging.info('early stop inner')
                break

            if diff_score > 1:
                logging.info('early stop inner base too weak {}'.format(score_detail))
                break
        
        logging.info('for left token {}, idx {}, set_len {}, find mask_token_set {}'.format(sub_snp_list[0], candidate_snp, len(mask_token_set), list(mask_token_set)[:6]))

        if diff_score > 1:
            logging.info('early stop final base too weak {}'.format(score_detail))
            break

        if len(mask_token_set) == 1:
            a2 = mask_snp_token_parse(mask_token_set, snp_2_str, dataset.vectorizer.vocabulary_)
            diff_score, score_detail = core_diff_mask_weight(valid_pred_raw, a2, model, SimpleDataset(valid_x, valid_y), device=device, custom_args=custom_args)
            if diff_score > 0.5: continue

        result.append( (candidate_snp, mask_token_set,  diff_score, score_detail))
        if len(mask_token_set) > custom_args.find_step: 
            logging.info('early stop final')
            break

        

    if len(result) == 0: return [], [], []
    result.sort(key=lambda x: len(x[1]), reverse=True)
    detail_str_f = lambda x: '{},len {}, diff_score {}, score_detail {}'.format(x[0], len(x[1]) , x[2] , x[3]) 
    top_one = detail_str_f(result[0])
    last_one = detail_str_f(result[-1])

    logging.info('in {} result len {}, result_0 {}, result_-1 {}'.format(
        find_max_sub_snp_list, len(result), top_one, last_one))
    final_mask = result[0][1]
    final_mask_snp_token_str = []
    for snp in final_mask:
        final_mask_snp_token_str.extend( snp_2_str.get(snp, []) )
    return final_mask, final_mask_snp_token_str, result


def main_proc(custom_args):
    device = torch.device(custom_args.cuda if torch.cuda.is_available() else "cpu")
    print((logger_file, custom_args.logging_level, device, custom_args.method))
    logging.basicConfig(filename=logger_file, level=custom_args.logging_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logging.info('------\n-----begin a new exp----\n---------')
    logging.info(custom_args)
    hyperparams = {'epoch':custom_args.epoch, 'lr':custom_args.lr
                   , 'batch_size': custom_args.batch_size, 'regular_weight':custom_args.regular_weight
                   , 'model_flag': custom_args.model_flag}
    logging.info(pprint.pformat(hyperparams))
    logging.info('is cuda {}'.format(torch.cuda.is_available()))
    logging.info('train_data_file ->{}, model_file->{}'.format(custom_args.train_data, custom_args.model_path))
    # train_data_file = 'data/GGS_vs_DC.pickle'

    model_class = build_model_class(custom_args.model_flag)

    logging.debug('init model, loss, opt, dataset Object')
    # device = torch.device("cpu")
    logging.info('use {} device'.format(device))
    if custom_args.method == 'optuna':
        logging.info('optuna find hyperparam')
        dataset = CustomDataset(custom_args.train_data)
        core_f = partial(objective_core, custom_args=custom_args, device=device
                         , model_class=model_class, dataset=dataset)
        study = optuna.create_study()
        study.optimize(core_f, n_trials=3*3*6, show_progress_bar=True)
        print(study.best_params)
    elif custom_args.method == 'kfold_valid':
        dataset = CustomDataset(custom_args.train_data)
        core_f = partial(objective_core, custom_args=custom_args, device=device
                         , model_class=model_class, dataset=dataset)
        metric1 = core_f(None)
        logging.info('finish kfold, final metric {}'.format(metric1))
    elif custom_args.method == 'norm':
        snp_token_str_rm = train_and_find_snp(custom_args)
        custom_args.model_flag = 'dnn'
        retrain_with_stop_words(custom_args, snp_token_str_rm)
    elif custom_args.method == 'only_retrain':
        stop_words_info = pickle.load(open(model_path + '.rm_info', 'rb'))
        retrain_with_stop_words(custom_args, stop_words_info['snp_token_str_rm'])
    elif custom_args.method == 'only_find_snp':
        snp_token_str_rm = train_and_find_snp(custom_args)
    elif custom_args.method == 'mask_weight':
        snp_token_str_rm = train_and_find_snp_method_2(custom_args)
    elif custom_args.method == 'only_train':
        retrain_with_stop_words(custom_args, [])
    elif custom_args.method == 'mask_weight_2':
        train_and_find_snp_method_3(custom_args)
    elif custom_args.method == 'mask_weight_with_init':
        stop_words_info = pickle.load(open(model_path + '.rm_info', 'rb'))
        train_and_find_snp_method_3(custom_args, stop_words=stop_words_info['snp_token_str_rm'])
    else:
        logging.info('no this method')
        
    logging.info('------\n-----end a new exp----\n---------')


def objective_core(trial: optuna.trial.Trial, custom_args, device, model_class, dataset):
    if trial is not None:
        epoch = trial.suggest_int('epoch', 10, 40, step=5)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128]) 
        regular_weight = trial.suggest_categorical('regular_weight', [1e-3, 1e-2, 1e-4])
        # optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
        logging.info('optuna epoch {}, batch_size {}, regular_weight {}'.format(
            epoch, batch_size, regular_weight))
    else:
        epoch = custom_args.epoch
        batch_size = custom_args.batch_size
        regular_weight = custom_args.regular_weight

    # dataset = CustomDataset(train_data_file, stop_words_file='default_20240522_1439.rm_info')
    # init_snp_emb_param = np.copy( get_snp_emb_table(snp_model).cpu().detach().numpy() ) 
    kf = KFold(n_splits=custom_args.kfold, shuffle=True)
    valid_metrics = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        snp_model = model_class()
        snp_model = snp_model.to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(snp_model.parameters(), lr=custom_args.lr)
        train_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(train_idx)
        )
        test_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(test_idx)
            ,drop_last=True
        )

        start_time = time.time()
        snp_model.train()
        for epoch_no in range(epoch):
            logging.info('begin epoch -> {}'.format(epoch_no))
            do_train(train_loader, snp_model, loss_fn, optimizer
                        , regular_weight=regular_weight
                        , device=device
                        , test_flag=custom_args.test_flag) 
        end_time = time.time()
        logging.info('train_size->{}, train process cost time -> {:.5f}'.format(len(train_idx)  , end_time - start_time))
        
        test_device = torch.device('cpu')
        logging.info('trained model back to cpu')
        snp_model = snp_model.to(test_device)
        logging.info('test_size->{}'.format(len(test_idx)))
        logging.info('norm test use all token')
        valid_metric = do_test(test_loader, snp_model, loss_fn, device=test_device)

        trained_params = snp_model.state_dict()
        logging.info('find lower weight token and save')
        snp_token_str_emb_table = trained_params.get('emb.weight').cpu().numpy()
        idx_rm, snp_token_str_rm, snp_rm = remove_snps_by_emb(snp_token_str_emb_table, reserve_cnt=custom_args.reserve_cnt, init_snp_emb_param=None, snp_2_idx = dataset.vectorizer.vocabulary_)
        # pickle.dump({'idx_rm': idx_rm, 'snp_rm':snp_rm, 'snp_token_str_rm':snp_token_str_rm}, open(model_path + '.rm_info', 'wb'))

        logging.info('mask remove snp_token')
        valid_metric_2 = do_test(test_loader, snp_model, loss_fn, mask=set(idx_rm), device=test_device)
        logging.info('finish fold ->{}'.format(fold))
        logging.info('--------------------')


        valid_metrics.append(valid_metric)

        if fold + 1 > custom_args.max_fold:
            break
    
    return -sum(valid_metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='dg_classify'
    )

    current_time_str = time.strftime("%Y%m%d_%H%M", time.localtime())
    parser.add_argument('train_data')
    parser.add_argument('logger_file')
    parser.add_argument('--logging_level', type=int, default=12)
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--model_path', type=str, default= 'default_{}'.format(current_time_str) )
    parser.add_argument('--token_vec_file', type=str, default= 'default_token_vec_{}'.format(current_time_str))
    parser.add_argument('--model_flag', type=str, default= 'dnn')
    parser.add_argument('--regular_weight', type=float)
    parser.add_argument('--test_flag', type=int, default=0)
    parser.add_argument('--model_dir', type=str, default='./' )
    parser.add_argument('--kfold', type=int, default='6' )
    parser.add_argument('--save_model', type=int, default= 0 )
    parser.add_argument('--restore_model_double_test', type=bool, default= False )
    parser.add_argument('--max_fold', type=int, default= 2 )
    parser.add_argument('--method', type=str, default='norm')
    parser.add_argument('--reserve_cnt', type=int, default=100 )
    parser.add_argument('--random_remove_snp', type=int, default=0 )
    parser.add_argument('--cuda', type=str, default='cuda:3' )
    parser.add_argument('--mix_after_rm', type=int, default=0 )
    parser.add_argument('--mit_flag', type=int, default=0 )
    parser.add_argument('--snp_select_model_filter_breed', type=str, default='' )
    parser.add_argument('--max_find_epoch', type=int, default=20 )
    parser.add_argument('--mask_diff_th', type=float, default=0.96 )
    parser.add_argument('--find_step', type=int, default=200 )




    args = parser.parse_args()
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    model_path = os.path.join(args.model_dir, args.model_path)
    token_vec_file = os.path.join(args.model_dir, args.token_vec_file)
    logger_file = os.path.join(args.model_dir, args.logger_file)
    args.model_path = model_path
    args.token_vec_file = token_vec_file
    args.logger_file = logger_file
    print(args)
    main_proc(args)

