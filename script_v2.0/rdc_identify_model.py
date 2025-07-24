import torch
import pickle
import time 
import logging
import argparse
import pprint
import os
import yaml

from model_def import build_model_class
from data_utils import CustomDataset
from model_train import train_a_model

def retrain_with_stop_words(custom_args, snp_token_str_rm):
    logging.info('Re-training with adding stop list')
    logging.info('--------------------------------')
    train_conf = custom_args.conf['train_param']
    data_conf = custom_args.conf['data']
    utils_conf = custom_args.conf['utils']
    
    device_str = train_conf.get('cuda', 'cuda:0' if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    logging.info(f"Device: {device}")
    
    dataset_with_stop_words = CustomDataset(
        data_conf['train_data'],
        stop_words_list=snp_token_str_rm,
        mit_flag=data_conf['mit_flag']
    )
    
    model_class = build_model_class(train_conf['model_flag'])
    
    model = train_a_model(
        device, 
        model_class, 
        dataset_with_stop_words, 
        train_conf
    )
    
    logging.info('Saving the final model and vectors')
    checkpoint_dir = utils_conf['checkpoint']
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    torch.save({
        'model_def': model,
        'model_param': model.state_dict(),
        'model_def_name': type(model).__name__
    }, os.path.join(checkpoint_dir, 'model.pt'))
    
    dataset_with_stop_words.save_vocabulary(
        os.path.join(checkpoint_dir, 'token_vec.pickle')
    )
    
    logging.info(f'Model and vocabulary have save as: {checkpoint_dir}')

def main_proc(custom_args):
    conf = custom_args.conf
    logging.info('------\n-----Starting the new model training----\n---------')   
    
    def train_with_stop_snp(custom_args):
        logging.info('loadding the SNP of stoping list')
        stop_words_info = pickle.load(open(conf['data']['stop_snps'], 'rb'))
        retrain_with_stop_words(custom_args, stop_words_info['snp_token_str_rm'])

    method_dict = {
        'train_with_stop_snp': train_with_stop_snp,
        'train': lambda custom_args: retrain_with_stop_words(custom_args, [])
    }

    func = method_dict.get(custom_args.method, lambda x: logging.error('Unimplemented method'))
    func(custom_args)

    logging.info('------\n-----Model training is completed----\n---------')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='dg_classify')
    parser.add_argument('--method', type=str, default='train_with_stop_snp')
    parser.add_argument('--conf', type=str, default='dg_rdg/config.yml')
    parser.add_argument('--test_flag', type=int, default=0)

    args = parser.parse_args()

    with open(args.conf, 'r') as _file:
        conf = yaml.safe_load(_file)
    args.conf = conf
    
    utils_conf = conf['utils']
    logging.basicConfig(
        filename=utils_conf['logger_file'],
        level=utils_conf['logging_level'],
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    model_dir = utils_conf['model_dir']
    checkpoint = utils_conf['checkpoint']
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(checkpoint, exist_ok=True)
    
    logging.info(f'Configuration:\n{pprint.pformat(conf)}')
    
    main_proc(args)