
import logging
import time 
import torch
from model_def import build_model_class
import argparse
import os

def pack_model(model_path, model_class, checkpoint):
    model_class = build_model_class(model_class)
    model_restore = model_class()
    _load_state = torch.load(model_path, map_location=torch.device('cpu'))
    model_restore.load_state_dict(_load_state)

    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)

    torch.save({
        'model_def' : model_restore
        ,'model_param' : model_restore.state_dict()
        ,'model_def_name' :  type(model_restore).__name__
        }, os.path.join(checkpoint, 'model.pt'))

    pass 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='tools'
    )
    parser.add_argument('--model_path', type=str, default='exp_debug_model/debug_model_0612')
    parser.add_argument('--model_class', type=str, default='snp_dnn_lr')
    parser.add_argument('--checkpoint', type=str, default='exp_debug_model/new_format')
    args = parser.parse_args()
    pack_model(args.model_path, args.model_class, args.checkpoint)

