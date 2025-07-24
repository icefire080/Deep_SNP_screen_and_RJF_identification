from utils import timer_simple, get_snp_emb_table, sparse_regular
import logging
import torch
import os
import time
from torch import nn 
from torch.utils.data import DataLoader
from collections import Counter
from sklearn.metrics import f1_score
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        if targets.dim() > 1:
            targets = targets.squeeze(1)
        
        BCE_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return loss.mean()

def evaluate(model, data_loader, device=None):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            
            if y.dim() > 1:
                y = y.squeeze(1)
                
            all_preds.extend(pred.argmax(dim=1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    f1 = f1_score(all_labels, all_preds, average='macro')
    return f1

def get_lr_scheduler(optimizer, scheduler_conf):
    scheduler_type = scheduler_conf.get('type', 'none')
    min_lr = float(scheduler_conf.get('min_lr', 1e-6))
    
    if scheduler_type == 'step':
        return lr_scheduler.StepLR(
            optimizer,
            step_size=int(scheduler_conf.get('step_size', 30)),
            gamma=float(scheduler_conf.get('gamma', 0.1))
        )
    elif scheduler_type == 'cosine':
        return lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(scheduler_conf.get('t_max', 100)),
            eta_min=float(scheduler_conf.get('min_lr', min_lr))
        )
    elif scheduler_type == 'plateau':
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_conf.get('mode', 'min'),
            factor=float(scheduler_conf.get('factor', 0.1)),
            patience=int(scheduler_conf.get('patience', 5)),
            min_lr_val = float(scheduler_conf.get('min_lr', min_lr)),
            threshold = float(scheduler_conf.get('threshold', 1e-4))
        )
    elif scheduler_type == 'onecycle':
        max_lr=float(scheduler_conf.get('max_lr', 0.01))
        total_steps=int(scheduler_conf.get('total_steps', 100))
        pct_start=float(scheduler_conf.get('pct_start', 0.3))
        final_div_factor = float(scheduler_conf.get('final_div_factor', 1e3))
        
        base_lr = optimizer.param_groups[0]['lr']
        min_lr_val = base_lr / final_div_factor
        
        if min_lr_val < min_lr:
            min_lr_val = min_lr
        
        return lr_scheduler.OneCycleLR(
            optimizer,
            max_lr = max_lr,
            total_steps = total_steps,
            pct_start =pct_start,
            div_factor = 10, # max_lr / base_lr
            final_div_factor = final_div_factor # base_lr / min_lr
        )

    else:
        logging.warning(f"unknown scheduler type {scheduler_type}, dont use scheduler")
        return None

def do_train(data_loader, model, loss_fn, optimizer, global_step, 
             print_batches=30, regular_weight=1e-5, device=None, test_flag=False):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for x, y in data_loader:
        global_step += 1
        num_batches += 1
        
        x = x.to(device)
        y = y.to(device)
        
        if y.dim() > 1:
            y = y.squeeze(1)
            
        pred = model(x)
        loss = loss_fn(pred, y)
        
        regular_loss = 0
        if regular_weight > 0:
            emb_param = get_snp_emb_table(model)
            if emb_param is not None:
                regular_loss = sparse_regular(emb_param, weight=1)
        
        total_batch_loss = loss + regular_weight * regular_loss
        total_batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += total_batch_loss.item()
        
        if global_step % print_batches == 1:
            y_dist = y.detach().cpu().numpy().flatten()
            y_dist = Counter(y_dist)
            current_lr = optimizer.param_groups[0]['lr']
            
            logging.info(
                f'Step {global_step}: Loss={loss.item():.4f}, '
                f'RegLoss={regular_loss:.4f}, LR={current_lr:.6f}, '
                f'LabelDist={dict(y_dist)}'
            )
        
        if test_flag and num_batches > 3:
            break
    
    return global_step, total_loss / num_batches

def train_a_model(device, model_class, dataset, train_conf):
    save_dir = train_conf.get('save_dir', './checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    
    model = model_class()
    model = model.to(device)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_conf['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=train_conf.get('num_workers', 0)
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=train_conf['batch_size'],
        shuffle=False,
        num_workers=train_conf.get('num_workers', 0)
    )
    
    loss_fn_name = train_conf.get('loss_fn', 'cross_entropy')
    if loss_fn_name == 'cross_entropy':
        smoothing = train_conf.get('label_smoothing', 0.0)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=smoothing)
    elif loss_fn_name == 'focal':
        loss_fn = FocalLoss(alpha=train_conf.get('focal_alpha', 0.75))
    else:
        logging.warning(f"Unknown loss function {loss_fn_name}, using default: CrossEntropyLoss")
        loss_fn = nn.CrossEntropyLoss()
    
    optimizer_name = train_conf.get('optimizer', 'adamw').lower()
    lr = train_conf['lr']
    weight_decay = train_conf.get('weight_decay', 0.01)
    
    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        momentum = train_conf.get('momentum', 0.9)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        logging.warning(f"Unknown optimizer {optimizer_name}, using defualt optimizer: AdamW")
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    scheduler = None
    if 'scheduler' in train_conf:
        scheduler = get_lr_scheduler(optimizer, train_conf['scheduler'])
    
    epochs = train_conf['epoch']
    print_batches = train_conf.get('print_global_step', 60)
    regular_weight = train_conf.get('regular_weight', 0.0)
    patience = train_conf.get('early_stopping_patience', 5)
    delta = float(train_conf.get('early_stopping_delta', 0.001))
    
    global_step = 0
    best_val_f1 = 0
    no_improve_epochs = 0
    
    logging.info(f'Start training: Model={model_class.__name__}, the number of samples={len(dataset)}')
    
    for epoch in range(epochs):
        start_time = time.time()
        global_step, avg_train_loss = do_train(
            train_loader, model, loss_fn, optimizer, global_step,
            print_batches=print_batches,
            regular_weight=regular_weight,
            device=device,
            test_flag=train_conf.get('test_flag', False)
        )
        
        val_f1 = evaluate(model, val_loader, device)
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - start_time
        
        if scheduler:
            if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_f1)  
            else:
                scheduler.step()  
        
        logging.info(
            f'Epoch {epoch+1}/{epochs}: '
            f'TrainLoss={avg_train_loss:.4f}, '
            f'ValF1={val_f1:.4f}, '
            f'LR={current_lr:.6f}, '
            f'Time={epoch_time:.1f}s'
        )
        
        if val_f1 > best_val_f1 + delta:
            best_val_f1 = val_f1
            no_improve_epochs = 0
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))
            logging.info(f'Find better model: ValF1={val_f1:.4f}')
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                logging.info(f'Activate early stoping: {no_improve_epochs} epoch have no improving')
                break
    
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pt')))
    logging.info(f'Finish training, best ValF1={best_val_f1:.4f}')
    
    return model