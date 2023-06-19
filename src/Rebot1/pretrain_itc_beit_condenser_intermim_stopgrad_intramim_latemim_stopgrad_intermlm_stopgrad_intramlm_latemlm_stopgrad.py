import argparse
import os
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.blip_pretrain_itc_beit_condenser_intermim_stopgrad_intramim_latemim_stopgrad_intermlm_stopgrad_intramlm_latemlm_stopgrad import blip_pretrain
import utils
from utils import warmup_lr_schedule, step_lr_schedule
from data_beit import create_dataset, create_sampler, create_loader
from torch.cuda.amp import autocast, GradScaler
past_steps = 0

def train(model, data_loader, optimizer, epoch, device, config, writer, scaler):
    # train
    model.train() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_intramim', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_intramlm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_intermim', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_intermlm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_latemim', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_latemlm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    
    global past_steps
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50   

    if config['laion_path']:
        data_loader.dataset.reload_laion(epoch)
    
    data_loader.sampler.set_epoch(epoch)

    for i, (image4encoder, image4vqkd, bool_masked_pos, caption) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        if epoch==0:
            warmup_lr_schedule(optimizer, i, config['warmup_steps'], config['warmup_lr'], config['init_lr'])
            
        optimizer.zero_grad()
        
        image4encoder = image4encoder.to(device,non_blocking=True)
        image4vqkd = image4vqkd.to(device,non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device,non_blocking=True)
        
        # ramp up alpha in the first 2 epochs
        alpha = config['alpha']*min(1,(epoch*len(data_loader)+i)/(2*len(data_loader))) 

        if args.fp16:
            with autocast():
                loss_ita, loss_intramim, loss_intramlm, loss_intermim, loss_intermlm, loss_latemim, loss_latemlm = model(image4encoder, image4vqkd, bool_masked_pos, caption, alpha = alpha)  
                loss = loss_ita + loss_intramim*args.intramim_weight + loss_intramlm*args.intramlm_weight + loss_intermim*args.intermim_weight + loss_intermlm*args.intermlm_weight + loss_latemim*args.intramim_weight + loss_latemlm*args.intramlm_weight
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_max_norm, norm_type=2)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_ita, loss_intramim, loss_intramlm, loss_intermim, loss_intermlm, loss_latemim, loss_latemlm = model(image4encoder, image4vqkd, bool_masked_pos, caption, alpha = alpha)  
            loss = loss_ita + loss_intramim*args.intramim_weight + loss_intramlm*args.intramlm_weight + loss_intermim*args.intermim_weight + loss_intermlm*args.intermlm_weight + loss_latemim*args.intramim_weight + loss_latemlm*args.intramlm_weight
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_max_norm, norm_type=2)
            optimizer.step()   
        if dist.get_rank()==0:
            writer.add_scalar('loss_ita', loss_ita, past_steps)
            writer.add_scalar('loss_intramim', loss_intramim, past_steps)
            writer.add_scalar('loss_intramlm', loss_intramlm, past_steps)
            writer.add_scalar('loss_intermim', loss_intermim, past_steps)
            writer.add_scalar('loss_intermlm', loss_intermlm, past_steps)
            writer.add_scalar('loss_latemim', loss_latemim, past_steps)
            writer.add_scalar('loss_latemlm', loss_latemlm, past_steps)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], past_steps)
            past_steps+=1  

        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(loss_intramim=loss_intramim.item())
        metric_logger.update(loss_intramlm=loss_intramlm.item())
        metric_logger.update(loss_intermim=loss_intermim.item())
        metric_logger.update(loss_intermlm=loss_intermlm.item())
        metric_logger.update(loss_latemim=loss_latemim.item())
        metric_logger.update(loss_latemlm=loss_latemlm.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])  

        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  


def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating dataset")
    datasets = [create_dataset('pretrain', config, min_scale=0.2, num_mask_patches=args.mim_mask_patches)]
    print('number of training samples: %d'%len(datasets[0]))

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()            
    samplers = create_sampler(datasets, [True], num_tasks, global_rank)         

    data_loader = create_loader(datasets,samplers,batch_size=[config['batch_size']], num_workers=[4], is_trains=[True], collate_fns=[None])[0]      

    #### Model #### 
    print("Creating model")
    model = blip_pretrain(image_size=config['image_size'], vit=config['vit'], queue_size=config['queue_size'],
                              mim_early_layers=args.mim_early_layers, mim_cls_layers=args.mim_cls_layers, intermim_init=args.intermim_init, intramim_init=args.intramim_init, mim_tie=args.mim_tie, 
                              mlm_early_layers=args.mlm_early_layers, mlm_cls_layers=args.mlm_cls_layers, mlm_probability=args.mlm_probability, intermlm_init=args.intermlm_init, intramlm_init=args.intramlm_init, mlm_tie=args.mlm_tie,)
    model = model.to(device)   

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
    
    start_epoch = 0
    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']    
        model.load_state_dict(state_dict)    
        
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']+1                
        print('resume checkpoint from %s'%args.checkpoint)    
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=True)
        model_without_ddp = model.module    
        
    print("Start training")
    start_time = time.time() 
    writer=None
    if dist.get_rank()==0:
        tensorboard_dir=args.tensorboard_dir
        if os.path.exists(tensorboard_dir)==False:
            os.makedirs(tensorboard_dir)
        writer = SummaryWriter(tensorboard_dir)
             
    scaler = GradScaler()          
    for epoch in range(start_epoch, config['max_epoch']):
        
        step_lr_schedule(optimizer, epoch, config['init_lr'], config['min_lr'], config['lr_decay_rate'])
                
        train_stats = train(model, data_loader, optimizer, epoch, device, config, writer, scaler) 
        if utils.is_main_process():  
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                        }                     
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch))  
            
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")

        dist.barrier()        
                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/pretrain.yaml')
    parser.add_argument('--output_dir', default='output/Pretrain')  
    parser.add_argument('--checkpoint', default='')    
    parser.add_argument('--evaluate', action='store_true')    
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--mim_mask_patches', default=75, type=int)
    parser.add_argument('--mim_early_layers', default=6, type=int)
    parser.add_argument('--mim_cls_layers', default=2, type=int)
    parser.add_argument('--intermim_init', default=1, type=int)
    parser.add_argument('--intramim_init', default=1, type=int)
    parser.add_argument('--intermim_weight', default=1.0, type=float)
    parser.add_argument('--intramim_weight', default=1.0, type=float)
    parser.add_argument('--mim_tie', default=0, type=int)

    parser.add_argument('--mlm_probability', default=0.15, type=float)
    parser.add_argument('--intermlm_weight', default=1.0, type=float)
    parser.add_argument('--intramlm_weight', default=1.0, type=float)
    parser.add_argument('--mlm_early_layers', default=6, type=int)
    parser.add_argument('--mlm_cls_layers', default=2, type=int)
    parser.add_argument('--intermlm_init', default=1, type=int)
    parser.add_argument('--intramlm_init', default=1, type=int)
    parser.add_argument('--mlm_tie', default=0, type=int)
    
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--tensorboard_dir', default='tensorboard/debug')

    parser.add_argument('--fp16', default=1, type=int)
    parser.add_argument('--grad_max_norm', default=3.0, type=float)

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))
    
    main(args, config)