# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import math
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

from models.modeling import VisionTransformer, CONFIGS, AdversarialNetwork
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader
from utils.dist_util import get_world_size
from utils.transform import get_transform
from utils.utils import visda_acc

from torchvision import transforms, datasets
from data.data_list_image import ImageList, ImageListIndex, rgb_loader
from models import lossZoo

import torch.distributed as dist
from tqdm import tqdm
from torch.cuda.amp import GradScaler

logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model, is_adv=False, is_adv2=False):
    model_to_save = model.module if hasattr(model, 'module') else model
    if not is_adv:
        model_checkpoint = os.path.join(args.output_dir, args.dataset, "%s_checkpoint.bin" % args.name)
    else:
        if not is_adv2:
             model_checkpoint = os.path.join(args.output_dir, args.dataset, "%s_checkpoint_adv.bin" % args.name)
        else:
             model_checkpoint = os.path.join(args.output_dir, args.dataset, "%s_checkpoint_adv_2.bin" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", os.path.join(args.output_dir, args.dataset))


def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]
    model = VisionTransformer(config, args.img_size, zero_head=True, 
                              num_classes=args.num_classes, msa_layer=args.msa_layer)
    model.load_from(np.load(args.pretrained_dir))
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def valid(args, model, ad_net, ad_net2, writer, test_loader, global_step, num):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    ad_net.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits, _, _ = model(x, ad_net=ad_net, ad_net2=ad_net2)

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    if args.dataset == 'visda17':
        accuracy, classWise_acc = visda_acc(all_preds, all_label)
    else:
        accuracy = simple_accuracy(all_preds, all_label)
    print("Valid Accuracy:")
    print(accuracy)
    num.append(accuracy)
    print(len(num))
    print(num)
    logger.info("\n")
    logger.info("Validation Results of: %s" % args.name)
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)
    if args.dataset == 'visda17':
        return accuracy, classWise_acc
    else:
        return accuracy, None

"""train.py ""
"""""
def rand_bbox(size, lam):
    W = 256
    H = 256
    """"""
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
 
    # uniform
    """2.""
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    #
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    """""
    return bbx1, bby1, bbx2, bby2
    
def train(args, model, gpu):
    print(gpu)
    ###############################N1#########################
    
    model.cuda(gpu)
    ###############################N1#########################
    if args.local_rank in [-1, 0]:
        os.makedirs(os.path.join(args.output_dir, args.dataset), exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.dataset, args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    transform_source, transform_target, transform_test = get_transform(args.dataset, args.img_size)
    ###############################N1#########################
    source_sampler = torch.utils.data.distributed.DistributedSampler(ImageList(open(args.source_list).readlines(), transform=transform_source, mode='RGB'))
    source_loader = torch.utils.data.DataLoader(
        ImageList(open(args.source_list).readlines(), transform=transform_source, mode='RGB'),
        batch_size=args.train_batch_size, shuffle=False, num_workers=0, pin_memory=True, sampler=source_sampler)

    
    target_sampler = torch.utils.data.distributed.DistributedSampler(ImageList(open(args.target_list).readlines(), transform=transform_target, mode='RGB'))
    target_loader = torch.utils.data.DataLoader(
        ImageListIndex(open(args.target_list).readlines(), transform=transform_target, mode='RGB'),
        batch_size=args.train_batch_size, shuffle=False, num_workers=0, pin_memory=True, sampler=target_sampler)
    ###############################N1#########################
    test_loader = torch.utils.data.DataLoader(
        ImageList(open(args.test_list).readlines(), transform=transform_test, mode='RGB'),
        batch_size=args.eval_batch_size, shuffle=True, num_workers=0)

    config = CONFIGS[args.model_type]
    ad_net = AdversarialNetwork(config.hidden_size, config.hidden_size//4)
    ###############################N1#########################
    #ad_net.to(args.device)
    ad_net.cuda(gpu)
    ad_net_local = AdversarialNetwork(config.hidden_size//12, config.hidden_size//12)
    #ad_net_local.to(args.device)
    ad_net_local.cuda(gpu)
    ad_net_local2 = AdversarialNetwork(config.hidden_size//3, config.hidden_size//3)
    ad_net_local2.cuda(gpu)
    ###############################N1#########################
    
    optimizer_ad = torch.optim.SGD(list(ad_net.parameters())+list(ad_net_local.parameters()),
                            lr=args.learning_rate/10, 
                            momentum=0.9,
                            weight_decay=args.weight_decay)
    
    optimizer = torch.optim.SGD([
                                    {'params': model.transformer.parameters(), 'lr': args.learning_rate/10},
                                    {'params': model.decoder.parameters(), 'lr': args.learning_rate},
                                    {'params': model.head.parameters()},
                                ],
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
        scheduler_ad = WarmupCosineSchedule(optimizer_ad, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
        scheduler_ad = WarmupLinearSchedule(optimizer_ad, warmup_steps=args.warmup_steps, t_total=t_total)
    #if args.local_rank != -1:
    ###############################N1#########################
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)                  #
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu]) 
    ad_net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(ad_net)                  #
    ad_net = torch.nn.parallel.DistributedDataParallel(ad_net, device_ids=[gpu]) 
    ad_net_local = torch.nn.SyncBatchNorm.convert_sync_batchnorm(ad_net_local)                  #
    ad_net_local = torch.nn.parallel.DistributedDataParallel(ad_net_local, device_ids=[gpu]) 
    ###############################N1#########################
    model.zero_grad()
    ad_net.zero_grad()
    ad_net_local.zero_grad()
    ad_net_local2.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    best_acc = 0
    best_classWise_acc = ''

    len_source = len(source_loader)
    len_target = len(target_loader)            
    num = []
    for global_step in range(1, t_total):
        source_sampler.set_epoch(global_step)
        target_sampler.set_epoch(global_step)
        model.train()
        ad_net.train()
        ad_net_local.train()
        ad_net_local2.train()
        if (global_step-1) % (len_source-1) == 0:
            iter_source = iter(source_loader)    
        if (global_step-1) % (len_target-1) == 0:
            iter_target = iter(target_loader)
        
        data_source = iter_source.next()
        data_target = iter_target.next()

        #x_s, y_s = tuple(t.to(args.device) for t in data_source)
        #x_t, _, index_t = tuple(t.to(args.device) for t in data_target)
        
        x_s, y_s = tuple(t.cuda(gpu) for t in data_source)
        x_t, _, index_t = tuple(t.cuda(gpu) for t in data_target)
        
        
        logits_s, logits_t, loss_ad_local, loss_rec, x_s, x_t, mixup_loss, loss_m = model(x_s, x_t, ad_net_local,ad_net_local2, y_s)
        ######################################
        loss_fct = CrossEntropyLoss().to(gpu)
        ######################################
        loss_clc = loss_fct(logits_s.view(-1, args.num_classes), y_s.view(-1))
        
        loss_im = lossZoo.im(logits_t.view(-1, args.num_classes))
        
        loss_ad_global = lossZoo.adv(torch.cat((x_s[:,0], x_t[:,0]), 0), ad_net)
        loss = loss_clc + args.beta * loss_ad_global + args.gamma * loss_ad_local + 0*loss_rec + args.lamda * (mixup_loss + loss_m)

        if args.use_im:
            loss += (args.theta * loss_im)
        
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(ad_net.parameters(), args.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(ad_net_local.parameters(), args.max_grad_norm)
        
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        
        optimizer_ad.step()
        optimizer_ad.zero_grad()
        scheduler_ad.step()
        
        if args.local_rank in [-1, 0]:
            writer.add_scalar("train/loss", scalar_value=loss.item(), global_step=global_step)
            writer.add_scalar("train/loss_clc", scalar_value=loss_clc.item(), global_step=global_step)
            writer.add_scalar("train/loss_ad_global", scalar_value=loss_ad_global.item(), global_step=global_step)
            writer.add_scalar("train/loss_ad_local", scalar_value=loss_ad_local.item(), global_step=global_step)
            writer.add_scalar("train/loss_rec", scalar_value=loss_rec.item(), global_step=global_step)
            writer.add_scalar("train/loss_im", scalar_value=loss_im.item(), global_step=global_step)
            writer.add_scalar("train/lr", scalar_value=scheduler.get_last_lr()[0], global_step=global_step)
        
        if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
            accuracy, classWise_acc = valid(args, model, ad_net_local,ad_net_local2, writer, test_loader, global_step, num)
            """i1 = "dataset/VisDA/aeroplane.txt"
            i2 = "dataset/VisDA/bicycle.txt"
            i3 = "dataset/VisDA/bus.txt"
            i4 = "dataset/VisDA/car.txt"
            i5 = "dataset/VisDA/horse.txt"
            i6 = "dataset/VisDA/knife.txt"
            i7 = "dataset/VisDA/motorcycle.txt"
            i8 = "dataset/VisDA/person.txt"
            i9 = "dataset/VisDA/plant.txt"
            i10 = "dataset/VisDA/skateboard.txt"
            i11 = "dataset/VisDA/train.txt"
            i12 = "dataset/VisDA/truck.txt"
            
            test_loader1 = torch.utils.data.DataLoader(
                ImageList(open(i1).readlines(), transform=transform_test, mode='RGB'),
                batch_size=args.eval_batch_size, shuffle=False, num_workers=4)
            test_loader2 = torch.utils.data.DataLoader(
                ImageList(open(i2).readlines(), transform=transform_test, mode='RGB'),
                batch_size=args.eval_batch_size, shuffle=False, num_workers=4)
            test_loader3 = torch.utils.data.DataLoader(
                ImageList(open(i3).readlines(), transform=transform_test, mode='RGB'),
                batch_size=args.eval_batch_size, shuffle=False, num_workers=4)
            test_loader4 = torch.utils.data.DataLoader(
                ImageList(open(i4).readlines(), transform=transform_test, mode='RGB'),
                batch_size=args.eval_batch_size, shuffle=False, num_workers=4)
            test_loader5 = torch.utils.data.DataLoader(
                ImageList(open(i5).readlines(), transform=transform_test, mode='RGB'),
                batch_size=args.eval_batch_size, shuffle=False, num_workers=4)
            test_loader6 = torch.utils.data.DataLoader(
                ImageList(open(i6).readlines(), transform=transform_test, mode='RGB'),
                batch_size=args.eval_batch_size, shuffle=False, num_workers=4)
            test_loader7 = torch.utils.data.DataLoader(
                ImageList(open(i7).readlines(), transform=transform_test, mode='RGB'),
                batch_size=args.eval_batch_size, shuffle=False, num_workers=4)
            test_loader8 = torch.utils.data.DataLoader(
                ImageList(open(i8).readlines(), transform=transform_test, mode='RGB'),
                batch_size=args.eval_batch_size, shuffle=False, num_workers=4)
            test_loader9 = torch.utils.data.DataLoader(
                ImageList(open(i9).readlines(), transform=transform_test, mode='RGB'),
                batch_size=args.eval_batch_size, shuffle=False, num_workers=4)
            test_loader10 = torch.utils.data.DataLoader(
                ImageList(open(i10).readlines(), transform=transform_test, mode='RGB'),
                batch_size=args.eval_batch_size, shuffle=False, num_workers=4)
            test_loader11 = torch.utils.data.DataLoader(
                ImageList(open(i11).readlines(), transform=transform_test, mode='RGB'),
                batch_size=args.eval_batch_size, shuffle=False, num_workers=4)
            test_loader12 = torch.utils.data.DataLoader(
                ImageList(open(i12).readlines(), transform=transform_test, mode='RGB'),
                batch_size=args.eval_batch_size, shuffle=False, num_workers=4)

            #accuracy, classWise_acc = valid(args, model, ad_net_local,ad_net_local2, writer, test_loader, global_step)
            accuracy1, classWise_acc1 = valid(args, model, ad_net_local,ad_net_local2, writer, test_loader1, global_step)
            accuracy2, classWise_acc2 = valid(args, model, ad_net_local,ad_net_local2, writer, test_loader2, global_step)
            accuracy3, classWise_acc3 = valid(args, model, ad_net_local,ad_net_local2, writer, test_loader3, global_step)
            accuracy4, classWise_acc4 = valid(args, model, ad_net_local,ad_net_local2, writer, test_loader4, global_step)
            accuracy5, classWise_acc5 = valid(args, model, ad_net_local,ad_net_local2, writer, test_loader5, global_step)
            accuracy6, classWise_acc6 = valid(args, model, ad_net_local,ad_net_local2, writer, test_loader6, global_step)
            accuracy7, classWise_acc7 = valid(args, model, ad_net_local,ad_net_local2, writer, test_loader7, global_step)
            accuracy8, classWise_acc8 = valid(args, model, ad_net_local,ad_net_local2, writer, test_loader8, global_step)
            accuracy9, classWise_acc9 = valid(args, model, ad_net_local,ad_net_local2, writer, test_loader9, global_step)
            accuracy10, classWise_acc10 = valid(args, model, ad_net_local,ad_net_local2, writer, test_loader10, global_step)
            accuracy11, classWise_acc11 = valid(args, model, ad_net_local,ad_net_local2, writer, test_loader11, global_step)
            accuracy12, classWise_acc12 = valid(args, model, ad_net_local,ad_net_local2, writer, test_loader12, global_step)
            accuracy = (accuracy1+accuracy2+accuracy3+accuracy4+accuracy5+accuracy6+accuracy7+accuracy8+accuracy9+accuracy10+accuracy11+accuracy12)/12
            print("average:",accuracy)
            classWise_acc = classWise_acc1"""

            if best_acc < accuracy:
                save_model(args, model)
                save_model(args, ad_net_local, is_adv=True)
                save_model(args, ad_net_local2, is_adv=True, is_adv2=True)
                best_acc = accuracy

                if classWise_acc is not None:
                    best_classWise_acc = classWise_acc
            model.train()
            ad_net_local.train()
            ad_net_local2.train()
            logger.info("Current Best Accuracy: %2.5f" % best_acc)
            logger.info("Current Best element-wise acc: %s" % best_classWise_acc)
        
    if args.local_rank in [-1, 0]:
        writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("Best element-wise Accuracy: \t%s" % best_classWise_acc)
    logger.info("End Training!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", help="Which downstream task.")
    parser.add_argument("--source_list", help="Path of the training data.")
    parser.add_argument("--target_list", help="Path of the test data.")
    parser.add_argument("--test_list", help="Path of the test data.")
    parser.add_argument("--num_classes", default=10, type=int,
                        help="Number of classes in the dataset.")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output5", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=512, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=1000, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--beta", default=0.1, type=float,
                        help="The importance of the adversarial loss.")
    parser.add_argument("--gamma", default=0.1, type=float,
                        help="The importance of the local adversarial loss.")
    parser.add_argument("--theta", default=1.0, type=float,
                        help="The importance of the IM loss.")
    parser.add_argument("--lamda", default=1.0, type=float,
                        help="The importance of the local adversarial loss.")
    parser.add_argument("--use_im", default=False, action="store_true",
                        help="Use information maximization loss.")
    parser.add_argument("--msa_layer", default=12, type=int,
                        help="The layer that incorporates local alignment.")
    parser.add_argument("--is_test", default=False, action="store_true",
                        help="If in test mode.")

    parser.add_argument("--learning_rate", default=3e-3, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        #torch.distributed.init_process_group(backend='nccl', init_method='env://')
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

        args.n_gpu = 1
    args.device = device
    
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))
    
    set_seed(args)
    if args.is_test:
        test(args)
    else:
        args, model = setup(args)
        #model.to(args.device)
        train(args, model, args.local_rank)


if __name__ == "__main__":
    main()
