# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

#import datasets
import util.misc as utils
#from datasets import build_dataset, get_coco_api_from_dataset
#from engine import train_one_epoch, evaluate

from engine_from_detr import train_one_epoch, evaluate

from models import build_model

from armory import paths
from armory.data.adversarial_datasets import carla_obj_det_test, carla_obj_det_dev
from armory.data.datasets import carla_obj_det_train

import torchvision.transforms as T
import os
import logging
from torch.utils.tensorboard import SummaryWriter
print("done with importing")

def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--sgd', action='store_true')
    
    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')


    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)


    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    
    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    
    # dataset parameters
    parser.add_argument('--dataset_file', default='carla_obj_det',  type=str,
                        choices = ['carla_obj_det'])
    parser.add_argument('--modality', default='rgb', type=str,
                        choices = ['rgb', 'depth', 'both'])

    '''
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    '''


    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help = "If this tag is true, we will run the evaluation over the test set.")
    parser.add_argument('--eval_split', default="test", type=str,
                        choices = ["small", "medium", "large", "test"],
                        help = "The split for the Carla adv test set to use.")
    parser.add_argument('--num_workers', default=2, type=int)

    parser.add_argument('--gpu_ids', type=str, default='0')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # for changing the classification head:
    parser.add_argument('--remove_patch_class', action='store_true', default=False,
                        help='whether to remove the class 4 for patch in CARLA label.')
    parser.add_argument('--add_1_from_detr', action='store_true', default=False,
                        help='whether to add an extra 1 into the classification channel as following DETR setup.')
    
    # for evaluation:
    parser.add_argument('--adv_test', action='store_true', default=False,
                        help='whether to evaluate / test with adversarial samples in addition to regular dataset.')
    parser.add_argument('--adv_learning_rate', type=float, default=0.01,
                        help='Learning rate for the adversarial generator to run.') 
    parser.add_argument('--adv_max_iter', type=int, default=100,
                        help='Number of iterations for the adversarial generator to run.')                 
    parser.add_argument('--test_split_ratio', type=float, default="0.2",
                        help='the test / evaluation samples is 20 percent of the regular dataset for CARLA')
    
    # if we want to evaluate over the carla test set instead of the carla dev set:
    parser.add_argument('--eval_over_carla_test_set', action='store_true', default=False,
                        help='whether to evaluate / test with adversarial samples in addition to regular dataset over the armory carla test set instead of dev set.')
    parser.add_argument('--carla_test_set_split', type=str, default="test",
                        choices=["small", "medium", "large", "test"],
                        help="select the split in the armory carla test set. If choose \"test\", then we will evaulate over the whole carla test set.")

    # for adv train:
    parser.add_argument('--adv_train', action='store_true', default=False,
                        help='Enable adversarial training.')

    # for debug
    parser.add_argument('--debug', action='store_true', default=False,
                        help='debug mode')
    parser.add_argument('--name', type=str, default="DETR_CARLA_eval")


    # for pretrain ckpt:
    parser.add_argument('--coco_pretrain', action='store_true', default=False,
                        help='Use coco pretrain model')
    parser.add_argument('--coco_pretrain_ckpt', type=str, 
                        default="./pretrain_ckpt/r50_deformable_detr-checkpoint.pth",
                        help='The path direction to the coco pretrain ckpt.')

    parser.add_argument('--coco_re_mapped_class', action='store_true', default=False,
                        help='The pretrain COCO checkpoint trained with \
                              mapping the classes of COCO from 90 classes to 3 classes in CARLA.')


    return parser


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    logger = logging.getLogger(__name__)
    writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # TODO: update build_model
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.dataset_file == "carla_obj_det":
        data_loader_train = carla_obj_det_train(epochs=args.epochs, batch_size=args.batch_size, modality= args.modality, split="train", framework="numpy", shuffle_files=False) # the split here can only be "train"
        if args.adv_test == True:
            # we will use this dataset for both benign evaluation and adversarial evaluation
            data_loader_val_adv = carla_obj_det_dev(epochs=2 * args.epochs, batch_size=1, modality=args.modality, split="dev", framework="numpy")
            data_loader_test_adv = carla_obj_det_test(epochs=2, batch_size=1, modality= args.modality, split=args.carla_test_set_split, framework="numpy")
        else:
            data_loader_val_adv = carla_obj_det_dev(epochs=args.epochs, batch_size=1, modality= args.modality, split="dev", framework="numpy")
            data_loader_test_adv = carla_obj_det_test(epochs=1, batch_size=1, modality= args.modality, split=args.carla_test_set_split, framework="numpy")

        transform = T.Compose([
                                #T.Resize(800),     # because the adv generator requires size shape
                                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                             ])
    else:
        raise NotImplementedError()

    if args.coco_pretrain_ckpt is not None:
        checkpoint = torch.load(args.coco_pretrain_ckpt, map_location='cpu')

        model_without_ddp.load_state_dict(checkpoint["model"])


    output_dir = Path(args.output_dir)

    if args.eval_over_carla_test_set:
        # In this case, we will evaluate over the Carla test set:
        evaluate(
            model,
            criterion,
            postprocessors, 
            data_loader_train,
            data_loader_test_adv,
            transform,
            device,
            args.output_dir,
            args,
            logger,
            writer,
            epoch=10)
    
    else:
        # In this case, we will evaluate over the Carla dev set:
        evaluate(
            model, 
            criterion, 
            postprocessors, 
            data_loader_train, 
            data_loader_val_adv,
            transform,
            device,
            args.output_dir,
            args,
            logger,
            writer,
            epoch=10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    paths.set_mode("host")  # make the accessory of dataset to the location that armory command use
    
    main(args)
