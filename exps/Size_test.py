# -*- coding: utf-8 -*-
# @Time    : 2021/8/4 23:36
# @Author  : LIU YI

import sys, time, torch, random, argparse
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from copy import deepcopy
from pathlib import Path
import warnings
import os
warnings.filterwarnings("ignore")

lib_dir = (Path(__file__).parent / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
from config_utils import load_config, obtain_basic_args as obtain_args
from procedures import prepare_seed, prepare_logger, save_checkpoint, copy_checkpoint
from procedures import get_optim_scheduler, get_procedures
from datasets import get_datasets
from models import obtain_model
from nas_infer_model import obtain_nas_infer_model
from utils import get_model_infos
from log_utils import AverageMeter, time_string, convert_secs2time
from models import create_cnn_model, count_parameters_in_MB

if __name__ == '__main__':

    args = obtain_args()
    args.dataset = 'cifar10'
    args.data_path = '../data'
    args.teacher_model = 'autodl-searched'
    args.teacher_path = './output/nas-infer/cifar10-BS96-gdas_serached/checkpoint/seed-25764-bestresnet110_autodl-searched_96.03%_08-04,08.pth'
    args.model_config = '../configs/archs/NAS-CIFAR-none.config'
    args.optim_config = '../configs/opts/NAS-CIFAR.config'
    args.extra_model_path = '../exps/algos/output/search-cell-dar/GDAS-cifar10-BN1/checkpoint/seed-4185-basic.pth'
    # args.extra_model_path = None
    args.procedure = 'KD'
    args.save_dir = './output/nas-infer/cifar10-BS96-gdas_serached'
    args.cutout_length = 16
    args.batch_size = 48
    args.rand_seed = -1
    args.workers = 4
    args.eval_frequency = 1
    args.print_freq = 500
    args.print_freq_eval = 1000

    models = ['googlenet', 'resnet110', 'resnet56', 'resnet44', 'resnet32', 'resnet26', 'resnet20', 'resnet14', 'resnet8', 'plane10',
             'plane8', 'plane6', 'plane4', 'plane2', 'vgg19', 'vgg16', 'vgg13', 'vgg11', 'alexnet', 'lenet',
             'squeezenet', 'shufflenetg2', 'shufflenetg3']

    train_data, valid_data, xshape, class_num = get_datasets(
        args.dataset, args.data_path, args.cutout_length
    )

    for one in models:
        base_model, optimizer, scheduler = create_cnn_model(one, args.dataset, 160, None, use_cuda=1)
        flop, param = get_model_infos(base_model, xshape)
        info = 'Model {}, Params={:.2f} MB, FLOPs={:.2f} M ... = {:.2f} G'.format(one, param, flop, flop / 1e3)
        print(info)
