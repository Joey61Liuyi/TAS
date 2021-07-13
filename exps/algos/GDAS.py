##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020 #
###########################################################################
# Searching for A Robust Neural Architecture in Four GPU Hours, CVPR 2019 #
###########################################################################
import sys, time, random, argparse
from copy import deepcopy
import torch
from pathlib import Path
import numpy as np
import torch.nn as nn
import math
import os
import torch.nn.init as init
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

lib_dir = (Path(__file__).parent / ".." / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
from config_utils import load_config, dict2config
from datasets import get_datasets, get_nas_search_loaders
from procedures import (
    prepare_seed,
    prepare_logger,
    save_checkpoint,
    copy_checkpoint,
    get_optim_scheduler,
)
from utils import get_model_infos, obtain_accuracy
from log_utils import AverageMeter, time_string, convert_secs2time
from models import get_cell_based_tiny_net, get_search_spaces
from nas_201_api import NASBench201API as API
from models import create_cnn_model, count_parameters_in_MB




# def NAS_TA_Train(teacher, network, student, base_input, base_target, arch_input, arch_target, T, lambda_, criterion, w_optimizer, a_optimizer):
#     network.train()
#     teacher.eval()
#     student.eval()
#     w_optimizer.zero_grad()
#     logits_teacher = teacher(base_input.cuda())
#     logits_student = student(base_input.cuda())
#     _, logits_ta = network(base_input.cuda())
#     loss_KD_TA = T * T * nn.KLDivLoss()(F.log_softmax(logits_ta / T, dim=1),F.softmax(logits_teacher / T, dim=1))+T * T * nn.KLDivLoss()(F.log_softmax(logits_ta / T, dim=1),F.softmax(logits_student / T, dim=1))
#     base_loss = (1 - lambda_) * criterion(logits_ta, base_target) + lambda_ * loss_KD_TA
#     base_loss.backward()
#     torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
#     w_optimizer.step()
#
#     a_optimizer.zero_grad()
#     _, logits = network(arch_input)
#     logits_teacher = teacher(arch_input.cuda())
#     logits_student = student(arch_input.cuda())
#     loss_KD_TA = T * T * nn.KLDivLoss()(F.log_softmax(logits / T, dim=1), F.softmax(logits_teacher / T, dim=1))+T * T * nn.KLDivLoss()(F.log_softmax(logits_ta / T, dim=1),F.softmax(logits_student / T, dim=1))
#     arch_loss = (1 - lambda_) * criterion(logits, arch_target) + lambda_ * loss_KD_TA
#     arch_loss.backward()
#     a_optimizer.step()
#
# def train_teacher(xloader,teacher_model, criterion, optimizer, logger, total_epochs, teacher_name):
#
#     best_acc = -1
#
#     for epoch in range(total_epochs):
#         data_time, batch_time = AverageMeter(), AverageMeter()
#         base_losses, base_top1, base_top5 = AverageMeter(), AverageMeter(), AverageMeter()
#         end = time.time()
#         for step, (base_inputs, base_targets, arch_inputs, arch_targets) in enumerate(
#                 xloader
#         ):
#             teacher_model.train()
#             base_targets = base_targets.cuda(non_blocking=True)
#             arch_targets = arch_targets.cuda(non_blocking=True)
#             data_time.update(time.time() - end)
#             optimizer.zero_grad()
#             logits_teacher = teacher_model(base_inputs.cuda())
#             loss = criterion(logits_teacher, base_targets)
#             loss.backward()
#             optimizer.step()
#
#             teacher_model.eval()
#             logits_teacher = teacher_model(arch_inputs.cuda())
#             Teacher_prec1, Teacher_prec5 = obtain_accuracy(
#                 logits_teacher, arch_targets.data, topk = (1,5)
#             )
#             base_losses.update(loss.item(), base_inputs.size(0))
#             base_top1.update(Teacher_prec1.item(), arch_inputs.size(0))
#             base_top5.update(Teacher_prec5.item(), arch_inputs.size(0))
#
#         logger.log('Training Teacher {:} at {:} epoch, the current acc is {:.2f}%, and loss is {:}'.format(teacher_name, epoch, base_top1.avg, base_losses.avg))
#         if 'resnet' in teacher_name:
#             adjust_learning_rate(optimizer, epoch, total_epochs)
#
#         if base_top1.avg >= best_acc:
#             g = os.walk(os.path.abspath('.'))
#             for path, dir_list, file_list in g:
#                 for file_name in file_list:
#                     if 'Teacher_model_{}_{:.2f}%'.format(teacher_name, best_acc) in file_name:
#                         tep = os.path.join(path, file_name)
#                         os.remove(tep)
#             best_acc = base_top1.avg
#             torch.save({
#                 'model_state_dict': teacher_model.state_dict(),
#                 'optimizer_dict': optimizer.state_dict(),
#                 'epoch': epoch
#             }, 'Teacher_model_{}_{:.2f}%_{}.pth.tar'.format(teacher_name, best_acc, time.strftime("%m-%d,%H", time.localtime()))
#             )
#
#
# def train_student(xloader,
#     network,
#     student_model,
#     criterion,
#     optimizer,
#     epoch_str,
#     print_freq,
#     logger,):
#     network.eval()
#     data_time, batch_time = AverageMeter(), AverageMeter()
#     base_losses, base_top1, base_top5 = AverageMeter(), AverageMeter(), AverageMeter()
#     end = time.time()
#     T = 5
#     lambda_ = 0.5
#
#     for step, (base_inputs, base_targets, arch_inputs, arch_targets) in enumerate(
#             xloader
#     ):
#         network.eval()
#         student_model.train()
#         base_targets = base_targets.cuda(non_blocking=True)
#         arch_targets = arch_targets.cuda(non_blocking=True)
#         # measure data loading time
#         data_time.update(time.time() - end)
#         optimizer.zero_grad()
#
#         try:
#             _, logits_TA = network(base_inputs.cuda())
#         except:
#             logits_TA = network(base_inputs.cuda())
#
#         logits_student = student_model(base_inputs.cuda())
#         loss_KD_TA = T * T * nn.KLDivLoss()(F.log_softmax(logits_student / T, dim=1),
#                                             F.softmax(logits_TA / T, dim=1))
#         base_loss = (1 - lambda_) * criterion(logits_student, base_targets) + lambda_ * loss_KD_TA
#         base_loss.backward()
#         torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
#         optimizer.step()
#
#         # record
#
#         student_model.eval()
#         logits_student = student_model(arch_inputs.cuda())
#
#         base_prec1, base_prec5 = obtain_accuracy(
#             logits_student.data, arch_targets.data, topk=(1, 5)
#         )
#
#         base_losses.update(base_loss.item(), base_inputs.size(0))
#         base_top1.update(base_prec1.item(), base_inputs.size(0))
#         base_top5.update(base_prec5.item(), base_inputs.size(0))
#         # update the architecture-weight
#
#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()
#
#         if step % print_freq == 0 or step + 1 == len(xloader):
#             Sstr = (
#                     "*SEARCH* "
#                     + time_string()
#                     + " [{:}][{:03d}/{:03d}]".format(epoch_str, step, len(xloader))
#             )
#             Tstr = "Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})".format(
#                 batch_time=batch_time, data_time=data_time
#             )
#             Wstr = "Student_Final [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]".format(
#                 loss=base_losses, top1=base_top1, top5=base_top5
#             )
#
#             logger.log(Sstr + " " + Tstr + " " + Wstr)
#
#     return base_losses.avg, base_top1.avg, base_top5.avg


def search_func_modified(xloader,
    teacher_model,
    network,
    student_model,
    criterion,
    scheduler,
    w_optimizer,
    a_optimizer,
    epoch_str,
    print_freq,
    logger,
):
    teacher_model.eval()
    student_model.eval()
    data_time, batch_time = AverageMeter(), AverageMeter()
    base_losses, base_top1, base_top5 = AverageMeter(), AverageMeter(), AverageMeter()
    arch_losses, arch_top1, arch_top5 = AverageMeter(), AverageMeter(), AverageMeter()
    end = time.time()
    T = 5
    lambda_ = 0.5

    for step, (base_inputs, base_targets, arch_inputs, arch_targets) in enumerate(
            xloader
    ):
        scheduler.update(None, 1.0 * step / len(xloader))
        base_targets = base_targets.cuda(non_blocking=True)
        arch_targets = arch_targets.cuda(non_blocking=True)
        # measure data loading time
        data_time.update(time.time() - end)

        network.train()
        w_optimizer.zero_grad()
        _, logits = network(base_inputs.cuda())
        logits_teacher = teacher_model(base_inputs.cuda())
        logits_student = student_model(base_inputs.cuda())
        loss_KD_TA = T * T * nn.KLDivLoss()(F.log_softmax(logits / T, dim=1),
                                            F.softmax(logits_teacher / T, dim=1))
        loss_KD_TA+= T * T * nn.KLDivLoss()(F.log_softmax(logits / T, dim=1),
                                            F.softmax(logits_student / T, dim=1))

        base_loss = (1 - lambda_) * criterion(logits, base_targets) + lambda_ * loss_KD_TA
        base_loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
        w_optimizer.step()

        # record
        base_prec1, base_prec5 = obtain_accuracy(
            logits.data, base_targets.data, topk=(1, 5)
        )
        base_losses.update(base_loss.item(), base_inputs.size(0))
        base_top1.update(base_prec1.item(), base_inputs.size(0))
        base_top5.update(base_prec5.item(), base_inputs.size(0))
        # update the architecture-weight
        a_optimizer.zero_grad()
        _, logits = network(arch_inputs.cuda())

        logits_teacher = teacher_model(arch_inputs.cuda())
        logits_student = student_model(arch_inputs.cuda())

        loss_KD_TA = T * T * nn.KLDivLoss()(F.log_softmax(logits / T, dim=1),
                                            F.softmax(logits_teacher / T, dim=1))
        loss_KD_TA += T * T * nn.KLDivLoss()(F.log_softmax(logits / T, dim=1),
                                             F.softmax(logits_student / T, dim=1))
        arch_loss = (1 - lambda_) * criterion(logits, arch_targets) + lambda_ * loss_KD_TA
        arch_loss.backward()
        a_optimizer.step()
        # record
        arch_prec1, arch_prec5 = obtain_accuracy(
            logits.data, arch_targets.data, topk=(1, 5)
        )
        arch_losses.update(arch_loss.item(), arch_inputs.size(0))
        arch_top1.update(arch_prec1.item(), arch_inputs.size(0))
        arch_top5.update(arch_prec5.item(), arch_inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if step % print_freq == 0 or step + 1 == len(xloader):
            Sstr = (
                    "*SEARCH* "
                    + time_string()
                    + " [{:}][{:03d}/{:03d}]".format(epoch_str, step, len(xloader))
            )
            Tstr = "Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})".format(
                batch_time=batch_time, data_time=data_time
            )
            Wstr = "Base [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]".format(
                loss=base_losses, top1=base_top1, top5=base_top5
            )
            Astr = "Arch [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]".format(
                loss=arch_losses, top1=arch_top1, top5=arch_top5
            )

            logger.log(Sstr + " " + Tstr + " " + Wstr + " " + Astr)

    return base_losses.avg, base_top1.avg, base_top5.avg, arch_losses.avg, arch_top1.avg, arch_top5.avg
#
#
# def search_func(
#     xloader,
#     teacher_model,
#     network,
#     student_model,
#     criterion,
#     scheduler,
#     w_optimizer,
#     a_optimizer,
#     student_optimizer,
#     epoch_str,
#     print_freq,
#     logger,
#     training_mode
# ):
#
#     teacher_model.eval()
#     data_time, batch_time = AverageMeter(), AverageMeter()
#     base_losses, base_top1, base_top5 = AverageMeter(), AverageMeter(), AverageMeter()
#     arch_losses, arch_top1, arch_top5 = AverageMeter(), AverageMeter(), AverageMeter()
#     student_losses, student_top1, student_top5 = AverageMeter(), AverageMeter(), AverageMeter()
#     end = time.time()
#     T = 5
#     lambda_ = 0.5
#
#     for step, (base_inputs, base_targets, arch_inputs, arch_targets) in enumerate(
#         xloader
#     ):
#         scheduler.update(None, 1.0 * step / len(xloader))
#         base_targets = base_targets.cuda(non_blocking=True)
#         arch_targets = arch_targets.cuda(non_blocking=True)
#         # measure data loading time
#         data_time.update(time.time() - end)
#
#         # pretraining TA
#
#
#         if a_optimizer == None:
#             if training_mode == 0:
#                 network.train()
#                 w_optimizer.zero_grad()
#                 logits_ta = network(base_inputs.cuda())
#                 logits_teacher = teacher_model(base_inputs.cuda())
#                 loss_KD_TA = T * T * nn.KLDivLoss()(F.log_softmax(logits_ta / T, dim=1),
#                                                     F.softmax(logits_teacher / T, dim=1))
#                 TA_loss = (1 - lambda_) * criterion(logits_ta, base_targets) + lambda_ * loss_KD_TA
#                 TA_loss.backward()
#                 # torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
#                 w_optimizer.step()
#
#                 network.eval()
#                 logits_TA_test = network(arch_inputs.cuda())
#
#                 TA_prec1, TA_prec5 = obtain_accuracy(
#                     logits_TA_test.data, arch_targets.data, topk=(1, 5)
#                 )
#
#                 base_losses.update(TA_loss.item(), base_inputs.size(0))
#                 base_top1.update(TA_prec1.item(), arch_inputs.size(0))
#                 base_top5.update(TA_prec5.item(), arch_inputs.size(0))
#
#             elif training_mode == 1:
#                 network.train()
#                 w_optimizer.zero_grad()
#                 logits_ta = network(base_inputs.cuda())
#                 logits_teacher = teacher_model(base_inputs.cuda())
#                 loss_KD_TA = T * T * nn.KLDivLoss()(F.log_softmax(logits_ta / T, dim=1),
#                                                     F.softmax(logits_teacher / T, dim=1))
#                 TA_loss = (1 - lambda_) * criterion(logits_ta, base_targets) + lambda_ * loss_KD_TA
#                 TA_loss.backward()
#                 # torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
#                 w_optimizer.step()
#
#                 network.eval()
#                 logits_TA_test = network(arch_inputs.cuda())
#
#                 TA_prec1, TA_prec5 = obtain_accuracy(
#                     logits_TA_test.data, arch_targets.data, topk=(1, 5)
#                 )
#
#                 base_losses.update(TA_loss.item(), base_inputs.size(0))
#                 base_top1.update(TA_prec1.item(), arch_inputs.size(0))
#                 base_top5.update(TA_prec5.item(), arch_inputs.size(0))
#
#                 # Training Student
#                 student_model.train()
#                 network.eval()
#                 student_optimizer.zero_grad()
#                 logits_ta = network(base_inputs.cuda())
#                 logits_student = student_model(base_inputs.cuda())
#                 loss_KD_Student = T * T * nn.KLDivLoss()(F.log_softmax(logits_student / T, dim=1),
#                                                          F.softmax(logits_ta / T, dim=1))
#                 student_loss = (1 - lambda_) * criterion(logits_student, base_targets) + lambda_ * loss_KD_Student
#                 student_loss.backward()
#                 # torch.nn.utils.clip_grad_norm_(student_model.parameters(), 5)
#                 student_optimizer.step()
#
#                 student_model.eval()
#                 logits_student_test = student_model(arch_inputs.cuda())
#
#                 student_prec1, student_prec5 = obtain_accuracy(
#                     logits_student_test.data, arch_targets.data, topk=(1, 5)
#                 )
#
#                 student_losses.update(student_loss.item(), base_inputs.size(0))
#                 student_top1.update(student_prec1.item(), arch_inputs.size(0))
#                 student_top5.update(student_prec5.item(), arch_inputs.size(0))
#
#             else:
#                 student_model.train()
#                 network.eval()
#                 student_optimizer.zero_grad()
#                 logits_ta = network(base_inputs.cuda())
#                 logits_student = student_model(base_inputs.cuda())
#                 loss_KD_Student = T * T * nn.KLDivLoss()(F.log_softmax(logits_student / T, dim=1),
#                                                          F.softmax(logits_ta / T, dim=1))
#                 student_loss = (1 - lambda_) * criterion(logits_student, base_targets) + lambda_ * loss_KD_Student
#                 student_loss.backward()
#                 # torch.nn.utils.clip_grad_norm_(student_model.parameters(), 5)
#                 student_optimizer.step()
#
#                 student_model.eval()
#                 logits_student_test = student_model(arch_inputs.cuda())
#
#                 student_prec1, student_prec5 = obtain_accuracy(
#                     logits_student_test.data, arch_targets.data, topk=(1, 5)
#                 )
#
#                 student_losses.update(student_loss.item(), base_inputs.size(0))
#                 student_top1.update(student_prec1.item(), arch_inputs.size(0))
#                 student_top5.update(student_prec5.item(), arch_inputs.size(0))
#
#         else:
#             if training_mode == 0:
#                 network.train()
#                 w_optimizer.zero_grad()
#                 _, logits = network(base_inputs.cuda())
#                 logits_teacher = teacher_model(base_inputs.cuda())
#                 loss_KD_TA = T * T * nn.KLDivLoss()(F.log_softmax(logits / T, dim=1),
#                                                     F.softmax(logits_teacher / T, dim=1))
#                 base_loss = (1 - lambda_) * criterion(logits, base_targets) + lambda_ * loss_KD_TA
#                 base_loss.backward()
#                 torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
#
#                 w_optimizer.step()
#
#                 # record
#                 base_prec1, base_prec5 = obtain_accuracy(
#                     logits.data, base_targets.data, topk=(1, 5)
#                 )
#                 base_losses.update(base_loss.item(), base_inputs.size(0))
#                 base_top1.update(base_prec1.item(), base_inputs.size(0))
#                 base_top5.update(base_prec5.item(), base_inputs.size(0))
#                 # update the architecture-weight
#                 a_optimizer.zero_grad()
#                 _, logits = network(arch_inputs.cuda())
#
#                 logits_teacher = teacher_model(arch_inputs.cuda())
#                 loss_KD_TA = T * T * nn.KLDivLoss()(F.log_softmax(logits / T, dim=1),
#                                                     F.softmax(logits_teacher / T, dim=1))
#                 arch_loss = (1 - lambda_) * criterion(logits, arch_targets) + lambda_ * loss_KD_TA
#                 arch_loss.backward()
#                 a_optimizer.step()
#                 # record
#                 arch_prec1, arch_prec5 = obtain_accuracy(
#                     logits.data, arch_targets.data, topk=(1, 5)
#                 )
#                 arch_losses.update(arch_loss.item(), arch_inputs.size(0))
#                 arch_top1.update(arch_prec1.item(), arch_inputs.size(0))
#                 arch_top5.update(arch_prec5.item(), arch_inputs.size(0))
#
#             elif training_mode == 1:
#                 network.train()
#                 w_optimizer.zero_grad()
#                 _, logits = network(base_inputs.cuda())
#                 logits_teacher = teacher_model(base_inputs.cuda())
#                 loss_KD_TA = T * T * nn.KLDivLoss()(F.log_softmax(logits / T, dim=1),
#                                                     F.softmax(logits_teacher / T, dim=1))
#                 base_loss = (1 - lambda_) * criterion(logits, base_targets) + lambda_ * loss_KD_TA
#                 base_loss.backward()
#                 torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
#
#                 w_optimizer.step()
#
#                 # record
#                 base_prec1, base_prec5 = obtain_accuracy(
#                     logits.data, base_targets.data, topk=(1, 5)
#                 )
#                 base_losses.update(base_loss.item(), base_inputs.size(0))
#                 base_top1.update(base_prec1.item(), base_inputs.size(0))
#                 base_top5.update(base_prec5.item(), base_inputs.size(0))
#                 # update the architecture-weight
#                 a_optimizer.zero_grad()
#                 _, logits = network(arch_inputs.cuda())
#
#                 logits_teacher = teacher_model(arch_inputs.cuda())
#                 loss_KD_TA = T * T * nn.KLDivLoss()(F.log_softmax(logits / T, dim=1),
#                                                     F.softmax(logits_teacher / T, dim=1))
#                 arch_loss = (1 - lambda_) * criterion(logits, arch_targets) + lambda_ * loss_KD_TA
#                 arch_loss.backward()
#                 a_optimizer.step()
#                 # record
#                 arch_prec1, arch_prec5 = obtain_accuracy(
#                     logits.data, arch_targets.data, topk=(1, 5)
#                 )
#                 arch_losses.update(arch_loss.item(), arch_inputs.size(0))
#                 arch_top1.update(arch_prec1.item(), arch_inputs.size(0))
#                 arch_top5.update(arch_prec5.item(), arch_inputs.size(0))
#
#                 # training student
#                 student_model.train()
#                 network.eval()
#                 student_optimizer.zero_grad()
#                 _, logits_ta = network(base_inputs.cuda())
#                 logits_student = student_model(base_inputs.cuda())
#                 loss_KD_Student = T * T * nn.KLDivLoss()(F.log_softmax(logits_student / T, dim=1),
#                                                          F.softmax(logits_ta / T, dim=1))
#                 student_loss = (1 - lambda_) * criterion(logits_student, base_targets) + lambda_ * loss_KD_Student
#                 student_loss.backward()
#                 # torch.nn.utils.clip_grad_norm_(student_model.parameters(), 5)
#                 student_optimizer.step()
#
#                 student_model.eval()
#                 logits_student_test = student_model(arch_inputs.cuda())
#
#                 student_prec1, student_prec5 = obtain_accuracy(
#                     logits_student_test.data, arch_targets.data, topk=(1, 5)
#                 )
#
#                 student_losses.update(student_loss.item(), base_inputs.size(0))
#                 student_top1.update(student_prec1.item(), arch_inputs.size(0))
#                 student_top5.update(student_prec5.item(), arch_inputs.size(0))
#
#             else:
#                 # training student
#                 student_model.train()
#                 network.eval()
#                 student_optimizer.zero_grad()
#                 _, logits_ta = network(base_inputs.cuda())
#                 logits_student = student_model(base_inputs.cuda())
#                 loss_KD_Student = T * T * nn.KLDivLoss()(F.log_softmax(logits_student / T, dim=1),
#                                                          F.softmax(logits_ta / T, dim=1))
#                 student_loss = (1 - lambda_) * criterion(logits_student, base_targets) + lambda_ * loss_KD_Student
#                 student_loss.backward()
#                 # torch.nn.utils.clip_grad_norm_(student_model.parameters(), 5)
#                 student_optimizer.step()
#
#                 student_model.eval()
#                 logits_student_test = student_model(arch_inputs.cuda())
#
#                 student_prec1, student_prec5 = obtain_accuracy(
#                     logits_student_test.data, arch_targets.data, topk=(1, 5)
#                 )
#
#                 student_losses.update(student_loss.item(), base_inputs.size(0))
#                 student_top1.update(student_prec1.item(), arch_inputs.size(0))
#                 student_top5.update(student_prec5.item(), arch_inputs.size(0))
#
#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()
#
#         if step % print_freq == 0 or step + 1 == len(xloader):
#             Sstr = (
#                 "*SEARCH* "
#                 + time_string()
#                 + " [{:}][{:03d}/{:03d}]".format(epoch_str, step, len(xloader))
#             )
#             Tstr = "Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})".format(
#                 batch_time=batch_time, data_time=data_time
#             )
#             Wstr = "Base [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]".format(
#                 loss=base_losses, top1=base_top1, top5=base_top5
#             )
#             Astr = "Arch [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]".format(
#                 loss=arch_losses, top1=arch_top1, top5=arch_top5
#             )
#
#             Studentstr = "Student [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]".format(
#                 loss=student_losses, top1=student_top1, top5=student_top5
#             )
#
#             logger.log(Sstr + " " + Tstr + " " + Wstr + " " + Astr + " " + Studentstr)
#
#     return base_losses.avg, base_top1.avg, base_top5.avg, arch_losses.avg, arch_top1.avg, arch_top5.avg, student_losses.avg, student_top1.avg, student_top5.avg


def main(xargs):
    assert torch.cuda.is_available(), "CUDA is not available."
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(xargs.workers)
    prepare_seed(xargs.rand_seed)
    logger = prepare_logger(args)

    train_data, valid_data, xshape, class_num = get_datasets(
        xargs.dataset, xargs.data_path, -1
    )
    # config_path = 'configs/nas-benchmark/algos/GDAS.config'
    config = load_config(
        xargs.config_path, {"class_num": class_num, "xshape": xshape}, logger
    )
    search_loader, _, valid_loader = get_nas_search_loaders(
        train_data,
        valid_data,
        xargs.dataset,
        "../../configs/nas-benchmark",
        config.batch_size,
        xargs.workers,
    )
    teacher_model = xargs.teacher_model
    TA = xargs.TA
    student = xargs.student_model
    teacher_checkpoint = xargs.teacher_checkpoint
    student_checkpoint = xargs.student_checkpoint
    epoch_online = xargs.epoch_online
    logger.log(
        "||||||| {:10s} ||||||| Search-Loader-Num={:}, batch size={:}".format(
            xargs.dataset, len(search_loader), config.batch_size
        )
    )
    logger.log("||||||| {:10s} ||||||| Config={:}".format(xargs.dataset, config))

    search_space = get_search_spaces("cell", xargs.search_space_name)
    if xargs.model_config is None:
        model_config = dict2config(
            {
                "name": "GDAS",
                "C": xargs.channel,
                "N": xargs.num_cells,
                "max_nodes": xargs.max_nodes,
                "num_classes": class_num,
                "space": search_space,
                "affine": False,
                "track_running_stats": bool(xargs.track_running_stats),
            },
            None,
        )
    else:
        model_config = load_config(
            xargs.model_config,
            {
                "num_classes": class_num,
                "space": search_space,
                "affine": False,
                "track_running_stats": bool(xargs.track_running_stats),
            },
            None,
        )
    search_model = get_cell_based_tiny_net(model_config)
    logger.log("search-model :\n{:}".format(search_model))
    logger.log("model-config : {:}".format(model_config))

    student_accuracy = {'best': -1}
    TA_accuracy = {'best': -1}

    w_optimizer, w_scheduler, criterion = get_optim_scheduler(
        search_model.get_weights(), config
    )
    a_optimizer = torch.optim.Adam(
        search_model.get_alphas(),
        lr=xargs.arch_learning_rate,
        betas=(0.5, 0.999),
        weight_decay=xargs.arch_weight_decay,
    )
    logger.log("w-optimizer : {:}".format(w_optimizer))
    logger.log("a-optimizer : {:}".format(a_optimizer))
    logger.log("w-scheduler : {:}".format(w_scheduler))
    logger.log("criterion   : {:}".format(criterion))
    flop, param = get_model_infos(search_model, xshape)
    logger.log("FLOP = {:.2f} M, Params = {:.2f} MB".format(flop, param))
    logger.log("search-space [{:} ops] : {:}".format(len(search_space), search_space))
    if xargs.arch_nas_dataset is None:
        api = None
    else:
        api = API(xargs.arch_nas_dataset)
    logger.log("{:} create API = {:} done".format(time_string(), api))

    last_info, model_base_path, model_best_path = (
        logger.path("info"),
        logger.path("model"),
        logger.path("best"),
    )
    network, criterion = torch.nn.DataParallel(search_model).cuda(), criterion.cuda()

    if last_info.exists():  # automatically resume from previous checkpoint
        logger.log(
            "=> loading checkpoint of the last-info '{:}' start".format(last_info)
        )
        last_info = torch.load(last_info)
        start_epoch = last_info["epoch"]
        checkpoint = torch.load(last_info["last_checkpoint"])
        genotypes = checkpoint["genotypes"]
        valid_accuracies = checkpoint["valid_accuracies"]
        search_model.load_state_dict(checkpoint["search_model"])
        w_scheduler.load_state_dict(checkpoint["w_scheduler"])
        w_optimizer.load_state_dict(checkpoint["w_optimizer"])
        a_optimizer.load_state_dict(checkpoint["a_optimizer"])
        logger.log(
            "=> loading checkpoint of the last-info '{:}' start with {:}-th epoch.".format(
                last_info, start_epoch
            )
        )
    else:
        logger.log("=> do not find the last-info file : {:}".format(last_info))
        start_epoch, valid_accuracies, genotypes = (
            0,
            {"best": -1},
            {-1: search_model.genotype()},
        )

    # start training
    start_time, search_time, epoch_time, total_epoch = (
        time.time(),
        AverageMeter(),
        AverageMeter(),
        config.epochs + config.warmup,
    )

    teacher_model, teacher_optimizer, teacher_scheduler = create_cnn_model(teacher_model, train_data, total_epoch, teacher_checkpoint, use_cuda=1)
    # if teacher_checkpoint:
    # teacher_model = load_checkpoint(teacher_model, teacher_checkpoint)
    # else:
    #     teacher_model = train_teacher(search_loader, teacher_model, criterion, teacher_optimizer, logger, total_epoch, xargs.teacher_model)

    # if TA:
    student_model, student_optimizer, student_scheduler = create_cnn_model(student, train_data, total_epoch, student_checkpoint, use_cuda=1)
    # if student_checkpoint:
    # student_model = load_checkpoint(student_model, student_checkpoint)
    # checkpoint = torch.load(student_checkpoint)
    # student_optimizer.load_state_dict(checkpoint['optimizer'])

    # if TA != 'GDAS':
    #     network, w_optimizer = create_cnn_model(TA, train_data, use_cuda=1)
    #     # w_optimizer = torch.optim.Adam(network .parameters(), lr=0.025, betas=(0.5, 0.999), weight_decay=1e-4)
    #     a_optimizer = None

    for epoch in range(start_epoch, total_epoch):
        w_scheduler.update(epoch, 0.0)
        need_time = "Time Left: {:}".format(
            convert_secs2time(epoch_time.val * (total_epoch - epoch), True)
        )
        epoch_str = "{:03d}-{:03d}".format(epoch, total_epoch)
        search_model.set_tau(
            xargs.tau_max - (xargs.tau_max - xargs.tau_min) * epoch / (total_epoch - 1)
        )
        logger.log(
            "\n[Search the {:}-th epoch] {:}, tau={:}, LR={:}".format(
                epoch_str, need_time, search_model.get_tau(), min(w_scheduler.get_lr())
            )
        )

        search_w_loss, search_w_top1, search_w_top5, valid_a_loss, valid_a_top1, valid_a_top5 = search_func_modified(
            search_loader,
            teacher_model,
            network,
            student_model,
            criterion,
            w_scheduler,
            w_optimizer,
            a_optimizer,
            epoch_str,
            xargs.print_freq,
            logger,
            )
        # else:
        #     Student_optimizer = None
        #     training_mode = 0
        #     search_w_loss, search_w_top1, search_w_top5, valid_a_loss, valid_a_top1, valid_a_top5, student_loss, student_top1, student_top5 = search_func(search_loader,
        #                                                                                                                                                     teacher_model,
        #                                                                                                                                                     network,
        #                                                                                                                                                     student_model,
        #                                                                                                                                                     criterion,
        #                                                                                                                                                     w_scheduler,
        #                                                                                                                                                     w_optimizer,
        #                                                                                                                                                     a_optimizer,
        #                                                                                                                                                     Student_optimizer,

        search_time.update(time.time() - start_time)
        logger.log(
            "[{:}] searching : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%, time-cost={:.1f} s".format(
                epoch_str, search_w_loss, search_w_top1, search_w_top5, search_time.sum
            )
        )
        logger.log(
            "[{:}] evaluate  : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%".format(
                epoch_str, valid_a_loss, valid_a_top1, valid_a_top5
            )
        )

        # logger.log(
        #     "[{:}] student  : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%".format(
        #         epoch_str, student_loss, student_top1, student_top5
        #     )
        # )

        # check the best accuracy

        # student_accuracy[epoch] = student_top1
        # if student_top1 > student_accuracy['best']:
        #     student_accuracy['best'] = student_top1

        TA_accuracy[epoch] = search_w_top1
        if search_w_top1 > TA_accuracy['best']:
            TA_accuracy['best'] = search_w_top1

        valid_accuracies[epoch] = valid_a_top1
        if valid_a_top1 > valid_accuracies["best"]:
            valid_accuracies["best"] = valid_a_top1
            genotypes["best"] = search_model.genotype()
            find_best = True
        else:
            find_best = False

        genotypes[epoch] = search_model.genotype()
        logger.log(
            "<<<--->>> The {:}-th epoch : {:}".format(epoch_str, genotypes[epoch])
        )
        # save checkpoint

        save_path = save_checkpoint(
            {
                "epoch": epoch + 1,
                "args": deepcopy(xargs),
                "search_model": search_model.state_dict(),
                "w_optimizer": w_optimizer.state_dict(),
                "a_optimizer": a_optimizer.state_dict(),
                "w_scheduler": w_scheduler.state_dict(),
                "genotypes": genotypes,
                "valid_accuracies": valid_accuracies,
            },
            model_base_path,
            logger,
        )
        last_info = save_checkpoint(
            {
                "epoch": epoch + 1,
                "args": deepcopy(args),
                "last_checkpoint": save_path,
            },
            logger.path("info"),
            logger,
        )
        if find_best:
            logger.log(
                "<<<--->>> The {:}-th epoch : find the highest validation accuracy : {:.2f}%.".format(
                    epoch_str, valid_a_top1
                )
            )
            copy_checkpoint(model_base_path, model_best_path, logger)
        with torch.no_grad():
            logger.log("{:}".format(search_model.show_alphas()))
        if api is not None:
            logger.log("{:}".format(api.query_by_arch(genotypes[epoch], "200")))
            # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()

        # if TA!='GDAS':
        #     student_model, student_optimizer = create_cnn_model(student, train_data, use_cuda=1)
        #
        # student_best = -1
        # for epoch in range(start_epoch, total_epoch):
        #     student_loss, student_top1, student_top5 = train_student(search_loader,
        #         network,
        #         student_model,
        #         criterion,
        #         student_optimizer,
        #         epoch_str,
        #         xargs.print_freq,
        #         logger,)
        #
        #     student_accuracy[epoch] = student_top1
        #     if student_top1 > student_accuracy['best']:
        #         student_accuracy['best'] = student_top1



        logger.log("\n" + "-" * 100)
        # check the performance from the architecture dataset
        if TA == 'GDAS':
            logger.log(
                "GDAS : run {:} epochs, cost {:.1f} s, last-geno is {:}.".format(
                    total_epoch, search_time.sum, genotypes[total_epoch - 1]
                )
            )
            if api is not None:
                logger.log("{:}".format(api.query_by_arch(genotypes[total_epoch - 1], "200")))


        logger.log('----------------')

        logger.log('we used {:} as our Teacher with param size {:}'.format(xargs.teacher_model, count_parameters_in_MB(teacher_model)))
        logger.log('we used {:} as our TA with param size {:}'.format(TA, count_parameters_in_MB(network)))
        logger.log('we used {:} as our Student with param size {:}'.format(xargs.student_model, count_parameters_in_MB(student_model)))

        logger.log('we used {:} online epochs out of total epochs of {:}'.format(xargs.epoch_online, total_epoch))
        logger.log('The best ACC of TA: {:.2f}%'.format(TA_accuracy['best']))
        logger.log('The best ACC of Student: {:.2f}%'.format(student_accuracy['best']))
        logger.log('----------------')

        logger.close()

    # else:
    #     if student:
    #         student_model, student_optimizer = create_cnn_model(student, train_data, use_cuda=1)
    #         student_best = -1
    #         for epoch in range(start_epoch, total_epoch):
    #             epoch_str = "{:03d}-{:03d}".format(epoch, total_epoch)
    #             student_loss, student_top1, student_top5 = train_student(search_loader,
    #                                                                      teacher_model,
    #                                                                      student_model,
    #                                                                      criterion,
    #                                                                      student_optimizer,
    #                                                                      epoch_str,
    #                                                                      xargs.print_freq,
    #                                                                      logger, )
    #
    #             student_accuracy[epoch] = student_top1
    #             if student_top1 > student_accuracy['best']:
    #                 student_accuracy['best'] = student_top1
    #
    #         logger.log('----------------')
    #         logger.log('we used {:} as our Teacher with param size {:}'.format(xargs.teacher_model, count_parameters_in_MB(teacher_model)))
    #         logger.log('we used {:} as our TA with param size {:}'.format(TA, count_parameters_in_MB(network)))
    #         logger.log('we used {:} as our Student with param size {:}'.format(xargs.student_model, count_parameters_in_MB(student_model)))
    #         logger.log('we used {:} online epochs out of total epochs of {:}'.format(xargs.epoch_online, total_epoch))
    #         logger.log('The best ACC of  : {:.2f}%'.format(TA_accuracy['best']))
    #         logger.log('The best ACC of Student: {:.2f}%'.format(student_accuracy['best']))
    #         logger.log('----------------')
    #         logger.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("GDAS")
    parser.add_argument("--data_path", default='../../data', type=str, help="Path to dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default= 'cifar10',
        choices=["cifar10", "cifar100", "ImageNet16-120"],
        help="Choose between Cifar10/100 and ImageNet-16.",
    )
    # channels and number-of-cells
    parser.add_argument("--search_space_name", default='darts', type=str, help="The search space name.")
    parser.add_argument("--max_nodes", type=int, help="The maximum number of nodes.")
    parser.add_argument("--channel", type=int, help="The number of channels.")
    parser.add_argument(
        "--num_cells", type=int, help="The number of cells in one "
                                      "。stage."
    )
    parser.add_argument(
        "--track_running_stats",
        type=int,
        default=1,
        choices=[0, 1],
        help="Whether use track_running_stats or not in the BN layer.",
    )
    parser.add_argument(
        "--config_path", default='../../configs/search-opts/GDAS-NASNet-CIFAR.config', type=str, help="The path of the configuration."
    )
    parser.add_argument(
        "--model_config",
        default='../../configs/search-archs/GDASFRC-NASNet-CIFAR.config',
        type=str,
        help="The path of the model configuration. When this arg is set, it will cover max_nodes / channels / num_cells.",
    )
    # architecture leraning rate
    parser.add_argument(
        "--arch_learning_rate",
        type=float,
        default=3e-4,
        help="learning rate for arch encoding",
    )
    parser.add_argument(
        "--arch_weight_decay",
        type=float,
        default=1e-3,
        help="weight decay for arch encoding",
    )
    parser.add_argument("--tau_min", default=10, type=float, help="The minimum tau for Gumbel")
    parser.add_argument("--tau_max", default=0.1, type=float, help="The maximum tau for Gumbel")
    # log
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="number of data loading workers (default: 2)",
    )
    parser.add_argument(
        "--save_dir", default='./output/search-cell-dar/GDAS-cifar10-BN1', type=str, help="Folder to save checkpoints and log."
    )
    parser.add_argument(
        "--arch_nas_dataset",
        type=str,
        help="The path to load the architecture dataset   (tiny-nas-benchmark).",
    )
    parser.add_argument("--print_freq", default=200, type=int, help="print frequency (default: 200)")
    parser.add_argument("--rand_seed", default= -1, type=int, help="manual seed")
    parser.add_argument("--teacher_model", default="resnet110", type=str, help="type of teacher mode")
    parser.add_argument("--TA", default='GDAS', type=str, help="type of TA")
    parser.add_argument("--student_model", default='plane2', type=str, help="type of student mode")
    parser.add_argument("--teacher_checkpoint", default='../output/nas-infer/cifar10-BS96-gdas_serached/checkpoint/seed-21045-best_resnet110_95.56%_07-05,22.pth', type=str, help="teacher mode's check point")
    parser.add_argument("--student_checkpoint", default='../output/nas-infer/cifar10-BS96-gdas_serached/checkpoint/seed-53972-best_plane2_69.40%_07-07,03.pth', type=str,
                        help="student mode's check point")
    parser.add_argument("--epoch_online", default=250, type=int, help="online training of TA and student")
    args = parser.parse_args()
    if args.rand_seed is None or args.rand_seed < 0:
        args.rand_seed = random.randint(1, 100000)
    # #
    # #
    # teacher_models = ['resnet110', 'resnet56', 'resnet44', 'resnet32', 'resnet26', 'resnet20', 'resnet14', 'resnet8', 'plane10', 'plane8', 'plane  6','plane4','plane2' ]
    # teacher_models = ['vgg16']
    # for one in teacher_models:
    #     args.teacher_model = one

    # TA_models = ['plane4', 'plane6', 'resnet26', 'resnet20']
    # for one in TA_models:
    #     args.TA = one
    main(args)
