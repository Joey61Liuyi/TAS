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
import torch.nn.functional as F
import os
import torch.nn.init as init
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


def adjust_learning_rate(optimizer, epoch, epoch_total):

    # depending on dataset
    if epoch < int(epoch_total / 2.0):
        lr = 0.1
    elif epoch < int(epoch_total * 3 / 4.0):
        lr = 0.1 * 0.1
    else:
        lr = 0.1 * 0.01

    # update optimizer's learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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

def train_teacher(xloader,teacher_model, criterion, optimizer, logger, total_epochs, teacher_name):

    best_acc = -1

    for epoch in range(total_epochs):
        data_time, batch_time = AverageMeter(), AverageMeter()
        base_losses, base_top1, base_top5 = AverageMeter(), AverageMeter(), AverageMeter()
        end = time.time()
        for step, (base_inputs, base_targets, arch_inputs, arch_targets) in enumerate(
                xloader
        ):
            teacher_model.train()
            base_targets = base_targets.cuda(non_blocking=True)
            arch_targets = arch_targets.cuda(non_blocking=True)
            data_time.update(time.time() - end)
            optimizer.zero_grad()
            logits_teacher = teacher_model(base_inputs.cuda())
            loss = criterion(logits_teacher, base_targets)
            loss.backward()
            optimizer.step()

            teacher_model.eval()
            logits_teacher = teacher_model(arch_inputs.cuda())
            Teacher_prec1, Teacher_prec5 = obtain_accuracy(
                logits_teacher, arch_targets.data, topk = (1,5)
            )
            base_losses.update(loss.item(), base_inputs.size(0))
            base_top1.update(Teacher_prec1.item(), arch_inputs.size(0))
            base_top5.update(Teacher_prec5.item(), arch_inputs.size(0))

        logger.log('Training Teacher {:} at {:} epoch, the current acc is {:.2f}%, and loss is {:}'.format(teacher_name, epoch, base_top1.avg, base_losses.avg))
        if 'resnet' in teacher_name:
            adjust_learning_rate(optimizer, epoch, total_epochs)

        if base_top1.avg >= best_acc:
            g = os.walk(os.path.abspath('.'))
            for path, dir_list, file_list in g:
                for file_name in file_list:
                    if 'Teacher_model_{}_{:.2f}%'.format(teacher_name, best_acc) in file_name:
                        tep = os.path.join(path, file_name)
                        os.remove(tep)
            best_acc = base_top1.avg
            torch.save({
                'model_state_dict': teacher_model.state_dict(),
                'optimizer_dict': optimizer.state_dict(),
                'epoch': epoch
            }, 'Teacher_model_{}_{:.2f}%_{}.pth.tar'.format(teacher_name, best_acc, time.strftime("%m-%d,%H", time.localtime()))
            )


def train_student(xloader,
    network,
    student_model,
    criterion,
    optimizer,
    epoch_str,
    print_freq,
    logger,):
    network.eval()
    data_time, batch_time = AverageMeter(), AverageMeter()
    base_losses, base_top1, base_top5 = AverageMeter(), AverageMeter(), AverageMeter()
    end = time.time()
    T = 5
    lambda_ = 0.5

    for step, (base_inputs, base_targets, arch_inputs, arch_targets) in enumerate(
            xloader
    ):
        network.eval()
        student_model.train()
        base_targets = base_targets.cuda(non_blocking=True)
        arch_targets = arch_targets.cuda(non_blocking=True)
        # measure data loading time
        data_time.update(time.time() - end)
        optimizer.zero_grad()

        try:
            _, logits_TA = network(base_inputs.cuda())
        except:
            logits_TA = network(base_inputs.cuda())

        logits_student = student_model(base_inputs.cuda())
        loss_KD_TA = T * T * nn.KLDivLoss()(F.log_softmax(logits_student / T, dim=1),
                                            F.softmax(logits_TA / T, dim=1))
        base_loss = (1 - lambda_) * criterion(logits_student, base_targets) + lambda_ * loss_KD_TA
        base_loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
        optimizer.step()

        # record

        student_model.eval()
        logits_student = student_model(arch_inputs.cuda())

        base_prec1, base_prec5 = obtain_accuracy(
            logits_student.data, arch_targets.data, topk=(1, 5)
        )

        base_losses.update(base_loss.item(), base_inputs.size(0))
        base_top1.update(base_prec1.item(), base_inputs.size(0))
        base_top5.update(base_prec5.item(), base_inputs.size(0))
        # update the architecture-weight

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
            Wstr = "Student_Final [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]".format(
                loss=base_losses, top1=base_top1, top5=base_top5
            )

            logger.log(Sstr + " " + Tstr + " " + Wstr)

    return base_losses.avg, base_top1.avg, base_top5.avg


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


def search_func(
    xloader,
    teacher_model,
    network,
    student_model,
    criterion,
    scheduler,
    w_optimizer,
    a_optimizer,
    student_optimizer,
    epoch_str,
    print_freq,
    logger,
    training_mode
):

    teacher_model.eval()
    data_time, batch_time = AverageMeter(), AverageMeter()
    base_losses, base_top1, base_top5 = AverageMeter(), AverageMeter(), AverageMeter()
    arch_losses, arch_top1, arch_top5 = AverageMeter(), AverageMeter(), AverageMeter()
    student_losses, student_top1, student_top5 = AverageMeter(), AverageMeter(), AverageMeter()
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

        # pretraining TA


        if a_optimizer == None:
            if training_mode == 0:
                network.train()
                w_optimizer.zero_grad()
                logits_ta = network(base_inputs.cuda())
                logits_teacher = teacher_model(base_inputs.cuda())
                loss_KD_TA = T * T * nn.KLDivLoss()(F.log_softmax(logits_ta / T, dim=1),
                                                    F.softmax(logits_teacher / T, dim=1))
                TA_loss = (1 - lambda_) * criterion(logits_ta, base_targets) + lambda_ * loss_KD_TA
                TA_loss.backward()
                # torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
                w_optimizer.step()

                network.eval()
                logits_TA_test = network(arch_inputs.cuda())

                TA_prec1, TA_prec5 = obtain_accuracy(
                    logits_TA_test.data, arch_targets.data, topk=(1, 5)
                )

                base_losses.update(TA_loss.item(), base_inputs.size(0))
                base_top1.update(TA_prec1.item(), arch_inputs.size(0))
                base_top5.update(TA_prec5.item(), arch_inputs.size(0))

            elif training_mode == 1:
                network.train()
                w_optimizer.zero_grad()
                logits_ta = network(base_inputs.cuda())
                logits_teacher = teacher_model(base_inputs.cuda())
                loss_KD_TA = T * T * nn.KLDivLoss()(F.log_softmax(logits_ta / T, dim=1),
                                                    F.softmax(logits_teacher / T, dim=1))
                TA_loss = (1 - lambda_) * criterion(logits_ta, base_targets) + lambda_ * loss_KD_TA
                TA_loss.backward()
                # torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
                w_optimizer.step()

                network.eval()
                logits_TA_test = network(arch_inputs.cuda())

                TA_prec1, TA_prec5 = obtain_accuracy(
                    logits_TA_test.data, arch_targets.data, topk=(1, 5)
                )

                base_losses.update(TA_loss.item(), base_inputs.size(0))
                base_top1.update(TA_prec1.item(), arch_inputs.size(0))
                base_top5.update(TA_prec5.item(), arch_inputs.size(0))

                # Training Student
                student_model.train()
                network.eval()
                student_optimizer.zero_grad()
                logits_ta = network(base_inputs.cuda())
                logits_student = student_model(base_inputs.cuda())
                loss_KD_Student = T * T * nn.KLDivLoss()(F.log_softmax(logits_student / T, dim=1),
                                                         F.softmax(logits_ta / T, dim=1))
                student_loss = (1 - lambda_) * criterion(logits_student, base_targets) + lambda_ * loss_KD_Student
                student_loss.backward()
                # torch.nn.utils.clip_grad_norm_(student_model.parameters(), 5)
                student_optimizer.step()

                student_model.eval()
                logits_student_test = student_model(arch_inputs.cuda())

                student_prec1, student_prec5 = obtain_accuracy(
                    logits_student_test.data, arch_targets.data, topk=(1, 5)
                )

                student_losses.update(student_loss.item(), base_inputs.size(0))
                student_top1.update(student_prec1.item(), arch_inputs.size(0))
                student_top5.update(student_prec5.item(), arch_inputs.size(0))

            else:
                student_model.train()
                network.eval()
                student_optimizer.zero_grad()
                logits_ta = network(base_inputs.cuda())
                logits_student = student_model(base_inputs.cuda())
                loss_KD_Student = T * T * nn.KLDivLoss()(F.log_softmax(logits_student / T, dim=1),
                                                         F.softmax(logits_ta / T, dim=1))
                student_loss = (1 - lambda_) * criterion(logits_student, base_targets) + lambda_ * loss_KD_Student
                student_loss.backward()
                # torch.nn.utils.clip_grad_norm_(student_model.parameters(), 5)
                student_optimizer.step()

                student_model.eval()
                logits_student_test = student_model(arch_inputs.cuda())

                student_prec1, student_prec5 = obtain_accuracy(
                    logits_student_test.data, arch_targets.data, topk=(1, 5)
                )

                student_losses.update(student_loss.item(), base_inputs.size(0))
                student_top1.update(student_prec1.item(), arch_inputs.size(0))
                student_top5.update(student_prec5.item(), arch_inputs.size(0))

        else:
            if training_mode == 0:
                network.train()
                w_optimizer.zero_grad()
                _, logits = network(base_inputs.cuda())
                logits_teacher = teacher_model(base_inputs.cuda())
                loss_KD_TA = T * T * nn.KLDivLoss()(F.log_softmax(logits / T, dim=1),
                                                    F.softmax(logits_teacher / T, dim=1))
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
                loss_KD_TA = T * T * nn.KLDivLoss()(F.log_softmax(logits / T, dim=1),
                                                    F.softmax(logits_teacher / T, dim=1))
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

            elif training_mode == 1:
                network.train()
                w_optimizer.zero_grad()
                _, logits = network(base_inputs.cuda())
                logits_teacher = teacher_model(base_inputs.cuda())
                loss_KD_TA = T * T * nn.KLDivLoss()(F.log_softmax(logits / T, dim=1),
                                                    F.softmax(logits_teacher / T, dim=1))
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
                loss_KD_TA = T * T * nn.KLDivLoss()(F.log_softmax(logits / T, dim=1),
                                                    F.softmax(logits_teacher / T, dim=1))
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

                # training student
                student_model.train()
                network.eval()
                student_optimizer.zero_grad()
                _, logits_ta = network(base_inputs.cuda())
                logits_student = student_model(base_inputs.cuda())
                loss_KD_Student = T * T * nn.KLDivLoss()(F.log_softmax(logits_student / T, dim=1),
                                                         F.softmax(logits_ta / T, dim=1))
                student_loss = (1 - lambda_) * criterion(logits_student, base_targets) + lambda_ * loss_KD_Student
                student_loss.backward()
                # torch.nn.utils.clip_grad_norm_(student_model.parameters(), 5)
                student_optimizer.step()

                student_model.eval()
                logits_student_test = student_model(arch_inputs.cuda())

                student_prec1, student_prec5 = obtain_accuracy(
                    logits_student_test.data, arch_targets.data, topk=(1, 5)
                )

                student_losses.update(student_loss.item(), base_inputs.size(0))
                student_top1.update(student_prec1.item(), arch_inputs.size(0))
                student_top5.update(student_prec5.item(), arch_inputs.size(0))

            else:
                # training student
                student_model.train()
                network.eval()
                student_optimizer.zero_grad()
                _, logits_ta = network(base_inputs.cuda())
                logits_student = student_model(base_inputs.cuda())
                loss_KD_Student = T * T * nn.KLDivLoss()(F.log_softmax(logits_student / T, dim=1),
                                                         F.softmax(logits_ta / T, dim=1))
                student_loss = (1 - lambda_) * criterion(logits_student, base_targets) + lambda_ * loss_KD_Student
                student_loss.backward()
                # torch.nn.utils.clip_grad_norm_(student_model.parameters(), 5)
                student_optimizer.step()

                student_model.eval()
                logits_student_test = student_model(arch_inputs.cuda())

                student_prec1, student_prec5 = obtain_accuracy(
                    logits_student_test.data, arch_targets.data, topk=(1, 5)
                )

                student_losses.update(student_loss.item(), base_inputs.size(0))
                student_top1.update(student_prec1.item(), arch_inputs.size(0))
                student_top5.update(student_prec5.item(), arch_inputs.size(0))

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

            Studentstr = "Student [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]".format(
                loss=student_losses, top1=student_top1, top5=student_top5
            )

            logger.log(Sstr + " " + Tstr + " " + Wstr + " " + Astr + " " + Studentstr)

    return base_losses.avg, base_top1.avg, base_top5.avg, arch_losses.avg, arch_top1.avg, arch_top5.avg, student_losses.avg, student_top1.avg, student_top5.avg


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

    def conv3x3(in_planes, out_planes, stride=1):
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

    class ResNet_Cifar(nn.Module):

        def __init__(self, block, layers, num_classes=10):
            super(ResNet_Cifar, self).__init__()
            self.inplanes = 16
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(16)
            self.relu = nn.ReLU(inplace=True)
            self.layer1 = self._make_layer(block, 16, layers[0])
            self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
            self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
            self.avgpool = nn.AvgPool2d(8, stride=1)
            self.fc = nn.Linear(64 * block.expansion, num_classes)

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

        def _make_layer(self, block, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion)
                )

            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes))

            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

            return x

    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, inplanes, planes, stride=1, downsample=None):
            super(BasicBlock, self).__init__()
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = conv3x3(planes, planes)
            self.bn2 = nn.BatchNorm2d(planes)
            self.downsample = downsample
            self.stride = stride

        def forward(self, x):
            residual = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual
            out = self.relu(out)

            return out

    class ConvNetMaker(nn.Module):
        """
        Creates a simple (plane) convolutional neural network
        """

        def __init__(self, layers):
            """
            Makes a cnn using the provided list of layers specification
            The details of this list is available in the paper
            :param layers: a list of strings, representing layers like ["CB32", "CB32", "FC10"]
            """
            super(ConvNetMaker, self).__init__()
            self.conv_layers = []
            self.fc_layers = []
            h, w, d = 32, 32, 3
            previous_layer_filter_count = 3
            previous_layer_size = h * w * d
            num_fc_layers_remained = len([1 for l in layers if l.startswith('FC')])
            for layer in layers:
                if layer.startswith('Conv'):
                    filter_count = int(layer[4:])
                    self.conv_layers += [nn.Conv2d(previous_layer_filter_count, filter_count, kernel_size=3, padding=1),
                                         nn.BatchNorm2d(filter_count), nn.ReLU(inplace=True)]
                    previous_layer_filter_count = filter_count
                    d = filter_count
                    previous_layer_size = h * w * d
                elif layer.startswith('MaxPool'):
                    self.conv_layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                    h, w = int(h / 2.0), int(w / 2.0)
                    previous_layer_size = h * w * d
                elif layer.startswith('FC'):
                    num_fc_layers_remained -= 1
                    current_layer_size = int(layer[2:])
                    if num_fc_layers_remained == 0:
                        self.fc_layers += [nn.Linear(previous_layer_size, current_layer_size)]
                    else:
                        self.fc_layers += [nn.Linear(previous_layer_size, current_layer_size), nn.ReLU(inplace=True)]
                    previous_layer_size = current_layer_size

            conv_layers = self.conv_layers
            fc_layers = self.fc_layers
            self.conv_layers = nn.Sequential(*conv_layers)
            self.fc_layers = nn.Sequential(*fc_layers)

        def forward(self, x):
            x = self.conv_layers(x)
            x = x.view(x.size(0), -1)
            x = self.fc_layers(x)
            return x

    def resnet14_cifar(**kwargs):
        model = ResNet_Cifar(BasicBlock, [2, 2, 2], **kwargs)
        return model

    def resnet8_cifar(**kwargs):
        model = ResNet_Cifar(BasicBlock, [1, 1, 1], **kwargs)
        return model

    def resnet20_cifar(**kwargs):
        model = ResNet_Cifar(BasicBlock, [3, 3, 3], **kwargs)
        return model

    def resnet26_cifar(**kwargs):
        model = ResNet_Cifar(BasicBlock, [4, 4, 4], **kwargs)
        return model

    def resnet32_cifar(**kwargs):
        model = ResNet_Cifar(BasicBlock, [5, 5, 5], **kwargs)
        return model

    def resnet44_cifar(**kwargs):
        model = ResNet_Cifar(BasicBlock, [7, 7, 7], **kwargs)
        return model

    def resnet56_cifar(**kwargs):
        model = ResNet_Cifar(BasicBlock, [9, 9, 9], **kwargs)
        return model

    def resnet110_cifar(**kwargs):
        model = ResNet_Cifar(BasicBlock, [18, 18, 18], **kwargs)
        return model

    class VGG(nn.Module):
        def __init__(self, features, output_dim):
            super().__init__()

            self.features = features

            self.avgpool = nn.AdaptiveAvgPool2d(7)

            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, output_dim),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            h = x.view(x.shape[0], -1)
            x = self.classifier(h)
            return x

    def get_vgg_layers(config, batch_norm):

        layers = []
        in_channels = 3

        for c in config:
            assert c == 'M' or isinstance(c, int)
            if c == 'M':
                layers += [nn.MaxPool2d(kernel_size=2)]
            else:
                conv2d = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = c

        return nn.Sequential(*layers)

    def vgg11_cifar(**kwargs):
        model = VGG(get_vgg_layers(vgg11_config, batch_norm=True), class_num)
        return model

    def vgg13_cifar(**kwargs):
        model = VGG(get_vgg_layers(vgg13_config, batch_norm=True), class_num)
        return model

    def vgg16_cifar(**kwargs):
        model = VGG(get_vgg_layers(vgg16_config, batch_norm=True), class_num)
        return model

    def vgg19_cifar(**kwargs):
        model = VGG(get_vgg_layers(vgg19_config, batch_norm=True), class_num)
        return model

    class AlexNet(nn.Module):

        def __init__(self, num_classes):
            super(AlexNet, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(in_channels=3,
                          out_channels=64,
                          kernel_size=3,
                          stride=1,
                          padding=1),

                nn.ReLU(inplace=True),

                nn.MaxPool2d(kernel_size=2,
                             stride=2),

                nn.Conv2d(in_channels=64,
                          out_channels=192,
                          kernel_size=3,
                          stride=1,
                          padding=1),

                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2,
                             stride=2),

                nn.Conv2d(in_channels=192,
                          out_channels=384,
                          kernel_size=3,
                          stride=1,
                          padding=1),

                nn.ReLU(inplace=True),

                nn.Conv2d(in_channels=384,
                          out_channels=256,
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.ReLU(inplace=True),

                nn.Conv2d(in_channels=256,
                          out_channels=256,
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2,
                             stride=2),
            )
            self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )

        # forward: forward propagation
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x

    class LeNet(nn.Module):
        def __init__(self):
            super(LeNet, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
            self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    class Inception(nn.Module):
        def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
            super(Inception, self).__init__()
            # 1x1 conv branch
            self.b1 = nn.Sequential(
                nn.Conv2d(in_planes, n1x1, kernel_size=1),
                nn.BatchNorm2d(n1x1),
                nn.ReLU(True),
            )

            # 1x1 conv -> 3x3 conv branch
            self.b2 = nn.Sequential(
                nn.Conv2d(in_planes, n3x3red, kernel_size=1),
                nn.BatchNorm2d(n3x3red),
                nn.ReLU(True),
                nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
                nn.BatchNorm2d(n3x3),
                nn.ReLU(True),
            )

            # 1x1 conv -> 5x5 conv branch
            self.b3 = nn.Sequential(
                nn.Conv2d(in_planes, n5x5red, kernel_size=1),
                nn.BatchNorm2d(n5x5red),
                nn.ReLU(True),
                nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
                nn.BatchNorm2d(n5x5),
                nn.ReLU(True),
                nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
                nn.BatchNorm2d(n5x5),
                nn.ReLU(True),
            )

            # 3x3 pool -> 1x1 conv branch
            self.b4 = nn.Sequential(
                nn.MaxPool2d(3, stride=1, padding=1),
                nn.Conv2d(in_planes, pool_planes, kernel_size=1),
                nn.BatchNorm2d(pool_planes),
                nn.ReLU(True),
            )

        def forward(self, x):
            y1 = self.b1(x)
            y2 = self.b2(x)
            y3 = self.b3(x)
            y4 = self.b4(x)
            return torch.cat([y1, y2, y3, y4], 1)

    class GoogLeNet(nn.Module):
        def __init__(self):
            super(GoogLeNet, self).__init__()
            self.pre_layers = nn.Sequential(
                nn.Conv2d(3, 192, kernel_size=3, padding=1),
                nn.BatchNorm2d(192),
                nn.ReLU(True),
            )

            self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
            self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

            self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

            self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
            self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
            self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
            self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
            self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

            self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
            self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

            self.avgpool = nn.AvgPool2d(8, stride=1)
            self.linear = nn.Linear(1024, 10)

        def forward(self, x):
            out = self.pre_layers(x)
            out = self.a3(out)
            out = self.b3(out)
            out = self.maxpool(out)
            out = self.a4(out)
            out = self.b4(out)
            out = self.c4(out)
            out = self.d4(out)
            out = self.e4(out)
            out = self.maxpool(out)
            out = self.a5(out)
            out = self.b5(out)
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out

    class MobileNet_Block(nn.Module):
        '''Depthwise conv + Pointwise conv'''

        def __init__(self, in_planes, out_planes, stride=1):
            super(MobileNet_Block, self).__init__()
            self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes,
                                   bias=False)
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn2 = nn.BatchNorm2d(out_planes)

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            return out

    class MobileNet(nn.Module):
        # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
        cfg = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]

        def __init__(self, num_classes=10):
            super(MobileNet, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(32)
            self.layers = self._make_layers(in_planes=32)
            self.linear = nn.Linear(1024, num_classes)

        def _make_layers(self, in_planes):
            layers = []
            for x in self.cfg:
                out_planes = x if isinstance(x, int) else x[0]
                stride = 1 if isinstance(x, int) else x[1]
                layers.append(MobileNet_Block(in_planes, out_planes, stride))
                in_planes = out_planes
            return nn.Sequential(*layers)

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layers(out)
            out = F.avg_pool2d(out, 2)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out

    class fire(nn.Module):
        def __init__(self, inplanes, squeeze_planes, expand_planes):
            super(fire, self).__init__()
            self.conv1 = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1, stride=1)
            self.bn1 = nn.BatchNorm2d(squeeze_planes)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=1, stride=1)
            self.bn2 = nn.BatchNorm2d(expand_planes)
            self.conv3 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=3, stride=1, padding=1)
            self.bn3 = nn.BatchNorm2d(expand_planes)
            self.relu2 = nn.ReLU(inplace=True)

            # using MSR initilization
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            out1 = self.conv2(x)
            out1 = self.bn2(out1)
            out2 = self.conv3(x)
            out2 = self.bn3(out2)
            out = torch.cat([out1, out2], 1)
            out = self.relu2(out)
            return out

    class SqueezeNet(nn.Module):
        def __init__(self):
            super(SqueezeNet, self).__init__()
            self.conv1 = nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1)  # 32
            self.bn1 = nn.BatchNorm2d(96)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16
            self.fire2 = fire(96, 16, 64)
            self.fire3 = fire(128, 16, 64)
            self.fire4 = fire(128, 32, 128)
            self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 8
            self.fire5 = fire(256, 32, 128)
            self.fire6 = fire(256, 48, 192)
            self.fire7 = fire(384, 48, 192)
            self.fire8 = fire(384, 64, 256)
            self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 4
            self.fire9 = fire(512, 64, 256)
            self.conv2 = nn.Conv2d(512, 10, kernel_size=1, stride=1)
            self.avg_pool = nn.AvgPool2d(kernel_size=4, stride=4)
            self.softmax = nn.LogSoftmax(dim=1)
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool1(x)
            x = self.fire2(x)
            x = self.fire3(x)
            x = self.fire4(x)
            x = self.maxpool2(x)
            x = self.fire5(x)
            x = self.fire6(x)
            x = self.fire7(x)
            x = self.fire8(x)
            x = self.maxpool3(x)
            x = self.fire9(x)
            x = self.conv2(x)
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            x = self.softmax(x)
            return x

    class ShuffleBlock(nn.Module):
        def __init__(self, groups):
            super(ShuffleBlock, self).__init__()
            self.groups = groups

        def forward(self, x):
            '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
            N, C, H, W = x.size()
            g = self.groups
            return x.view(N, g, C // g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)

    class Bottleneck(nn.Module):
        def __init__(self, in_planes, out_planes, stride, groups):
            super(Bottleneck, self).__init__()
            self.stride = stride

            mid_planes = out_planes // 4
            g = 1 if in_planes == 24 else groups
            self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, groups=g, bias=False)
            self.bn1 = nn.BatchNorm2d(mid_planes)
            self.shuffle1 = ShuffleBlock(groups=g)
            self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, groups=mid_planes, bias=False)
            self.bn2 = nn.BatchNorm2d(mid_planes)
            self.conv3 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, groups=groups, bias=False)
            self.bn3 = nn.BatchNorm2d(out_planes)

            self.shortcut = nn.Sequential()
            if stride == 2:
                self.shortcut = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1))

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.shuffle1(out)
            out = F.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            res = self.shortcut(x)
            out = F.relu(torch.cat([out, res], 1)) if self.stride == 2 else F.relu(out + res)
            return out

    class ShuffleNet(nn.Module):
        def __init__(self, cfg):
            super(ShuffleNet, self).__init__()
            out_planes = cfg['out_planes']
            num_blocks = cfg['num_blocks']
            groups = cfg['groups']

            self.conv1 = nn.Conv2d(3, 24, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(24)
            self.in_planes = 24
            self.layer1 = self._make_layer(out_planes[0], num_blocks[0], groups)
            self.layer2 = self._make_layer(out_planes[1], num_blocks[1], groups)
            self.layer3 = self._make_layer(out_planes[2], num_blocks[2], groups)
            self.linear = nn.Linear(out_planes[2], 10)

        def _make_layer(self, out_planes, num_blocks, groups):
            layers = []
            for i in range(num_blocks):
                stride = 2 if i == 0 else 1
                cat_planes = self.in_planes if i == 0 else 0
                layers.append(Bottleneck(self.in_planes, out_planes - cat_planes, stride=stride, groups=groups))
                self.in_planes = out_planes
            return nn.Sequential(*layers)

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out

    def ShuffleNetG2():
        cfg = {
            'out_planes': [200, 400, 800],
            'num_blocks': [4, 8, 4],
            'groups': 2
        }
        return ShuffleNet(cfg)

    def ShuffleNetG3():
        cfg = {
            'out_planes': [240, 480, 960],
            'num_blocks': [4, 8, 4],
            'groups': 3
        }
        return ShuffleNet(cfg)




    resnet_book = {
        '8': resnet8_cifar,
        '14': resnet14_cifar,
        '20': resnet20_cifar,
        '26': resnet26_cifar,
        '32': resnet32_cifar,
        '44': resnet44_cifar,
        '56': resnet56_cifar,
        '110': resnet110_cifar,
    }
    plane_cifar10_book = {
        '2': ['Conv16', 'MaxPool', 'Conv16', 'MaxPool', 'FC10'],
        '4': ['Conv16', 'Conv16', 'MaxPool', 'Conv32', 'Conv32', 'MaxPool', 'FC10'],
        '6': ['Conv16', 'Conv16', 'MaxPool', 'Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool', 'FC10'],
        '8': ['Conv16', 'Conv16', 'MaxPool', 'Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool',
              'Conv128', 'Conv128', 'MaxPool', 'FC64', 'FC10'],
        '10': ['Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool', 'Conv128', 'Conv128', 'MaxPool',
               'Conv256', 'Conv256', 'Conv256', 'Conv256', 'MaxPool', 'FC128', 'FC10'],
    }
    plane_cifar100_book = {
        '2': ['Conv32', 'MaxPool', 'Conv32', 'MaxPool', 'FC100'],
        '4': ['Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool', 'FC100'],
        '6': ['Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool', 'Conv128', 'Conv128', 'FC100'],
        '8': ['Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool', 'Conv128', 'Conv128', 'MaxPool',
              'Conv256', 'Conv256', 'MaxPool', 'FC64', 'FC100'],
        '10': ['Conv32', 'Conv32', 'MaxPool', 'Conv64', 'Conv64', 'MaxPool', 'Conv128', 'Conv128', 'MaxPool',
               'Conv256', 'Conv256', 'Conv256', 'Conv256', 'MaxPool', 'FC512', 'FC100'],
    }
    vgg11_config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    vgg13_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    vgg16_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512,
                    512, 'M']
    vgg19_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
                    512, 512, 512, 512, 'M']
    vgg_cifar10_book = {
        '11': vgg11_cifar,
        '13': vgg13_cifar,
        '16': vgg16_cifar,
        '19': vgg19_cifar,
    }

    def is_resnet(name):
        """
        Simply checks if name represents a resnet, by convention, all resnet names start with 'resnet'
        :param name:
        :return:
        """
        name = name.lower()
        if name.startswith("resnet"):
            return 'resnet'
        elif name.startswith('plane'):
            return 'plane'
        elif name.startswith('alexnet'):
            return 'alexnet'
        elif name.startswith('vgg'):
            return 'vgg'
        elif name.startswith('resnext'):
            return 'resnext'
        elif name.startswith('lenet'):
            return 'lenet'
        elif name.startswith('googlenet'):
            return 'googlenet'
        elif name.startswith('mobilenet'):
            return 'mobilenet'
        elif name.startswith('squeezenet'):
            return 'squeezenet'
        elif name.startswith('shufflenet'):
            return 'shufflenet'


    def create_cnn_model(name, dataset="cifar100", use_cuda=False):
        """
        Create a student for training, given student name and dataset
        :param name: name of the student. e.g., resnet110, resnet32, plane2, plane10, ...
        :param dataset: the dataset which is used to determine last layer's output size. Options are cifar10 and cifar100.
        :return: a pytorch student for neural network
        """
        num_classes = 100 if dataset == 'cifar100' else 10
        model = None
        if is_resnet(name) == 'resnet':
            resnet_size = name[6:]
            resnet_model = resnet_book.get(resnet_size)(num_classes=num_classes)
            model = resnet_model
        elif is_resnet(name) == 'plane':
            plane_size = name[5:]
            model_spec = plane_cifar10_book.get(plane_size) if num_classes == 10 else plane_cifar100_book.get(
                plane_size)
            plane_model = ConvNetMaker(model_spec)
            model = plane_model
        elif is_resnet(name) == 'vgg':
            vgg_size = name[3:]
            vgg_model = vgg_cifar10_book.get(vgg_size)(num_classes=num_classes)
            model = vgg_model
        elif is_resnet(name) == 'alexnet':
            alexnet_model = AlexNet(num_classes)
            model = alexnet_model
        elif is_resnet(name) == 'lenet':
            lenet_model = LeNet()
            model = lenet_model
        elif is_resnet(name) == 'googlenet':
            googlenet_model = GoogLeNet()
            model = googlenet_model
        elif is_resnet(name) == 'mobilenet':
            mobilenet_model = MobileNet()
            model = mobilenet_model
        elif is_resnet(name) == 'squeezenet':
            squeezenet_model = SqueezeNet()
            model = squeezenet_model
        elif is_resnet(name) == 'shufflenet':
            shufflenet_type = name[10:]
            if shufflenet_type == 'g2' or shufflenet_type == 'G2':
                shufflenet_model = ShuffleNetG2()
            else:
                shufflenet_model = ShuffleNetG3()
            model = shufflenet_model



        # copy to cuda if activated
        if use_cuda:
            model = model.cuda()

        return model

    def load_checkpoint(model, checkpoint_path):
        """
        Loads weights from checkpoint
        :param model: a pytorch nn student
        :param str checkpoint_path: address/path of a file
        :return: pytorch nn student with weights loaded from checkpoint
        """
        model_ckp = torch.load(checkpoint_path)
        model.load_state_dict(model_ckp['model_state_dict'])
        return model

    def count_parameters_in_MB(model):
        return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6

    teacher_model = create_cnn_model(teacher_model, train_data, use_cuda=1)
    if teacher_checkpoint:
        teacher_model = load_checkpoint(teacher_model, teacher_checkpoint)
    else:
        if 'resnet' in xargs.teacher_model:
            optimizer = torch.optim.SGD(teacher_model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        elif 'plane' in xargs.teacher_model:
            optimizer = torch.optim.SGD(teacher_model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        elif 'vgg' in xargs.teacher_model:
            optimizer = torch.optim.Adam(teacher_model.parameters(), lr=1e-4)
        elif 'alexnet' in xargs.teacher_model:
            optimizer = torch.optim.Adam(teacher_model.parameters(), lr=0.001)
        elif 'lenet' in xargs.teacher_model:
            optimizer = torch.optim.Adam(teacher_model.parameters(), lr=0.001)
        elif 'googlenet' in xargs.teacher_model:
            optimizer = torch.optim.Adam(teacher_model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        elif 'mobilenet' in xargs.teacher_model:
            optimizer = torch.optim.Adam(teacher_model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        elif 'squeezenet' in xargs.teacher_model:
            optimizer = torch.optim.SGD(teacher_model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        elif 'shufflenet' in xargs.teacher_model:
            optimizer = torch.optim.SGD(teacher_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        teacher_model = train_teacher(search_loader, teacher_model, criterion, optimizer, logger, total_epoch, xargs.teacher_model)


    if TA:
        student_model = create_cnn_model(student, train_data, use_cuda=1)
        if 'resnet' in student:
            optimizer = torch.optim.SGD(student_model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        elif 'plane' in student:
            optimizer = torch.optim.SGD(student_model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        elif 'vgg' in student:
            optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4)
        elif 'alexnet' in student:
            optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
        elif 'lenet' in student:
            optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
        elif 'googlenet' in student:
            optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08,
                                         weight_decay=0)
        elif 'mobilenet' in student:
            optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08,
                                         weight_decay=0)
        elif 'squeezenet' in student:
            optimizer = torch.optim.SGD(student_model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        elif 'shufflenet' in student:
            optimizer = torch.optim.SGD(student_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        if student_checkpoint:
            student_model = load_checkpoint(student_model, student_checkpoint)
            checkpoint = torch.load(student_checkpoint)
            optimizer.load_state_dict(checkpoint['optimizer_dict'])


        # Student_optimizer = torch.optim.SGD(student_model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        # Student_optimizer = torch.optim.Adam(student_model.parameters(), lr=0.025, betas=(0.5, 0.999), weight_decay=1e-4)

        if TA != 'GDAS':
            network = create_cnn_model(TA, train_data, use_cuda=1)
            # w_optimizer = torch.optim.Adam(network .parameters(), lr=0.025, betas=(0.5, 0.999), weight_decay=1e-4)
            if "resnet" in TA:
                w_optimizer = torch.optim.SGD(network.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
            else:
                w_optimizer = torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
            a_optimizer = None

        for epoch in range(start_epoch, total_epoch):
            w_scheduler.update(epoch, 0.0)
            need_time = "Time Left: {:}".format(
                convert_secs2time(epoch_time.val * (total_epoch- epoch), True)
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

            if 'resnet' in TA:
                adjust_learning_rate(w_optimizer, epoch, total_epoch)

            if epoch < total_epoch-epoch_online:
                training_mode = 0
            elif epoch >= total_epoch - epoch_online and epoch < total_epoch:
                training_mode = 1
            else:
                training_mode = 2

            if TA == 'GDAS':
                search_w_loss, search_w_top1, search_w_top5, valid_a_loss, valid_a_top1, valid_a_top5= search_func_modified(
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
            else:
                Student_optimizer = None
                training_mode = 0
                search_w_loss, search_w_top1, search_w_top5, valid_a_loss, valid_a_top1, valid_a_top5, student_loss, student_top1, student_top5 = search_func(search_loader,
                                                                                                                                                                teacher_model,
                                                                                                                                                                network,
                                                                                                                                                                student_model,
                                                                                                                                                                criterion,
                                                                                                                                                                w_scheduler,
                                                                                                                                                                w_optimizer,
                                                                                                                                                                a_optimizer,
                                                                                                                                                                Student_optimizer,
                                                                                                                                                                epoch_str,
                                                                                                                                                                xargs.print_freq,
                                                                                                                                                                logger,
                                                                                                                                                                training_mode
                                                                                                                                                                )

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


            if TA == 'GDAS':

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

        if TA!='GDAS':
            student_model = create_cnn_model(student, train_data, use_cuda=1)
            if 'resnet' in student:
                optimizer = torch.optim.SGD(student_model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
            elif 'plane' in student:
                optimizer = torch.optim.SGD(student_model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
            elif 'vgg' in student:
                optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4)
            elif 'alexnet' in student:
                optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
            elif 'lenet' in student:
                optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
            elif 'googlenet' in student:
                optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08,
                                             weight_decay=0)
            elif 'mobilenet' in student:
                optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08,
                                             weight_decay=0)
            elif 'squeezenet' in student:
                optimizer = torch.optim.SGD(student_model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
            elif 'shufflenet' in student:
                optimizer = torch.optim.SGD(student_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        student_best = -1
        for epoch in range(start_epoch, total_epoch):
            student_loss, student_top1, student_top5 = train_student(search_loader,
                network,
                student_model,
                criterion,
                optimizer,
                epoch_str,
                xargs.print_freq,
                logger,)

            student_accuracy[epoch] = student_top1
            if student_top1 > student_accuracy['best']:
                student_accuracy['best'] = student_top1



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

    else:
        if student:
            student_model = create_cnn_model(student, train_data, use_cuda=1)
            if 'resnet' in student:
                optimizer = torch.optim.SGD(student_model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
            elif 'plane' in student:
                optimizer = torch.optim.SGD(student_model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
            elif 'vgg' in student:
                optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4)
            elif 'alexnet' in student:
                optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
            elif 'lenet' in student:
                optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
            elif 'googlenet' in student:
                optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08,
                                             weight_decay=0)
            elif 'mobilenet' in student:
                optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08,
                                             weight_decay=0)
            elif 'squeezenet' in student:
                optimizer = torch.optim.SGD(student_model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
            elif 'shufflenet' in student:
                optimizer = torch.optim.SGD(student_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

            student_best = -1
            for epoch in range(start_epoch, total_epoch):
                epoch_str = "{:03d}-{:03d}".format(epoch, total_epoch)
                student_loss, student_top1, student_top5 = train_student(search_loader,
                                                                         teacher_model,
                                                                         student_model,
                                                                         criterion,
                                                                         optimizer,
                                                                         epoch_str,
                                                                         xargs.print_freq,
                                                                         logger, )

                student_accuracy[epoch] = student_top1
                if student_top1 > student_accuracy['best']:
                    student_accuracy['best'] = student_top1

            logger.log('----------------')

            logger.log('we used {:} as our Teacher with param size {:}'.format(xargs.teacher_model,
                                                                               count_parameters_in_MB(teacher_model)))
            logger.log('we used {:} as our TA with param size {:}'.format(TA, count_parameters_in_MB(network)))
            logger.log('we used {:} as our Student with param size {:}'.format(xargs.student_model,
                                                                               count_parameters_in_MB(student_model)))

            logger.log('we used {:} online epochs out of total epochs of {:}'.format(xargs.epoch_online, total_epoch))
            logger.log('The best ACC of  : {:.2f}%'.format(TA_accuracy['best']))
            logger.log('The best ACC of Student: {:.2f}%'.format(student_accuracy['best']))
            logger.log('----------------')

            logger.close()

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
        default='../../configs/search-archs/GDAS-NASNet-CIFAR.config',
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
    parser.add_argument("--student_model", default='alexnet', type=str, help="type of student mode")
    parser.add_argument("--teacher_checkpoint", default='Teacher_model_resnet110_90.82%_06-14,15.pth.tar', type=str, help="teacher mode's check point")
    parser.add_argument("--student_checkpoint", default='Teacher_model_alexnet_81.30%_06-17,14.pth.tar', type=str,
                        help="student mode's check point")
    parser.add_argument("--epoch_online", default=250, type=int, help="online training of TA and student")
    args = parser.parse_args()
    if args.rand_seed is None or args.rand_seed < 0:
        args.rand_seed = random.randint(1, 100000)
    # #
    # #
    # teacher_models = ['resnet110', 'resnet56', 'resnet44', 'resnet32', 'resnet26', 'resnet20', 'resnet14', 'resnet8', 'plane10', 'plane8', 'plane6','plane4','plane2' ]
    # teacher_models = ['vgg16']
    # for one in teacher_models:
    #     args.teacher_model = one

    # TA_models = ['plane4', 'plane6', 'resnet26', 'resnet20']
    # for one in TA_models:
    #     args.TA = one
    main(args)
