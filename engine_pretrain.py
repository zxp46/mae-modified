# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    sim_loss = -0.96
    for data_iter_step, ([samples, smaller_samples, sep], _) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # pzx modified #
        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            total_loss, sim_loss, loss, _, _, = model(samples, smaller_samples, mask_ratio=args.mask_ratio,
                                                      device=device, double_loss=True, epoch=epoch)
        total_loss_value, sim_loss_value, loss_value = total_loss.item(), sim_loss.item(), loss.item()
        print("Losses(total, new, mae) is {}, stopping training".format((total_loss_value, sim_loss_value, loss_value)))
        # end modify #
        if not math.isfinite(total_loss_value):
            print("Loss is {}, stopping training".format(total_loss_value))
            sys.exit(1)

        total_loss, loss, sim_loss = total_loss / accum_iter, loss / accum_iter, sim_loss / accum_iter
        loss_scaler(total_loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=total_loss_value)
        metric_logger.update(mae_loss=loss_value)
        metric_logger.update(style_loss=sim_loss_value)

        lr = optimizer.param_groups[0]["lr"]

        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(total_loss_value)
        loss_decoder_reduce = misc.all_reduce_mean(loss_value)
        loss_encoder_reduce = misc.all_reduce_mean(sim_loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('contrastive_loss', loss_encoder_reduce, epoch_1000x)
            log_writer.add_scalar('decoder_loss', loss_decoder_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
