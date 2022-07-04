#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/16 17:16
# @Author  : JJkinging
# @File    : train_eval.py
import time

import torch
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_

from metrics.global_metrics import global_pointer_f1_score
from train_config.globalpointer_config import Config
from utils.adversarial import FGM, PGD


def train(dataloader, model, loss_func, optimizer, scheduler):
    model.train()
    config = Config()
    device = model.device
    scaler = None
    if config.use_fp16:
        scaler = GradScaler()
    k = 3
    size = len(dataloader.dataset)
    total_n, total_d = 0.0, 0.0
    if config.do_adv == 'FGM':
        adversial = FGM(model, emb_name='word_embeddings', epsilon=1.0)
    elif config.do_adv == 'PGD':
        adversial = PGD(model, emb_name='word_embeddings', epsilon=1.0, alpha=0.3)

    epoch_start_time = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0

    tqdm_batch_iterator = tqdm(dataloader)
    for batch_index, batch in enumerate(tqdm_batch_iterator):
        batch_start_time = time.time()
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        label = batch[2].to(device)
        if config.use_fp16:
            with autocast():
                pred = model(input_ids, attention_mask)
                loss = loss_func(label, pred) / config.accumulate_step
        else:
            pred = model(input_ids, attention_mask)
            loss = loss_func(label, pred) / config.accumulate_step
        num, den = global_pointer_f1_score(label, pred)
        total_n += num
        total_d += den
        if config.use_fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if config.do_adv == 'FGM':
            adversial.attack()
            if config.use_fp16:
                with autocast():
                    pred_adv = model(input_ids, attention_mask)
                    loss_adv = loss_func(label, pred_adv) / config.accumulate_step
            else:
                pred_adv = model(input_ids, attention_mask)
                loss_adv = loss_func(label, pred_adv) / config.accumulate_step
            if config.use_fp16:
                scaler.scale(loss_adv).backward()
            else:
                loss_adv.backward()
            adversial.restore()
        elif config.do_adv == 'PGD':
            adversial.backup_grad()
            for t in range(k):
                adversial.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.data
                if t != k - 1:
                    optimizer.zero_grad()
                else:
                    adversial.restore_grad()
                if config.use_fp16:
                    with autocast():
                        pred_adv = model(input_ids, attention_mask)
                        loss_adv = loss_func(label, pred_adv) / config.accumulate_step
                else:
                    pred_adv = model(input_ids, attention_mask)
                    loss_adv = loss_func(label, pred_adv) / config.accumulate_step

                if config.use_fp16:
                    scaler.scale(loss_adv).backward()
                else:
                    loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            adversial.restore()  # 恢复embedding参数

        if config.use_fp16:
            scaler.unscale_(optimizer)
        if (batch_index + 1) % config.accumulate_step == 0:
            clip_grad_norm_(model.parameters(), max_norm=config.clip_norm)
            if config.use_fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            # ema.update()
            scheduler.step()
            optimizer.zero_grad()

        batch_time_avg += time.time() - batch_start_time
        running_loss += loss.item()

        description = "Avg. batch proc. time: {:.6f}s, loss: {:.6f}, train_f1: {:.6f}" \
            .format(batch_time_avg / (batch_index + 1),
                    running_loss / (batch_index + 1),
                    2*total_n/total_d)
        tqdm_batch_iterator.set_description(description)

    epoch_time = time.time() - epoch_start_time
    epoch_loss = running_loss / len(dataloader)
    train_f1 = 2 * total_n / total_d

    return epoch_time, epoch_loss, train_f1


def evaluate(dataloader, loss_func, model):
    device = model.device
    size = len(dataloader.dataset)
    model.eval()
    # ema.apply_shadow()
    val_loss = 0
    epoch_start = time.time()
    total_n, total_d = 0.0, 0.0
    with torch.no_grad():
        for data in dataloader:
            input_ids = data[0].to(device)
            attention_mask = data[1].to(device)
            label = data[2].to(device)
            pred = model(input_ids, attention_mask)
            val_loss += loss_func(label, pred).item()
            num, den = global_pointer_f1_score(label, pred)
            total_n += num
            total_d += den
    val_loss /= size
    val_f1 = 2 * total_n / total_d
    val_time = time.time() - epoch_start
    # ema.restore()
    return val_time, val_loss, val_f1
