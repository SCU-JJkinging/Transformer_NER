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
from metrics.ner_metrics import SeqEntityScore
from train_config.softmax_config import Config
from utils.adversarial import FGM, PGD
import numpy as np


def train(dataloader, model, optimizer, scheduler):
    model.train()
    config = Config()
    device = model.device
    scaler = None
    if config.use_fp16:
        scaler = GradScaler()
    k = 3
    adversarial = None
    if config.do_adv == 'FGM':
        adversarial = FGM(model, emb_name='word_embeddings', epsilon=1.0)
    elif config.do_adv == 'PGD':
        adversarial = PGD(model, emb_name='word_embeddings', epsilon=1.0, alpha=0.3)

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
                loss, logit = model(input_ids, attention_mask, label)
        else:
            loss, logit = model(input_ids, attention_mask, label)
        if config.use_fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if config.do_adv == 'FGM':
            adversarial.attack()
            if config.use_fp16:
                with autocast():
                    loss_adv, logit_adv = model(input_ids, attention_mask, label)
            else:
                loss_adv, logit_adv = model(input_ids, attention_mask, label)
            if config.use_fp16:
                scaler.scale(loss_adv).backward()
            else:
                loss_adv.backward()
            adversarial.restore()
        elif config.do_adv == 'PGD':
            adversarial.backup_grad()
            for t in range(k):
                adversarial.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.data
                if t != k - 1:
                    optimizer.zero_grad()
                else:
                    adversarial.restore_grad()
                if config.use_fp16:
                    with autocast():
                        loss_adv, logit_adv = model(input_ids, attention_mask, label)
                else:
                    loss_adv, logit_adv = model(input_ids, attention_mask, label)

                if config.use_fp16:
                    scaler.scale(loss_adv).backward()
                else:
                    loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            adversarial.restore()  # 恢复embedding参数

        if config.use_fp16:
            scaler.unscale_(optimizer)

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

        description = "Avg. batch proc. time: {:.6f}s, loss: {:.6f}" \
            .format(batch_time_avg / (batch_index + 1),
                    running_loss / (batch_index + 1))
        tqdm_batch_iterator.set_description(description)

    epoch_time = time.time() - epoch_start_time
    epoch_loss = running_loss / len(dataloader)

    return epoch_time, epoch_loss


def evaluate(dataloader, model, id2label):
    config = Config()
    metric = SeqEntityScore(id2label, markup=config.markup)
    device = model.device
    size = len(dataloader.dataset)
    model.eval()
    # ema.apply_shadow()
    val_loss = 0
    with torch.no_grad():
        for data in dataloader:
            input_ids = data[0].to(device)
            attention_mask = data[1].to(device)
            label = data[2].to(device)
            input_lens = attention_mask.sum(dim=-1)
            loss, logits = model(input_ids, attention_mask, label)
            preds = np.argmax(logits.cpu().numpy(), axis=2).tolist()
            val_loss += loss.item()
            out_label_ids = label.cpu().numpy().tolist()
            input_lens = input_lens.cpu().numpy().tolist()
            for i, label in enumerate(out_label_ids):
                label_tmp = []
                pred_tmp = []
                for j, m in enumerate(label):
                    if j == 0:  # 排除'[CLS]'
                        continue
                    elif j == input_lens[i] - 1:
                        metric.update(pred_paths=[pred_tmp], label_paths=[label_tmp])
                        break
                    else:
                        label_tmp.append(id2label[out_label_ids[i][j]])
                        pred_tmp.append(id2label[preds[i][j]])

    val_loss /= size
    eval_info, entity_info = metric.result()
    results = {f'{key}': value for key, value in eval_info.items()}
    results['loss'] = val_loss
    print("***** Eval results %s *****")
    info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
    print(info)
    print("***** Entity results %s *****")
    for key in sorted(entity_info.keys()):
        print("******* %s results ********" % key)
        info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[key].items()])
        print(info)

    # ema.restore()
    return results  # {'acc': precision, 'recall': recall, 'f1': f1, 'loss': loss}
