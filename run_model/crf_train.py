#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/17 12:47
# @Author  : JJkinging
# @File    : crf_train.py
# import sys
# sys.path.append('/home/seatrend/jinxiang/Efficient_GlobalPointer')
import os
import torch
from run_model.crf_util import evaluate, train
from train_config.crf_config import Config
from transformers import AdamW, get_linear_schedule_with_warmup
from data_processing.data_process import yeild_data
from model.model import BertCrfForNer
from utils.tools import EMA
from utils.tools import setup_seed


def main():
    setup_seed(1234)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print("Using {} device".format(device))

    config = Config()

    label_list = ['O', 'B-pro', 'I-pro', 'B-dis', 'I-dis', 'B-sym', 'I-sym', 'B-ite',
                  'I-ite', 'B-bod', 'I-bod', 'B-dru', 'I-dru', 'B-mic', 'I-mic',
                  'B-equ', 'I-equ', 'B-dep', 'I-dep', 'S-pro', 'S-dis', 'S-sym',
                  'S-ite', 'S-bod', 'S-dru', 'S-mic', 'S-equ', 'S-dep']

    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}

    train_dataloader = yeild_data(config.train_file_data, label2id, model_type='crf', is_train=True, ddp=False)
    val_dataloader = yeild_data(config.val_file_data, label2id, model_type='crf', is_train=False, ddp=False)

    model = BertCrfForNer(config.pretrained_model_path,
                          len(label_list),
                          config.hidden_size,
                          device).to(device)

    # ema = EMA(model, 0.999)
    # ema.register()

    total_steps = len(train_dataloader) * config.epochs
    no_decay = ["bias", "LayerNorm.weight"]
    bert_param_optimizer = list(model.bert.named_parameters())
    crf_param_optimizer = list(model.crf.named_parameters())
    linear_param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01, 'lr': config.lr},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': config.lr},

        {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01, 'lr': config.crf_lr},
        {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': config.crf_lr},

        {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01, 'lr': config.crf_lr},
        {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': config.crf_lr}
    ]

    optimizer = AdamW(params=optimizer_grouped_parameters, lr=config.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(config.warmup_prop * total_steps),
                                                num_training_steps=total_steps)

    best_score = 0
    start_epoch = 1
    # Data for loss curves plot.
    epochs_list = []
    train_losses = []
    valid_losses = []
    # Continuing training from a checkpoint if one was given as argument.
    if config.checkpoint:
        checkpoint = torch.load(config.checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]

        print("\t* Training will continue on existing model from epoch {}..."
              .format(start_epoch))

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epochs_list = checkpoint["epochs_list"]
        train_losses = checkpoint["train_loss"]
        valid_losses = checkpoint["valid_loss"]

    # Compute loss and accuracy before starting (or resuming) training.
    # results = evaluate(val_dataloader, model, id2label)
    # print("-> val_loss = {:.6f} val_f1: {:.6f}".format(results['loss'], results['f1']))

    # -------------------- Training epochs ------------------- #
    print("\n",
          20 * "=",
          "Training Model model on device: {}".format(device),
          20 * "=")
    patience_counter = 0
    for epoch in range(start_epoch, config.epochs + 1):
        epochs_list.append(epoch)
        print("* Training epoch {}:".format(epoch))

        train_time, train_loss = train(train_dataloader, model, optimizer, scheduler)
        train_losses.append(train_loss)
        print(
            "-> Training time: {:.4f}s train_loss = {:.6f}".format(train_time, train_loss))
        with open('../result/crf/metric.txt', 'a', encoding='utf-8') as fp:
            fp.write(
                'Epoch:' + str(epoch) + '\t' + 'train_loss:' + str(round(train_loss, 6)) + '\t')

        results = evaluate(val_dataloader, model, id2label)
        print("-> val_loss = {:.6f} val_f1: {:.6f}".format(results['loss'], results['f1']))

        with open('../result/crf/metric.txt', 'a', encoding='utf-8') as fp:
            fp.write('val_loss:' + str(round(results['loss'], 6)) + '\t' + 'val_f1:' +
                     str(round(results['f1'], 6)) + '\n')

        valid_losses.append(valid_losses)

        # Early stopping on validation accuracy.
        if results['f1'] < best_score:
            patience_counter += 1
        else:
            best_score = results['f1']
            patience_counter = 0
            torch.save({"epoch": epoch,
                        "model": model.state_dict(),
                        "best_score": best_score,
                        "epochs_list": epochs_list,
                        "train_losses": train_losses,
                        "valid_losses": valid_losses},
                       os.path.join(config.target_dir, "model_best.pth.tar"))

        # Save the model at each epoch.
        torch.save({"epoch": epoch,
                    "model": model.state_dict(),
                    "best_score": best_score,
                    "optimizer": optimizer.state_dict(),
                    "epochs_list": epochs_list,
                    "train_losses": train_losses,
                    "valid_losses": valid_losses},
                   os.path.join(config.target_dir, "model_{}.pth.tar".format(epoch)))

        if patience_counter >= config.patience:
            print("-> Early stopping: patience limit reached, stopping...")
            break


if __name__ == '__main__':
    main()
