#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/17 13:36
# @Author  : JJkinging
# @File    : crf_config.py
class Config(object):
    # 配置类

    def __init__(self):
        self.pretrained_model_path = r'E:\python_project\pretrained_model\hfl_chinese_roberta_wwm_ext'
        self.train_file_data = r'E:\python_project\Transformer_NER\dataset\CMeEE\CMeEE_train.json'
        self.val_file_data = r'E:\python_project\Transformer_NER\dataset\CMeEE\CMeEE_dev.json'
        self.test_file_data = r'E:\python_project\Transformer_NER\dataset\CMeEE\CMeEE_test.json'
        self.target_dir = '../result/softmax/model_checkpoint'
        self.out_file = 'CMeEE_test.json'
        self.lr = 2e-5
        self.max_len = 256
        self.batch_size = 8
        self.epochs = 10
        self.head_size = 64
        self.hidden_size = 768
        self.inference_maxlen = 256
        self.warmup_prop = 0.0
        self.clip_norm = 0.25
        self.dim_in = 768
        self.dim_hid = 768
        self.do_adv = None
        self.use_fp16 = False
        self.checkpoint = None  # '../result/softmax/model_checkpoint/model_2.pth.tar'
        self.patience = 10
        self.markup = 'bios'
        self.loss_type = 'ce'

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        return '\n'.join(['%s:%s' % item for item in self.__dict__.items()])


if __name__ == '__main__':
    con = Config()
    con.update(gpu=8)
    print(con)
