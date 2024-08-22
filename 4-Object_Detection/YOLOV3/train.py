#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : train.py
#   Author      : YunYang1994
#   Created date: 2019-07-18 09:18:54
#   Description :
#
#================================================================

import os
import time
import shutil
import numpy as np
import tensorflow as tf
import core.utils as utils
from tqdm import tqdm
from core.dataset import Dataset
from core.yolov3 import YOLOv3, decode, compute_loss
from core.config import cfg

from scripts.utils import write_csv
import timeit

trainset = Dataset('train')
logdir = "./data/log"
steps_per_epoch = len(trainset)
global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch
total_steps = cfg.TRAIN.EPOCHS * steps_per_epoch

start_time = timeit.default_timer()
skipped_time = 0

input_tensor = tf.keras.layers.Input([416, 416, 3])
conv_tensors = YOLOv3(input_tensor)

output_tensors = []
for i, conv_tensor in enumerate(conv_tensors):
    pred_tensor = decode(conv_tensor, i)
    output_tensors.append(conv_tensor)
    output_tensors.append(pred_tensor)

model = tf.keras.Model(input_tensor, output_tensors)
optimizer = tf.keras.optimizers.Adam()
if os.path.exists(logdir): shutil.rmtree(logdir)
writer = tf.summary.create_file_writer(logdir)

total_loss = 0
loss_count = 0

@tf.function
def train_step(image_data, target):
    global skipped_time, loss_count
    with tf.GradientTape() as tape:
        pred_result = model(image_data, training=True)
        giou_loss=conf_loss=prob_loss=0

        # optimizing process
        for i in range(3):
            conv, pred = pred_result[i*2], pred_result[i*2+1]
            loss_items = compute_loss(pred, conv, *target[i], i)
            giou_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]

        total_loss = giou_loss + conf_loss + prob_loss

        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        print_time = timeit.default_timer()
        tf.print("=> STEP " + tf.strings.as_string(global_steps) +
                 "\tlr: " + tf.strings.as_string(optimizer.lr, 6) +
                 "\tgiou_loss: " + tf.strings.as_string(giou_loss, 2) +
                 "\tconf_loss: " + tf.strings.as_string(conf_loss, 2) +
                 "\tprob_loss: " + tf.strings.as_string(prob_loss, 2) +
                 "\ttotal_loss: " + tf.strings.as_string(total_loss, 2))
        skipped_time += timeit.default_timer() - print_time
        # update learning rate
        global_steps.assign_add(1)
        if global_steps < warmup_steps:
            lr = global_steps / warmup_steps *cfg.TRAIN.LR_INIT
        else:
            lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
                (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
            )
        optimizer.lr.assign(tf.cast(lr, tf.float32))

        # writing summary data
        print_time = timeit.default_timer()
        with writer.as_default():
            tf.summary.scalar("lr", optimizer.lr, step=global_steps)
            tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
            tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
            tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
            tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
        writer.flush()
        total_loss += total_loss
        loss_count += 1
        skipped_time += timeit.default_timer() - print_time

IMAGES = 10
image_count = 0

for epoch in range(cfg.TRAIN.EPOCHS):
    for image_data, target in trainset:
        if image_count < IMAGES:
            train_step(image_data, target)
            image_count += 1
        else:
            break
    print_time = timeit.default_timer()
    model.save_weights("./yolov3")
    skipped_time += timeit.default_timer() - print_time

time = timeit.default_timer() - start_time - skipped_time
avg_loss = float(total_loss) / float(loss_count)

write_csv(__file__, epochs=cfg.TRAIN.EPOCHS, loss=float(avg_loss), time=time)
