#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : main.py
#   Author      : YunYang1994
#   Created date: 2019-10-25 15:18:10
#   Description :
#
#================================================================

import tensorflow as tf
from resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

from scripts.utils import write_csv
import timeit

lr = 0.0001
batch_size = 32
EPOCHS = 5

# Build your model here
model = ResNet18()
optimizer = tf.keras.optimizers.Adam(lr)

# Load and prepare the cifar10 dataset.
cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = tf.reshape(y_train, (-1,)), tf.reshape(y_test, (-1,))

SAMPLE = 1000

# Use tf.data to batch and shuffle the dataset
train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).take(SAMPLE).shuffle(100).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices(
        (x_test, y_test)).take(SAMPLE).batch(batch_size)

# Choose an optimizer and loss function for training
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

# Select metrics to measure the loss and the accuracy of the model
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

start_time = timeit.default_timer()
skipped_time = 0

total_loss = 0
loss_count = 0

total_accuracy = 0
accuracy_count = 0

# Use tf.GradientTape to train the model.
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        # print("=> label shape: ", labels.shape, "pred shape", predictions.shape)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)

for epoch in range(EPOCHS):
    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    print_time = timeit.default_timer()
    template = '=> Epoch {}, Loss: {:.4}, Accuracy: {:.2%}, Test Loss: {:.4}, Test Accuracy: {:.2%}'
    print(template.format(epoch+1,
                          train_loss.result(),
                          train_accuracy.result(),
                          test_loss.result(),
                          test_accuracy.result()))
    total_loss += train_loss.result()
    loss_count += 1
    total_accuracy += train_accuracy.result()
    accuracy_count += 1
    skipped_time += timeit.default_timer() - print_time
    # Reset the metrics for the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

time = timeit.default_timer() - start_time - skipped_time
avg_loss = float(total_loss) / float(loss_count)
avg_accuracy = float(total_accuracy)/ float(accuracy_count)

write_csv(__file__, epochs=EPOCHS, loss=float(avg_loss), accuracy=float(avg_accuracy), time=time)
