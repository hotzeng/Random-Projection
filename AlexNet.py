#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0

#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import pandas as pd

tf.logging.set_verbosity(tf.logging.INFO)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 32, 32, 3])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=96,
      kernel_size=[11, 11],
      strides=(4,4),
      padding="same",
      activation=tf.nn.relu)



  # lrn1
  radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
  lrn1 = tf.nn.local_response_normalization(conv1,
                                            depth_radius=radius,
                                            alpha=alpha,
                                            beta=beta,
                                            bias=bias)


  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=lrn1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=256,
      kernel_size=[5, 5],
      strides=(1,1),
      padding="same",
      activation=tf.nn.relu)


  # lrn2
  radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
  lrn2 = tf.nn.local_response_normalization(conv2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)



  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=lrn2, pool_size=[2, 2], strides=2)

  # conv3
  conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=384,
      kernel_size=[3, 3],
      strides=(1,1),
      padding="same",
      activation=tf.nn.relu)

  # conv4
  conv4 = tf.layers.conv2d(
      inputs=conv3,
      filters=384,
      kernel_size=[3, 3],
      strides=(1,1),
      padding="same",
      activation=tf.nn.relu)

  # conv5
  conv5 = tf.layers.conv2d(
      inputs=conv4,
      filters=256,
      kernel_size=[3, 3],
      strides=(1,1),
      padding="same",
      activation=tf.nn.relu)

  # maxpool5
  k_h = 2; k_w = 2; s_h = 2; s_w = 2; padding = 'VALID'
  pool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  pool5_flat = tf.reshape(pool5, [-1, int(np.prod(pool5.get_shape()[1:]))])

  # fc6
  # Densely connected layer with 4096 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  fc6 = tf.layers.dense(inputs=pool5_flat, units=4096, activation=tf.nn.relu)

  # fc7
  fc7 = tf.layers.dense(inputs=fc6, units=4096, activation=tf.nn.relu)

  # fc8
  fc8 = tf.layers.dense(inputs=fc7, units=10)


  # Add dropout operation; 0.6 probability that element will be kept
  #dropout = tf.layers.dropout(
  #    inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  # logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=fc8, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(fc8, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=fc8)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    #optimizer = tf.train.AdamOptimizer
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  # Load training and eval data

  #mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  #train_data = mnist.train.images # Returns np.array
  #train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  #eval_data = mnist.test.images # Returns np.array
  #eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

  cifar_data1 = unpickle('../Data/Cifar10/cifar-10-batches-py/data_batch_1')
  cifar_data2 = unpickle('../Data/Cifar10/cifar-10-batches-py/data_batch_2')
  cifar_data3 = unpickle('../Data/Cifar10/cifar-10-batches-py/data_batch_3')
  cifar_data4 = unpickle('../Data/Cifar10/cifar-10-batches-py/data_batch_4')
  cifar_data5 = unpickle('../Data/Cifar10/cifar-10-batches-py/data_batch_5')
  test_data = unpickle('../Data/Cifar10/cifar-10-batches-py/test_batch')

  train_data = np.vstack((cifar_data1[b'data'], cifar_data2[b'data'], cifar_data3[b'data'], cifar_data4[b'data'], cifar_data5[b'data']))
  train_data = np.transpose( np.reshape(train_data, (-1,3,32,32 )), (0,2,3,1) )
  # normalize and subtract mean value
  train_data = train_data / 255
  train_data = train_data - np.mean(train_data)
  train_data = train_data.astype('float32')

  train_labels = np.hstack((np.asarray(cifar_data1[b'labels']), np.asarray(cifar_data2[b'labels']), np.asarray(cifar_data3[b'labels']), np.asarray(cifar_data4[b'labels']), np.asarray(cifar_data5[b'labels'])))
  eval_data = test_data[b'data']
  eval_data = np.transpose( np.reshape(train_data, (-1,3,32,32 )), (0,2,3,1) )
  eval_data = eval_data / 255
  eval_data = eval_data - np.mean(eval_data)
  eval_data = eval_data.astype('float32')
  eval_labels = test_data[b'labels']
  

  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  #logging_hook = tf.train.LoggingTensorHook(every_n_iter=50)
      #tensors=tensors_to_log, every_n_iter=50)


  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=10,
      num_epochs=None,
      shuffle=True)
  mnist_classifier.train(
      input_fn=train_input_fn,
      steps=200000)
      #hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()
