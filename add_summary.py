import numpy as np
import tensorflow as tf
from datetime import datetime

root_logdir = 'tf_logs/bilstm'
now_time = datetime.utcnow().strftime("%Y%m%d%H%M%S")
logdir = root_logdir  + '_' + now_time


def make_summary(tag, value):
  return tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])


graph = tf.Graph()
with graph.as_default():
    with tf.Session() as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        train_writer = tf.summary.FileWriter(logdir, sess.graph)

        x = tf.constant(0)
        for i in range(10):
            y = x + 2
            x = x + 1
            x_out,y_out = sess.run([x,y])
            print(x_out,y_out)
            train_writer.add_summary(make_summary(tag="train/y", value=y_out), i)
            train_writer.add_summary(make_summary(tag="train/x", value=x_out), i)
            train_writer.add_summary(make_summary(tag="train/i", value=i), i)


