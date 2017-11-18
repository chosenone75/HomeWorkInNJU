#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import re
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import cPickle

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string(
    "data_file",
    "data",
    "Data source for the training and validation data.")

# Model Hyperparameters
tf.flags.DEFINE_integer(
    "embedding_dim",
    300,
    "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string(
    "filter_sizes",
    "1,2,3",
    "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer(
    "num_filters",
    100,
    "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float(
    "dropout_keep_prob",
    0.5,
    "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float(
    "l2_reg_lambda",
    3,
    "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 50, "Batch Size (default: 64)")
tf.flags.DEFINE_integer(
    "num_epochs",
    200,
    "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer(
    "evaluate_every",
    160,
    "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer(
    "num_checkpoints",
    5,
    "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer(
    "early_stopping_steps",
    10,
    "Number of steps until early stopping (default: 5)")
tf.flags.DEFINE_float(
    "learning_rate",
    0.001,
    "The start learning rate (default: 0.001)")
tf.flags.DEFINE_float(
    "decay_step",
    500,
    "Decay step for Rmsprop (default: 500)")
tf.flags.DEFINE_float(
    "decay_rate",
    0.98,
    "Decay rate for Rmsprop (default: 0.98)")
# Misc Parameters
tf.flags.DEFINE_boolean(
    "allow_soft_placement",
    True,
    "Allow device soft device placement")
tf.flags.DEFINE_boolean(
    "log_device_placement",
    False,
    "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# misc helper function
def tokenizer(iterator):
    """Tokenizer generator.
    Args:
      iterator: Input iterator with strings.
    Yields:
      array of tokens per each value in the input.
    """
    for value in iterator:
        yield re.split('\s+', str(value))


def load_pre_trained_vec(filepath='w2v.vec', vocab=None):
    if os.path.exists(filepath):
        print 'here'
        W, _ = cPickle.load(open(filepath, 'rb'))
    else:
        # load word2vec and save the necessary word vector
        w2v = data_helpers.load_bin_vec(
            'GoogleNews-vectors-negative300.bin', vocab)
        print(len(w2v))
        data_helpers.add_unknown_words(w2v, vocab)
        W = data_helpers.get_WordVector(w2v, vocab)
        cPickle.dump([W, vocab], open(filepath, 'wb'))
    return W

# Data Preparation
# ==================================================


# Load data
print("Loading data...")
train_x_text, train_y = data_helpers.load_files(
    FLAGS.data_file, subset='train')
dev_x_text, dev_y = data_helpers.load_files(FLAGS.data_file, subset='dev')
test_x = data_helpers.load_test_files(FLAGS.data_file, subset='test')

all_text = np.concatenate((train_x_text, dev_x_text, test_x), axis=0)
print len(all_text)

# Build vocabulary
max_document_length = max([len(text.split(" ")) for text in train_x_text])
print "max document length:", max_document_length

vocab_processor = learn.preprocessing.VocabularyProcessor(
    max_document_length, tokenizer_fn=tokenizer)

train_x_id = np.array(list(vocab_processor.fit_transform(train_x_text)))
dev_x_id = np.array(list(vocab_processor.transform(dev_x_text)))
# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(train_y)))
x_shuffled = train_x_id[shuffle_indices]
y_shuffled = train_y[shuffle_indices]

x_train, x_dev = train_x_id, dev_x_id
y_train, y_dev = train_y, dev_y

print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print(
    "Train/Dev/Test split: {:d}/{:d}/{:d}".format(len(y_train), len(y_dev), len(test_x)))


W = load_pre_trained_vec(vocab=vocab_processor.vocabulary_._reverse_mapping)

# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            pretrained_w2v=W,
            use_pretrained=True,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)

        learning_rate = tf.train.exponential_decay(
            FLAGS.learning_rate, global_step, FLAGS.decay_step, FLAGS.decay_rate)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(
            grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram(
                    "{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar(
                    "{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(
            os.path.join(
                os.path.curdir,
                "runs",
                timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge(
            [loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(
            train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already
        # exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(
            tf.global_variables(),
            max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print(
                "{}: step {}, loss {:g}, acc {:g}".format(
                    time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print(
                "{}: step {}, loss {:g}, acc {:g}".format(
                    time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)
            return accuracy
        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

        early_stopping_steps = FLAGS.early_stopping_steps

        best_dev_acc = 0
        stopping_steps = 0

        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                cur_acc = dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
                # early stopping: cache the best global step
                if cur_acc >= best_dev_acc:
                    best_dev_acc = cur_acc
                    stopping_steps = 0
                    # save current model
                    path = saver.save(
                        sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
                # else:
                #     stopping_steps += 1
                #     if stopping_steps > early_stopping_steps:
                #         print 'early stopping with best dev accuracy: {:g}'.format(best_dev_acc)
                #         break
