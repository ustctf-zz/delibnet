# coding=utf-8

""" calculate the per sentence loss
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import numpy as np
import tensorflow as tf
import operator

from tensor2tensor.utils import trainer_utils as utils
from tensor2tensor.utils import usr_dir
from tensor2tensor.utils import decoding
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import metrics

__author__ = "Chang Chen; revised by Yingce"


flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("t2t_usr_dir", None, "Path to a Python module that will be imported.")

flags.DEFINE_string("srcFile", None, "Path to a Python module that will be imported.")
flags.DEFINE_string("firstPFile", None, "Path to a Python module that will be imported.")
flags.DEFINE_string("tgtFile", None, "Path to a Python module that will be imported.")
flags.DEFINE_string("scoreFile", None, "Path to a Python module that will be imported.")

flags.DEFINE_integer("dupsrc", 1, "Path to a Python module that will be imported.")
flags.DEFINE_integer("eval_batch", 1, "Path to a Python module that will be imported.")
flags.DEFINE_string("schedule", "train_and_evaluate",
                    "Method of tf.contrib.learn.Experiment to run.")
flags.DEFINE_string("output_dir", "", "Base output directory for run.")


def get_sorted_inputs(*args):
    tf.logging.info("Getting sorted inputs")
    # read file and sort inputs according them according to input length.
    text_files = args 
    m_args = len(args)
    # if FLAGS.decode_shards > 1:
    #    text_files = [ff + ("%.2d" % FLAGS.worker_id) for ff in args]
        
    text_lines = [None for _ in range(m_args)]
    
    for fid, fname in enumerate(text_files):
        text_lines[fid] = [x.strip() for x in tf.gfile.Open(fname)]
   
    if FLAGS.dupsrc > 1:
        text_lines[0] = [x for x in text_lines[0] for _ in range(FLAGS.dupsrc)]
        text_lines[1] = [x for x in text_lines[1] for _ in range(FLAGS.dupsrc)]
    
    input_lens = [(i, len(line.strip().split())) for i, line in enumerate(text_lines[0])]
    sorted_input_lens = sorted(input_lens, key=operator.itemgetter(1))
      
    sorted_keys = {}
    sorted_text_lines = [[] for _ in range(m_args)]
   
    for i, (index, _) in enumerate(sorted_input_lens):
        for jj in range(m_args):
            sorted_text_lines[jj].append(text_lines[jj][index])
        sorted_keys[index] = i
    return sorted_text_lines, sorted_keys


def input_iter(problem_id, num_decode_batches, 
               sorted_inputs, sorted_firstP, sorted_targets, 
               vocabulary, vocabulary_trg):
    tf.logging.info(" batch %d" % num_decode_batches)
    # First reverse all the input sentences so that if you're going to get OOMs,
    # you'll see it in the first batch
    sorted_inputs.reverse()
    sorted_firstP.reverse()
    sorted_targets.reverse()
    for b in range(num_decode_batches):
        tf.logging.info("Decoding batch %d" % b)
        batch_length, batch_firstP_length, batch_target_len = 0, 0, 0
        batch_inputs, batch_firstP, batch_targets = [], [], []
        curr_batch_inputs = sorted_inputs[b*FLAGS.eval_batch: (b + 1)*FLAGS.eval_batch]
        curr_batch_firstP = sorted_firstP[b*FLAGS.eval_batch: (b + 1)*FLAGS.eval_batch]
        curr_batch_targets = sorted_targets[b*FLAGS.eval_batch: (b + 1)*FLAGS.eval_batch]

        for (inputs, firstP, target) in zip(curr_batch_inputs, curr_batch_firstP, curr_batch_targets):
            input_ids = vocabulary.encode(inputs)
            firstP_ids = vocabulary_trg.encode(firstP)
            target_ids = vocabulary_trg.encode(target)
            
            input_ids.append(text_encoder.EOS_ID)
            batch_inputs.append(input_ids)
            
            firstP_ids.append(text_encoder.EOS_ID)
            batch_firstP.append(firstP_ids)
            
            target_ids.append(text_encoder.EOS_ID)
            batch_targets.append(target_ids)
            
            batch_length = max(len(input_ids), batch_length)
            batch_firstP_length = max(len(firstP_ids), batch_firstP_length)
            batch_target_len = max(len(target_ids), batch_target_len)
            
        final_batch_inputs = []
        final_batch_firstP = []
        final_batch_targets = []
        
        for (input_ids, firstP_ids, target_ids) in zip(batch_inputs, batch_firstP, batch_targets):
            assert len(input_ids) <= batch_length
            assert len(firstP_ids) <= batch_firstP_length
            assert len(target_ids) <= batch_target_len
            final_batch_inputs.append(input_ids + [0] * (batch_length - len(input_ids)))
            final_batch_firstP.append(firstP_ids + [0] * (batch_firstP_length - len(firstP_ids)))
            final_batch_targets.append(target_ids + [0] * (batch_target_len - len(target_ids)))
        yield {
            "inputs": np.array(final_batch_inputs),
            "firstP": np.array(final_batch_firstP),
            "targets": np.array(final_batch_targets),
            "problem_choice": np.array(problem_id)
        }


def main(_):
    # Set the logging level.
    tf.logging.set_verbosity(tf.logging.INFO)

    # Import module at usr_dir, if provided.
    if FLAGS.t2t_usr_dir is not None:
        usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)
        
    # Get inputs (list formatted) from file.
    assert FLAGS.srcFile is not None
    assert FLAGS.firstPFile is not None
    assert FLAGS.tgtFile is not None
    
    [sorted_inputs, sorted_firstP, sorted_targets], sorted_keys = \
            get_sorted_inputs(FLAGS.srcFile, FLAGS.firstPFile, FLAGS.tgtFile)
            
    num_decode_batches = (len(sorted_inputs) - 1) // FLAGS.eval_batch + 1

    assert len(sorted_inputs) == len(sorted_firstP) == len(sorted_targets)
    
    tf.logging.info("Writing decodes into %s" % FLAGS.scoreFile)
    outfile = tf.gfile.Open(FLAGS.scoreFile, "w")
    
    # Generate hyper-parameters.
    hparams = utils.create_hparams(FLAGS.hparams_set, FLAGS.data_dir, passed_hparams=FLAGS.hparams)
    utils.add_problem_hparams(hparams, FLAGS.problems)
    # Create input function.
    num_datashards = utils.devices.data_parallelism().n
    mode = tf.estimator.ModeKeys.EVAL
    input_fn = utils.input_fn_builder.build_input_fn(mode, hparams,
                                                     data_dir=FLAGS.data_dir,
                                                     num_datashards=num_datashards,
                                                     worker_replicas=FLAGS.worker_replicas,
                                                     worker_id=FLAGS.worker_id,
                                                     batch_size=FLAGS.eval_batch)

    # Get wrappers for feeding datas into models.
    inputs, target = input_fn()
    features = inputs
    features['targets'] = target
    inputs_vocab = hparams.problems[0].vocabulary["inputs"]
    targets_vocab = hparams.problems[0].vocabulary["targets"]
    feed_iters = input_iter(0, num_decode_batches,
                            sorted_inputs, sorted_firstP, sorted_targets,
                            inputs_vocab, targets_vocab)

    model_fn = utils.model_builder.build_model_fn(FLAGS.model, problem_names=[FLAGS.problems],
                                                  train_steps=FLAGS.train_steps,
                                                  worker_id=FLAGS.worker_id, worker_replicas=FLAGS.worker_replicas,
	                                              eval_run_autoregressive=FLAGS.eval_run_autoregressive,
	                                              decode_hparams=decoding.decode_hparams(FLAGS.decode_hparams))
    est_spec = model_fn(features, target, mode, hparams)
    score, _ = metrics.padded_neg_log_perplexity(est_spec.predictions['predictions'], target)
    score = tf.reduce_sum(score, axis=[1,2,3])
    
    # Create session.
    sv = tf.train.Supervisor(logdir=FLAGS.output_dir, global_step=tf.Variable(0, dtype=tf.int64, trainable=False, name='global_step'))
    sess = sv.PrepareSession(config=tf.ConfigProto(allow_soft_placement=True))
    sv.StartQueueRunners(sess, tf.get_default_graph().get_collection(tf.GraphKeys.QUEUE_RUNNERS))

    sumt = 0
    scores_list = []
    
    # Loop for batched translation.
    for i, features in enumerate(feed_iters):
        t = time.time()
        inputs_ = features["inputs"]
        firstP_ = features["firstP"]
        targets_ = features["targets"]
        
        while inputs_.ndim < 4: inputs_ = np.expand_dims(inputs_, axis=-1)
        while firstP_.ndim < 4: firstP_ = np.expand_dims(firstP_, axis=-1)
        while targets_.ndim < 4: targets_ = np.expand_dims(targets_, axis=-1)
        scores = sess.run(score, feed_dict={inputs['inputs']: inputs_, inputs["firstP"]: firstP_, target: targets_})
        scores_list.extend(scores.tolist())
        dt = time.time() - t
        sumt += dt
        avgt = sumt / (i+1)
        needt = (num_decode_batches - i+1) * avgt
        
        print("Batch %d/%d worktime=(%s), lefttime=(%s)" % (i+1, num_decode_batches, time.strftime('%H:%M:%S',time.gmtime(sumt)),time.strftime('%H:%M:%S',time.gmtime(needt))))
     
    scores_list.reverse()

    # Write to file with the original order.
    for index in range(len(sorted_inputs)):
        outfile.write("%.8f\n" % (scores_list[sorted_keys[index]]))
    
if __name__ == "__main__":
    tf.app.run()
