import tensorflow.compat.v1 as tf
from tensorflow.python.ipu import (ipu_compiler, ipu_infeed_queue, ipu_outfeed_queue, loops)
from tensorflow.python.ipu.utils import (create_ipu_config, set_ipu_model_options, auto_select_ipus, configure_ipu_system)
import numpy as np
import os
import shutil
import re
from uuid import uuid4
from datetime import datetime

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import ipu
from tensorflow.python.data.ops.dataset_ops import Dataset

tf.disable_v2_behavior()

config = create_ipu_config(profiling=True, use_poplar_text_report=False)
config = auto_select_ipus(config, 1)
configure_ipu_system(config=config)

batch_size = 1
max_sequence_length = 32

inputs_shape_and_dtype = [("input_ids", (max_sequence_length, ), np.int32), 
                          ("input_mask", (max_sequence_length, ), np.int32), 
                          ("segment_ids", (max_sequence_length, ), np.int32)]

dataset = tf.data.Dataset.from_tensors(
        tuple([np.random.randint(10, size=shape).astype(dtype)
        for _, shape, dtype in inputs_shape_and_dtype])
    )
dataset = dataset.repeat()
dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
dataset = dataset.shuffle(batch_size * 100)
dataset = dataset.cache()

infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)
outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

with tf.Graph().as_default():
    with tf.Session() as sess:
        meta_graph = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], "bertBfrozen")
        bert_graph_def = sess.graph_def

bert_graph_def_new = tf.Graph().as_graph_def()
bert_graph_def_new.CopyFrom(bert_graph_def)

for n in bert_graph_def_new.node:
    if n.op == "Placeholder":
        bert_graph_def.node.remove(n)


NUM_ITERATIONS = 1

def inference_step(input_ids, input_mask, segment_ids):
    input_map = {
        "input_ids:0": input_ids,
        "input_mask:0": input_mask,
        "segment_ids:0": segment_ids,
    }
    pooler_output = tf.import_graph_def(bert_graph_def, name="", 
        input_map=input_map, return_elements=["bert/pooler/dense/Tanh:0"])

    outfeed = outfeed_queue.enqueue(pooler_output)
    return outfeed


def inference_loop():
    r = loops.repeat(NUM_ITERATIONS, inference_step, inputs=[], infeed_queue = infeed_queue)
    return r

embedded_runtime_mode = True

if embedded_runtime_mode:
    sess_cfg = tf.ConfigProto()
    # sess_cfg.log_device_placement = True
    sess_cfg.graph_options.rewrite_options.memory_optimization = (
        rewriter_config_pb2.RewriterConfig.OFF
    )

    exec_path = "bert-base-exec/app.poplar_exec"
    runtime_api_timeout_us = 5000
    export_savedModel_path = "bert-base-poplar-exec"

    with tf.Session(config=sess_cfg) as sess:
        compile_op = ipu.ops.application_compile_op.experimental_application_compile_op(
                                inference_loop, output_path=exec_path)
        sess.run(compile_op)


    ctx = ipu.embedded_runtime.embedded_runtime_start(
                exec_path, [], "application", timeout=runtime_api_timeout_us)

    with tf.Graph().as_default():
        ctx = ipu.embedded_runtime.embedded_runtime_start(
                exec_path, [], "application", timeout=runtime_api_timeout_us)

        input_ids = tf.placeholder(tf.int32, shape = [batch_size, max_sequence_length], name = "input_ids")
        input_mask = tf.placeholder(tf.int32, shape = [batch_size, max_sequence_length], name = "input_mask")
        segment_ids = tf.placeholder(tf.int32, shape = [batch_size, max_sequence_length], name = "segment_ids")

        result = ipu.embedded_runtime.embedded_runtime_call([input_ids, input_mask, segment_ids], ctx)

        shutil.rmtree(export_savedModel_path, ignore_errors=True)
        builder = tf.saved_model.builder.SavedModelBuilder(export_savedModel_path)
        with tf.Session() as sess:
            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={ 
                        "input_ids": tf.saved_model.utils.build_tensor_info(input_ids),
                        "input_mask": tf.saved_model.utils.build_tensor_info(input_mask),
                        "segment_ids": tf.saved_model.utils.build_tensor_info(segment_ids),
                    },
                    outputs={ "pooler_output": tf.saved_model.utils.build_tensor_info(result[0]) },
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
                )
            )

            with tf.Session() as sess:
                builder.add_meta_graph_and_variables(
                    sess, [tf.saved_model.tag_constants.SERVING],
                    signature_def_map={
                        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                            prediction_signature 
                    },
                )
                builder.save()

else:
    with tf.device("/device:IPU:0"):
        run_loop = ipu_compiler.compile(inference_loop, inputs=[])

    dequeue_outfeed = outfeed_queue.dequeue()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(infeed_queue.initializer)
        sess.run(run_loop)
        print(sess.run(dequeue_outfeed))