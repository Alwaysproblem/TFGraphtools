import tensorflow

# from tensorflow.python.ipu import (ipu_compiler, ipu_infeed_queue, ipu_outfeed_queue, loops, scopes, utils)
# import tensorflow.compat.v1 as tf

import tensorflow.compat.v1 as tf
# from tensorflow_core.python import ipu
from tensorflow.python.ipu import (ipu_compiler, ipu_infeed_queue, ipu_outfeed_queue, loops, scopes, utils)
from tensorflow.python.ipu.scopes import ipu_scope
from tensorflow.python.ipu.utils import (create_ipu_config, set_ipu_model_options, auto_select_ipus, configure_ipu_system)
from tensorflow.python import keras
import numpy as np

tf.disable_v2_behavior()

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
        meta_graph = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], "bertLfrozen-ipu-dy")
        bert_graph_def = sess.graph_def


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

with tf.device("/device:IPU:0"):
    run_loop = ipu_compiler.compile(inference_loop, inputs=[])

dequeue_outfeed = outfeed_queue.dequeue()


config = create_ipu_config(profiling=True, use_poplar_text_report=False)
config = auto_select_ipus(config, 1)
configure_ipu_system(config=config)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(infeed_queue.initializer)
    sess.run(run_loop)
    print(sess.run(dequeue_outfeed))