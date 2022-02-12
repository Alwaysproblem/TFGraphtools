#%%
import tensorflow.compat.v1 as tf
import os
import numpy as np
import argparse
import time
from tensorflow.core.protobuf import rewriter_config_pb2

from tensorflow.python import ipu

# Builds ipu_options
cfg = ipu.config.IPUConfig()
cfg.auto_select_ipus = 1

cfg.configure_ipu_system()

np.random.seed(1991)

tf.disable_v2_behavior()

# input_str = (
# """[dat]
# input_ids=276:101,3300,671,1372,6054,4281,679,2207,2552,6649,6822,671,1366,1282,5101,3918,4638,759,7027,102,1,122,1372,6054,4281,679,2207,2552,2957,6822,749,671,1366,1282,5101,3918,4638,3369,759,7027,7481,117,2124,4635,1921,2518,677,4260,758,5101,3241,677,4717,6230,3198,1348,3998,678,676,2
# input_mask=119:1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
# token_type_ids=119:0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
# unique_id=36:8d32328c-a32d-11eb-a0c2-b8599f4d8d8a"""
# )
#%%

parser = argparse.ArgumentParser()
parser.add_argument("-iter", dest = "num_iterations", type=int, default=2)
parser.add_argument("-b", dest = "batch_size", type=int, default=1)
parser.add_argument("-s", dest = "sequence_length", type=int, default=64)

args = parser.parse_args()

sess_cfg = tf.ConfigProto()
sess_cfg.graph_options.rewrite_options.memory_optimization = (
  rewriter_config_pb2.RewriterConfig.OFF
)

with tf.Session(config=sess_cfg) as sess:
  meta = tf.saved_model.loader.load(sess, ['serve'], "origin-l6-ipu-fast-wo-kv")
  # out_names = [ "bert/final_ids:0", "bert/final_scores:0"]
  out_names = [ i.name for i in meta.signature_def["serving_default"].outputs.values()]
  out_names_pl = [ sess.graph.get_tensor_by_name(o_name) for o_name in out_names]

  StandardKvParser_2 = sess.graph.get_tensor_by_name("TensorDict/StandardKvParser_2:0")
  StandardKvParser_3 = sess.graph.get_tensor_by_name("TensorDict/StandardKvParser_3:0")

  durations = []
  for iter_count in range(args.num_iterations):

    ids = np.array([101,3300,671,1372,6054,4281,679,2207,2552,6649,6822,671,1366,1282,5101,3918,4638,759,7027,102,1,122,1372,6054,4281,679,2207,2552,2957,6822,749,671,1366,1282,5101,3918,4638,3369,759,7027,7481,117,2124,4635,1921,2518,677,4260,758,5101,3241,677,4717,6230,3198,1348,3998,678,676,2] + [0] * 4)
    ids = np.tile(ids, (args.batch_size, 1))
  
    start = time.time()
    predictions = sess.run(out_names_pl, feed_dict={
      StandardKvParser_3: ids,
      StandardKvParser_2: np.ones((args.batch_size, args.sequence_length)),
    })
    stop = time.time()
    duration = stop - start
    print("predictions={}".format(predictions))
    durations.append(duration)  
    report_string = "Iter {0}: latency {1:<7.3} sec/sample at batch size = {2}.".format(iter_count, duration, args.batch_size)
    print(report_string)

  print("Average statistics excluding the first half iterations.")
  print("-------------------------------------------------------------------------------------------")
  durations = durations[len(durations)//2:]
  print("Average latency: {:.5f}".format(np.mean(durations)))

#   total_start = time.time()
#   for iter_count in range(args.num_iterations):
#     ids = np.array([101,3300,671,1372,6054,4281,679,2207,2552,6649,6822,671,1366,1282,5101,3918,4638,759,7027,102,1,122,1372,6054,4281,679,2207,2552,2957,6822,749,671,1366,1282,5101,3918,4638,3369,759,7027,7481,117,2124,4635,1921,2518,677,4260,758,5101,3241,677,4717,6230,3198,1348,3998,678,676,2] + [0] * 4)
#     ids = np.tile(ids, (args.batch_size, 1))
  
#     start = time.time()
#     predictions = sess.run(out_names_pl, feed_dict={
#       StandardKvParser_3: ids,
#       StandardKvParser_2: np.ones((args.batch_size, args.sequence_length)),
#     })

#   total_duration = time.time() - total_start

#   print("Throughput at bs={}: {} samples/sec.".format(args.batch_size, args.batch_size * args.num_iterations/total_duration))