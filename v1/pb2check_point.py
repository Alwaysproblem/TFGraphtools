import tensorflow.compat.v1 as tf
import os
import numpy as np

input_str = (
"""[dat]
input_ids=276:101,3300,671,1372,6054,4281,679,2207,2552,6649,6822,671,1366,1282,5101,3918,4638,759,7027,102,1,122,1372,6054,4281,679,2207,2552,2957,6822,749,671,1366,1282,5101,3918,4638,3369,759,7027,7481,117,2124,4635,1921,2518,677,4260,758,5101,3241,677,4717,6230,3198,1348,3998,678,676,2
input_mask=119:1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
token_type_ids=119:0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
unique_id=36:8d32328c-a32d-11eb-a0c2-b8599f4d8d8a"""
)

with tf.Session() as sess:
    meta = tf.saved_model.loader.load(sess, ['serve'], "orgin/ipu_query_gen9/export/")
    out_names = [ i.name for i in meta.signature_def["serving_default"].outputs.values()]
    out_names_pl = [ sess.graph.get_tensor_by_name(o_name) for o_name in out_names]
    saver = tf.train.Saver()
    o = sess.run(out_names_pl, feed_dict={sess.graph.get_tensor_by_name("TensorDict/batch:0"): (input_str,)})
    print(o)
    save_path = saver.save(sess, "checkpoint/model.ckpt")
    print("Model saved in path: %s" % save_path)