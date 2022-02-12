import tensorflow as tf
import os
import shutil
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import builder

def freeze_saved_model(saved_model_path, output_dir, tag = tf.saved_model.tag_constants.SERVING):
    saved_model_builder = builder.SavedModelBuilder(output_dir)
    with tf.Session() as sess:
        meta_graph = tf.saved_model.loader.load(sess, [tag] if tag else [], saved_model_path)
        signature = meta_graph.signature_def
        output_node_names = [ tens.name.split(':')[0] for tens in meta_graph.signature_def['serving_default'].outputs.values() ]
        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=sess.graph_def,
            output_node_names=output_node_names
        )
    with tf.Graph().as_default():
        tf.import_graph_def(output_graph_def, name="")
        # We don't use any specific converter here.
        with tf.Session() as sess:
            saved_model_builder.add_meta_graph_and_variables(
                sess,
                [tag] if tag else [],
                signature_def_map=signature)
        saved_model_builder.save()

if __name__ == '__main__':
    saved_model_path = '/data/yongxi/Desktop/unilm/orgin/ipu_query_gen9/export'
    output_dir = '/data/yongxi/Desktop/unilm/frozen_cpu'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)
    freeze_saved_model(saved_model_path, output_dir, tag = "serve")