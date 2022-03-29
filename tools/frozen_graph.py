import tensorflow as tf
import os
import shutil
import argparse
from tensorflow.python.framework import graph_util
from tensorflow.python.saved_model import builder

def freeze_saved_model(saved_model_path, output_dir, tag=tf.saved_model.tag_constants.SERVING, output_tensors_names=None):
  if os.path.exists(output_dir):
    shutil.rmtree(output_dir, ignore_errors=True)
  saved_model_builder = builder.SavedModelBuilder(output_dir)
  with tf.Session() as sess:
    meta_graph = tf.saved_model.loader.load(sess, [tag] if tag else [], saved_model_path)
    if output_tensors_names:
      output_node_names = output_tensors_names
    else:
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
        signature_def_map=meta_graph.signature_def)
    saved_model_builder.save()

if __name__ == '__main__':
  parser = argparse.ArgumentParser("Comparison the model output")
  parser.add_argument("--dir",
            dest="model_path",
            default="saved_model_path",
            type=str)
  parser.add_argument("-o",
            "--output-dir",
            dest="output_dir",
            default="ipu_model",
            type=str)
  parser.add_argument("-o",
            "--tag",
            dest="tag",
            default="serve",
            type=str)
  parser.add_argument("-tensor",
            "--output-tensors-names",
            dest="output_tensors_names",
            default=None,
            nargs='+',
            type=str)

  args = parser.parse_args()
  freeze_saved_model(args.saved_model_path, args.output_dir, tag=args.tag)
