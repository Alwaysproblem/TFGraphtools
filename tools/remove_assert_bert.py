import tensorflow as tf
import numpy as np
import shutil
from tensorflow.python.framework import tensor_util
from tensorflow.core.framework import types_pb2
from tfv1perf.utils import load_tf_graph
import argparse

def save(graph_def, output_dir,
          meta,
          tag=tf.saved_model.tag_constants.SERVING):
  with tf.Graph().as_default():
    tf.import_graph_def(graph_def, name="")
    with tf.Session() as sess:
      shutil.rmtree(output_dir, ignore_errors=True)
      builder = tf.saved_model.builder.SavedModelBuilder(output_dir)
      builder.add_meta_graph_and_variables(
          sess,
          [tag],
          signature_def_map=meta.signature_def,
      )
      builder.save()

def main(model_dir, output_dir):
  graph, meta = load_tf_graph(model_dir)
  graph_def = graph.as_graph_def()
  graph_def_copy = tf.GraphDef()
  graph_def_copy.CopyFrom(graph_def)

  for idx, node in enumerate(graph_def.node):
    if 'assert_less_equal' in node.name:
      graph_def_copy.node.remove(node)

  for node in graph_def_copy.node:
    for i in node.input:
      if i.startswith('^') and "assert_less_equal" in i:
        node.input.remove(i)

  with tf.Graph().as_default():
    tf.import_graph_def(graph_def_copy, name="")

  save(graph_def_copy, output_dir=output_dir, meta=meta)

if __name__ == "__main__":
  parser = argparse.ArgumentParser("Comparison the model output")
  parser.add_argument("--dir",
                      dest="saved_model_path",
                      default="original_model",
                      type=str)
  parser.add_argument("-o",
                      "--output-dir",
                      dest="output_dir",
                      default="ipu_model",
                      type=str)
  parser.add_argument("-t", "--tag", dest="tag", default="serve", type=str)

  args = parser.parse_args()
  main(args.saved_model_path, args.output_dir)
