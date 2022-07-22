import shutil
import argparse
import tensorflow as tf

tf.disable_v2_behavior()

model_name = "models/shoucai/shoucai_ipu"
export_savedModel_path = "shoucai_gpu"


def save(graph_def, output_dir, meta):
  with tf.Graph().as_default():
    tf.import_graph_def(graph_def, name="")
    with tf.Session() as sess:
      shutil.rmtree(output_dir, ignore_errors=True)
      builder = tf.saved_model.builder.SavedModelBuilder(output_dir)
      builder.add_meta_graph_and_variables(
          sess,
          [tf.saved_model.tag_constants.SERVING],
          signature_def_map=meta.signature_def,
      )
      builder.save()

def clear_devices(model_path,
                  output_dir,
                  tag=tf.saved_model.tag_constants.SERVING):
  with tf.Session() as sess:
    meta = tf.saved_model.loader.load(sess, [tag] if tag else [],
                                      model_path,
                                      clear_devices=True)
  graph_def = meta.graph_def
  for node in graph_def.node:
    if "_class" in node.attr:
      del node.attr["_class"]
    if '_XlaCompile' in node.attr:
      del node.attr['_XlaCompile']
    if '_XlaScope' in node.attr:
      del node.attr['_XlaScope']
    if '_XlaSeparateCompiledGradients' in node.attr:
      del node.attr['_XlaSeparateCompiledGradients']
    if '_XlaSharding' in node.attr:
      del node.attr['_XlaSharding']

  save(graph_def, output_dir=output_dir, meta=meta)


if __name__ == '__main__':
  parser = argparse.ArgumentParser("Comparison the model output")
  parser.add_argument("--dir",
                      dest="saved_model_path",
                      default=model_name,
                      type=str)
  parser.add_argument("-o",
                      "--output-dir",
                      dest="output_dir",
                      default=export_savedModel_path,
                      type=str)
  parser.add_argument("-t", "--tag", dest="tag", default="serve", type=str)

  args = parser.parse_args()
  clear_devices(args.saved_model_path, args.output_dir, tag=args.tag)