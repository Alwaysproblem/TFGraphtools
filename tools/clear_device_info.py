import shutil
import argparse
import tensorflow as tf

tf.disable_v2_behavior()

model_name = "models/ljf-402-wo-kv"
export_savedModel_path = f"{model_name}-clear-device"


def clear_devices(model_path,
                  output_dir,
                  tag=tf.saved_model.tag_constants.SERVING):
  with tf.Session() as sess:
    meta = tf.saved_model.loader.load(sess, [tag] if tag else [],
                                      model_path,
                                      clear_devices=True)

    shutil.rmtree(output_dir, ignore_errors=True)
    builder = tf.saved_model.builder.SavedModelBuilder(output_dir)
    builder.add_meta_graph_and_variables(
        sess,
        [tf.saved_model.tag_constants.SERVING],
        signature_def_map=meta.signature_def,
    )
    builder.save()


if __name__ == '__main__':
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
  clear_devices(args.saved_model_path, args.output_dir, tag=args.tag)
