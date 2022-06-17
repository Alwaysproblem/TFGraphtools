import sys
import os
import argparse
import tensorflow as tf2
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import tensorflow.compat.v1 as tf1


def select_output_path(args):
  """Select the output path for the converted model.

  Args:
      args (argparse.Namespace): The parsed arguments.

  Returns:
      list: [input_model_path, output_model_path]
  """
  tf2_model_path = args.tf2_model_path
  tf1_model_path = args.tf1_model_path

  if not tf2_model_path or not tf1_model_path:
    raise ValueError(
        'Please specify both --tf2-model-path and --tf1-model-path.')

  if (os.path.exists(tf2_model_path) and not os.path.exists(tf1_model_path)):
    return tf2_model_path, tf1_model_path
  elif (not os.path.exists(tf2_model_path) and os.path.exists(tf1_model_path)):
    return tf1_model_path, tf2_model_path

  raise ValueError('Both --tf2-model-path and --tf1-model-path exist.')


def load_tf2(model_path, keras_flag=False):
  if keras_flag:
    model = tf2.keras.model.load()


def prune_noops(model):
  pass


def load_tf1(model_path, tag=[tf1.saved_model.tag_constants.SERVING]):
  with tf1.Session() as sess:
    return tf1.saved_model.loader.load(sess, tag, model_path)


def parse_args(args):
  parser = argparse.ArgumentParser(
      description='Conversion between the TF2 model and the TF1 frozen model.')
  parser.add_argument('-tfv2',
                      '--tf2-model-path',
                      type=str,
                      help='Path to the TF2 model.')
  parser.add_argument('-tfv1',
                      '--tf1-model-path',
                      type=str,
                      help='Path to the TF1 model.')
  parser.add_argument(
      '-k',
      '--keras',
      action='store_true',
      help=('if set, load it with TF2 keras api. '
            'It does not work when converting from TF1 to TF2.'))
  return parser.parse_args(args)


def main():
  pass


if __name__ == '__main__':
  sys.exit(main())
