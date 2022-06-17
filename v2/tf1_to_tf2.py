import sys
import argparse
import tensorflow as tf
import tensorflow.compat.v1 as tf1

def load_tf1(model_path, keras=False):
  pass

def tf2_function_wrapper(model):
  pass

def parse_args(args):
  parser = argparse.ArgumentParser(description='Convert a TF2 model to TF1')
  parser.add_argument('-m', '--model_path', type=str, help='Path to the TF2 model')
  parser.add_argument('-o', '--output_path', type=str, help='Path to save the TF1 model')
  return parser.parse_args(args)

def main():
  pass

if __name__ == '__main__':
  sys.exit(main())