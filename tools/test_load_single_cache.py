import os
import time
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python import ipu
from tensorflow.compiler.plugin.poplar.driver import config_pb2
from tensorflow.python.ipu import utils
import argparse

TF_2_NP = {
    tf.float32: np.float32,
    tf.int32: np.int32,
    tf.float16: np.float16,
    tf.int64: np.int64,
    tf.bool: np.bool
}


def ipu_cfg(model_path):
  # Builds ipu_options
  os.environ[
      "TF_POPLAR_FLAGS"] = f'--executable_cache_path={model_path} --show_progress_bar=true'
  if os.path.exists(f"{model_path}/ipu_cfg.bin"):
    with open(f"{model_path}/ipu_cfg.bin", 'rb') as f:
      _ipu_cfg = config_pb2.IpuOptions()
      _ipu_cfg.ParseFromString(f.read())
    utils.configure_ipu_system(_ipu_cfg)
  else:
    # Builds ipu_option
    partials_type = 'half'
    cfg = ipu.config.IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.convolutions.poplar_options = {"partialsType": partials_type}
    cfg.matmuls.poplar_options = {"partialsType": partials_type}
    cfg.configure_ipu_system()


np.random.seed(1991)
tf.disable_v2_behavior()


def run_model(model_path, tag=None, tput=False, bs=1, kvparser_idx=[]):
  with tf.Session(graph=tf.Graph()) as sess:
    meta = tf.saved_model.loader.load(
        sess,
        [tag] if tag is not None else tf.saved_model.tag_constants.SERVING,
        model_path)
    # output_op_names = [ f"TensorDict/StandardKvParser:{i}" for i in range(4) ]
    input_op_names = [f"TensorDict/StandardKvParser:{i}" for i in kvparser_idx]
    output_op_names = [
        i.name for i in meta.signature_def["serving_default"].outputs.values()
    ]
    #input_op_names = [ i.name for i in meta.signature_def["serving_default"].inputs.values()]
    input_names_pl = [
        sess.graph.get_tensor_by_name(o_name) for o_name in input_op_names
    ]
    out_names_pl = [
        sess.graph.get_tensor_by_name(o_name) for o_name in output_op_names
    ]
    inputs_name_shape_dtype = [(pl, [
        bs,
    ] + pl.get_shape().as_list()[1:], pl.dtype) for pl in input_names_pl]

    feed_dict = {
        it: np.random.randint(low=0, high=100,
                              size=ishape).astype(TF_2_NP[idtype])
        for it, ishape, idtype in inputs_name_shape_dtype
    }

    if tput:
      for _ in range(10):
        o = sess.run(out_names_pl, feed_dict=feed_dict)

      durs = []
      for _ in range(100):
        s = time.perf_counter()
        o = sess.run(out_names_pl, feed_dict=feed_dict)
        durs.append(time.perf_counter() - s)
      print(f"{np.array(durs).mean() * 1000}ms")
    else:
      o = sess.run(out_names_pl, feed_dict=feed_dict)

  return o


if __name__ == "__main__":

  parser = argparse.ArgumentParser("the kvparser data extraction")
  parser.add_argument("-m",
                      "--model",
                      dest="model_path",
                      required=True,
                      type=str,
                      action='store')
  parser.add_argument("-bs",
                      "--batch-size",
                      dest="batch_size",
                      type=int,
                      default=1,
                      action='store')
  parser.add_argument("--tput",
                      dest="tput",
                      default=False,
                      action='store_true')
  parser.add_argument("-kidx",
                      "--kvparser-idx",
                      dest="kvparser_idx",
                      type=int,
                      action='store',
                      help="the wanted output index of kvparser",
                      default=[0],
                      nargs='+')
  args = parser.parse_args()
  ipu_cfg(args.model_path)
  standard_output = run_model(args.model_path,
                              tag="serve",
                              kvparser_idx=args.kvparser_idx,
                              tput=args.tput,
                              bs=args.batch_size)
  print(standard_output)
