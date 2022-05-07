import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.python.ipu.config import configure_ipu_system
from tensorflow.python import ipu
from tensorflow.compiler.plugin.poplar.driver import config_pb2


def ipu_cfg(model_path):
  if os.path.exists(os.path.join(model_path, "ipu_cfg.bin")):
    with open(os.path.join(model_path, "ipu_cfg.bin"), "rb") as f:
      _ipu_cfg = config_pb2.IpuOptions()
      _ipu_cfg.ParseFromString(f.read())
    configure_ipu_system(_ipu_cfg)
  else:
    cfg = ipu.config.IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()


def prepare_raw_data(summary_dir: str,
                     batch_size=1,
                     repeat_cnt=1,
                     device="cpu"):
  cpu_indata = dict(
      np.load(os.path.join(summary_dir, "cpu_indata_worse_case.npz"),
              allow_pickle=True))["arr_0"].tolist()
  ipu_indata = dict(
      np.load(os.path.join(summary_dir, "ipu_indata_worse_case.npz"),
              allow_pickle=True))["arr_0"].tolist()

  if device == "ipu":
    return ipu_indata

  return cpu_indata


def run_model_raw_data(model_path, output_pl_names=None, bs=1, summary_dir=None, device="cpu"):
  with tf.Session(graph=tf.Graph()) as sess:
    meta = tf.saved_model.loader.load(sess, ["serve"], model_path)
    if not output_pl_names:
      output_pl_names = [
          plname.name
          for plname in meta.signature_def["serving_default"].outputs.values()
      ]
    output_pl = [
        sess.graph.get_tensor_by_name(plname) for plname in output_pl_names
    ]
    data = prepare_raw_data(summary_dir,
                            batch_size=bs,
                            repeat_cnt=bs,
                            device=device)
    out = []
    for fd in data:
      o = sess.run(output_pl, feed_dict=fd)
      out.extend(o)
      print(o)
  return out


def check_same(s, f, threshold=0.01):
  def equals(i, j):
    if (i.dtype in (np.float, np.float64, np.float16, np.float32)
        and j.dtype in (np.float, np.float64, np.float16, np.float32)):
      return np.abs(i - j) < threshold
    return i == j

  return list(map(lambda x, y: np.all(equals(x, y)), s, f))


def print_to_file(o, device='cpu'):
  with open(f"{device}_out.txt", "w") as f:
    print(o, file=f)


if __name__ == "__main__":
  parser = argparse.ArgumentParser("Comparison the model output")
  parser.add_argument("-cpu",
                      "--cpu-device-model",
                      dest="cpu_model_path",
                      default="cpu_freeze",
                      type=str)
  parser.add_argument("-ipu",
                      "--ipu-device-model",
                      dest="ipu_model_path",
                      default="ipu_model",
                      type=str)
  parser.add_argument("-sum",
                      "--workflow-summary-dir",
                      dest="workflow_summary_dir",
                      default="workflow_summary",
                      type=str)
  parser.add_argument("-bs",
                      "--batch-size",
                      dest="batch_size",
                      default=1,
                      type=int)
  parser.add_argument("-o",
                      "--output-tensors-names",
                      dest="output_tensors_names",
                      default=None,
                      nargs='+',
                      type=str)

  args = parser.parse_args()
  cpu_o = run_model_raw_data(args.cpu_model_path,
                             args.output_tensors_names,
                             bs=args.batch_size,
                             summary_dir=args.workflow_summary_dir)
  print_to_file(cpu_o, device='cpu')
  ipu_cfg(args.ipu_model_path)
  ipu_o = run_model_raw_data(args.ipu_model_path,
                             args.output_tensors_names,
                             bs=args.batch_size,
                             summary_dir=args.workflow_summary_dir,
                             device="ipu")
  print_to_file(ipu_o, device='ipu')
  print(f"the check same: {check_same(cpu_o, ipu_o, 0.01)}")
