#!/usr/bin/env python3
from tfv1perf import Evaluator
import numpy as np
import argparse


class TFv1Evaluator(Evaluator):
  def prepare_ipu_data(self,
                       input_pl_names: list,
                       keep_original=False,
                       batch_size=0,
                       repeat_cnt=0):
    data_dict = np.load("raw_data_wo_kv_load.npz")
    data_feed = {
        name: data_dict[(
            f'tensordict_standardkvparser_'
            f'{name.split(":")[0].split("_")[-1] if name.split(":")[0].split("_")[-1].isnumeric() else 0}_args'
        )]
        for name in input_pl_names
    }
    return data_feed, keep_original

  def prepare_cpu_data(self,
                       input_pl_names: list,
                       keep_original=False,
                       batch_size=0,
                       repeat_cnt=0):
    data_dict = np.load("raw_data_wo_kv_load.npz")
    data_feed = {
        name: data_dict[(
            f'tensordict_standardkvparser_'
            f'{name.split(":")[0].split("_")[-1] if name.split(":")[0].split("_")[-1].isnumeric() else 0}_args'
        )]
        for name in input_pl_names
    }
    return data_feed, keep_original


if __name__ == "__main__":
  parser = argparse.ArgumentParser("the kvparser data extraction")
  parser.add_argument("-cpu",
                      "--cpu-model",
                      dest="cpu_model_path",
                      required=True,
                      type=str)
  parser.add_argument("-ipu",
                      "--ipu-model",
                      dest="ipu_model_path",
                      required=True,
                      type=str)
  parser.add_argument("-bs",
                      "--batch-size",
                      dest="batch_size",
                      type=int,
                      default=1)
  parser.add_argument("-q",
                      "--num-of-ipus",
                      dest="num_ipus",
                      type=int,
                      default=1)

  args = parser.parse_args()

  evl = TFv1Evaluator(
      cpu_model_path=args.cpu_model_path,
      ipu_model_path=args.ipu_model_path,
      nums_of_ipus=args.num_ipus,
  )

  evl.diff(batch_size=args.batch_size)
  print(evl.diff_reporter[0].records)
