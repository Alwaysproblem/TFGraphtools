#!/usr/bin/env python3
from tfv1perf import Evaluator
import pandas as pd
import argparse

pd.set_option("display.max_columns", 20)


class TFv1Evaluator(Evaluator):
  pass


if __name__ == "__main__":

  parser = argparse.ArgumentParser("the kvparser data extraction")
  parser.add_argument("-m",
                      "--model",
                      dest="model_path",
                      required=True,
                      type=str)
  parser.add_argument("-bs",
                      "--batch-size",
                      dest="batch_size",
                      type=int,
                      default=1)
  parser.add_argument("-ipus",
                      "--num-of-ipus",
                      dest="num_ipus",
                      type=int,
                      default=1)
  parser.add_argument("-ps",
                      "--pipeline-stages",
                      dest="pipeline_stages",
                      type=int,
                      default=0)
  parser.add_argument("-ths", "--threads", dest="threads", type=int, default=1)
  parser.add_argument("-i",
                      "--iterations",
                      dest="iterations",
                      type=int,
                      default=100)
  args = parser.parse_args()

  if args.pipeline_stages is not None:
    threads = args.pipeline_stages if args.threads == 1 else args.threads
  else:
    threads = args.threads

  evl = TFv1Evaluator(
      ipu_model_path=args.model_path,
      nums_of_ipus=args.num_ipus,
      pipeline_stages=args.pipeline_stages,
      tput_pipeline_threads=threads,
      iterations=args.iterations,
  )

  evl.tput(args.batch_size)

  print(evl.tput_reporter[0].records)
