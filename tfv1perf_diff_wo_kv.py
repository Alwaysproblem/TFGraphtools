from tfv1perf import Evaluator
import numpy as np
import argparse

class TFv1Evaluator(Evaluator):
  def prepare_ipu_data(self, input_pl_names: list, keep_original=False, batch_size=0, repeat_cnt=0):
    data_dict = np.load("raw_data_wo_kv_load.npz")
    data_feed = {
        name: data_dict[(f'tensordict_standardkvparser_'
        f'{name.split(":")[0].split("_")[-1] if name.split(":")[0].split("_")[-1].isnumeric() else 0}_args')]
        for name in input_pl_names}
    return data_feed, keep_original

  def prepare_cpu_data(self, input_pl_names: list, keep_original=True, batch_size=0, repeat_cnt=0):
    with open(f"{self.cpu_model.model_path}/raw_data.dat") as dat_file:
      dat_content = dat_file.read().strip().split('[dat]')

    input_str_list = []
    for s in dat_content:
      if s and not s.startswith("File://"):
        s = '[dat]' + s
        input_str_list.append(s)

    input_str_list_repeat = input_str_list * repeat_cnt
    input_strs = [ "".join(input_str_list_repeat[i: i + batch_size]) for i in range(0, len(dat_content), batch_size) if "".join(input_str_list_repeat[i: i + batch_size])]
    tensordict_batch = input_pl_names[0]

    return [{tensordict_batch: (input_s,)} for input_s in input_strs], keep_original

if __name__ == "__main__":
  parser = argparse.ArgumentParser("the kvparser data extraction")
  parser.add_argument("-cpu", "--cpu-model", dest="cpu_model_path", required=True, type=str)
  parser.add_argument("-ipu", "--ipu-model", dest="ipu_model_path", required=True, type=str)
  parser.add_argument("-bs", "--batch-size", dest="batch_size", type=int, default=1)
  parser.add_argument("-q", "--num-of-ipus", dest="num_ipus", type=int, default=1)

  args = parser.parse_args()

  evl = TFv1Evaluator(
      cpu_model_path=args.cpu_model_path,
      ipu_model_path=args.ipu_model_path,
      nums_of_ipus=args.num_ipus,
  )

  evl.diff(batch_size=2)
  print(evl.diff_reporter[0].records)