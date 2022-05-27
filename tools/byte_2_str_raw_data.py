import argparse
import os

RAW_DATA = "raw_data.dat"


def main(model_dir, overwrite=False):
  with open(os.path.join(model_dir, RAW_DATA)) as raw_data_f:
    content = raw_data_f.read()
  content = eval(content).decode('utf-8')
  if overwrite:
    with open(os.path.join(model_dir, RAW_DATA), 'w') as raw_data_fw:
      print(content, file=raw_data_fw)
    return
  with open(os.path.join(model_dir, f"{RAW_DATA}.bak"),
            'w') as raw_data_bak_fw:
    print(content, file=raw_data_bak_fw)


if __name__ == "__main__":
  parser = argparse.ArgumentParser("Comparison the model output")
  parser.add_argument("--dir",
                      dest="saved_model_path",
                      default="original_model",
                      type=str)
  parser.add_argument("--overwrite",
                      action="store_true",
                      default=False,
                      dest="overwrite")

  args = parser.parse_args()
  main(args.saved_model_path, args.overwrite)