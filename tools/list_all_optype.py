from tfv1perf.utils import load_tf_graph
import argparse


def main(model_dir):
  graph, meta = load_tf_graph(model_dir)
  graph_def = graph.as_graph_def()
  print(set( node.op for node in graph_def.node))


if __name__ == "__main__":
  parser = argparse.ArgumentParser("Comparison the model output")
  parser.add_argument("--dir",
                      dest="saved_model_path",
                      default="original_model",
                      type=str)

  args = parser.parse_args()
  main(args.saved_model_path)