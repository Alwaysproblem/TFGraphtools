import configparser
import subprocess
import argparse
import json
import os
import shutil

WIDTH = os.get_terminal_size().columns


def check_exist_image(image_name):
  return subprocess.check_output(f"docker images -q {image_name}".split())


def build_test_env(docker_tag="",
                   BASE="",
                   UID=0,
                   GID=0,
                   latestsdk=None,
                   popsdk=None):

  default_dictionary = {
      "docker_tag":
      "saved_model_tool:test",
      "BASE":
      "centos:7.2.1511",
      "UID":
      30017,
      "GID":
      501,
      "latestsdk":
      'https://artifactory.sourcevertex.net/poplar-sdk-builds/poplar_sdk_centos/versioned/master/2.6.0-EA.1/2022-05-25_22%3A04%3A45_4cc3a2e115/centos_7_6_installer/poplar_sdk-centos_7_6-2.6.0-EA.1%2B1012-4cc3a2e115.tar.gz',
      "popsdk":
      'https://artifactory.sourcevertex.net/poplar-sdk-builds/poplar_sdk_centos/versioned/sdk-release-2.3/2.3.1/2021-11-03_16%3A08%3A58_89796d462d/centos_7_6_installer/poplar_sdk-centos_7_6-2.3.1%2B793-89796d462d.tar.gz'
  }
  if docker_tag:
    default_dictionary['docker_tag'] = docker_tag
  if BASE:
    default_dictionary['BASE'] = BASE
  if UID:
    default_dictionary['UID'] = UID
  if GID:
    default_dictionary['GID'] = GID
  if latestsdk:
    default_dictionary['latestsdk'] = latestsdk
  if popsdk:
    default_dictionary['popsdk'] = popsdk

  return subprocess.check_call([
      "docker", "build", "--no-cache", "-t",
      f"{default_dictionary['docker_tag']}", "-f",
      "saved_model_cli_ci.dockerfile", "--build-arg",
      f"BASE={default_dictionary['BASE']}", "--build-arg",
      f"UID={default_dictionary['UID']}", "--build-arg",
      f"GID={default_dictionary['GID']}", "--build-arg",
      f"latestsdk={default_dictionary['latestsdk']}", "--build-arg",
      f"popsdk={default_dictionary['popsdk']}", "--network=host", "."
  ])


def destory_test_env(docker_name):
  subprocess.check_call(["docker", "image", "rm", "-f", docker_name])


def run_test(docker_tag,
             test_path,
             ipu_config_file,
             config_json_dict=None,
             config_json_file=None,
             docker_name="ismt-sdk-test",
             job_index=0,
             batch_sizes="1",
             clean_only=False):

  config_json_content = {}

  if not config_json_dict and not config_json_file:
    return

  if config_json_dict:
    config_json_content = config_json_dict
    with open(os.path.join(test_path, "config_test.json"), "w") as f:
      json.dump(config_json_dict, f, indent=2)

  if config_json_file:
    with open(config_json_file, "r") as fr:
      c = fr.read()
      config_json_content = json.loads(c)
      with open(os.path.join(test_path, "config_test.json"), "w") as fw:
        fw.write(c)

  precommand = ("docker run "
                "  --privileged --rm "
                f" -v {ipu_config_file}:/etc/ipuof.conf.d/p.conf "
                "  --net=host --ulimit memlock=-1:-1 "
                f" -ti --name {docker_name}-{job_index} "
                f" -v {test_path}:/home/scotty/workflow {docker_tag} "
                " --convert-config config_test.json --log debug")

  if clean_only:
    subprocess.check_call((precommand + " --action clean -y").split())
    if config_json_content.get("pipeline_cfg", None):
      if config_json_content["pipeline_cfg"].get("profiling_root_dir", None):
        shutil.rmtree(
            config_json_content["pipeline_cfg"]["profiling_root_dir"],
            ignore_errors=True)
      if config_json_content["pipeline_cfg"].get("solution_dir", None):
        shutil.rmtree(config_json_content["pipeline_cfg"]["solution_dir"],
                      ignore_errors=True)
    if config_json_content.get("embedded_runtime_save_config", None):
      shutil.rmtree(config_json_content["embedded_runtime_save_config"]
                    ["embedded_runtime_exec_cachedir"],
                    ignore_errors=True)
    return

  if not config_json_content.get("pipeline_cfg", None):
    subprocess.check_call((precommand + " --action convert diff --rm -y").split())
    subprocess.check_call((precommand + " --action tput --batch-sizes " +
                     batch_sizes if batch_sizes else "1").split())
  else:
    subprocess.check_call((precommand + " --action convert diff tput --batch-sizes --rm -y" +
                     batch_sizes if batch_sizes else "1").split())


def run_tests(config, clean_only=False):
  for test_config in filter(lambda x: x.startswith("test_"),
                            config.sections()):
    if config[test_config].get('path', ""):
      if not clean_only:
        print(f"Start test in the section {test_config}".center(WIDTH, "="))
        print(f"Run test in the path {config[test_config]['path']}")
      else:
        print(f"Clean files in the path {config[test_config]['path']}".center(
            WIDTH, "="))
      if ('{' in config[test_config]['config_json']
          or '}' in config[test_config]['config_json']):
        run_test(config['env']['test_image_tag'],
                 config[test_config]['path'],
                 config['env']['ipu_config_file'],
                 config_json_dict=json.loads(
                     config[test_config]['config_json']),
                 docker_name=config['env']['test_image_name'],
                 batch_sizes=config[test_config]['batch_sizes'],
                 clean_only=clean_only)
      else:
        run_test(config['env']['test_image_tag'],
                 config[test_config]['path'],
                 config['env']['ipu_config_file'],
                 config_json_file=config[test_config]['config_json'],
                 docker_name=config['env']['test_image_name'],
                 batch_sizes=config[test_config]['batch_sizes'],
                 clean_only=clean_only)
      print(f"Done".center(WIDTH, "="))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-f",
                      "--file",
                      default="config.ini",
                      action='store',
                      dest="file")
  parser.add_argument(
      "-c",
      "--clean",
      default=False,
      action="store_true",
  )
  parser.add_argument(
      "-d",
      "--destroy-test-env",
      default=False,
      action="store_true",
  )
  parser.add_argument(
      "-b",
      "--build-test-env",
      default=False,
      action="store_true",
  )
  args = parser.parse_args()
  config = configparser.ConfigParser()
  config.read(args.file)

  if args.build_test_env:
    build_test_env(
        config['env']['test_image_tag'],
        BASE=config['env']['base'],
        UID=config['env']['uid'],
        GID=config['env']['gid'],
        popsdk=config['env']['popsdk'],
        latestsdk=config['env']['latestsdk'],
    )
    return

  if args.destroy_test_env:
    destory_test_env(config['env']['test_image_tag'])
    return

  o = check_exist_image(config['env']['test_image_tag'])
  if not o:
    build_test_env(
        config['env']['test_image_tag'],
        BASE=config['env']['base'],
        UID=config['env']['uid'],
        GID=config['env']['gid'],
        popsdk=config['env']['popsdk'],
        latestsdk=config['env']['latestsdk'],
    )
  if args.clean:
    return run_tests(config, clean_only=args.clean)
  run_tests(config)
  # destory_test_env(config['env']['test_image_tag'])

if __name__ == "__main__":
  main()