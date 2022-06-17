#!/bin/bash

# build ci docker image
ci_docker_tag=saved_model_tool:test

docker build \
  -t ${ci_docker_tag} \
  -f saved_model_cli_ci.dockerfile \
  --network=host \
  --build-arg \
  BASE=reg.docker.alibaba-inc.com/zwf98950/script_worker_base:enable_network .

ci_docker_id=$(docker images ${ci_docker_tag} --format {{.ID}})

alias run_docker_cli='docker run -it \
           --privileged \
           --ipc=host \
           --net=host \
           --ulimit memlock=-1:-1 \
           --cap-add=IPC_LOCK \
           --device=/dev/infiniband/ \
           -v /etc/ipuof.conf.d/:/etc/ipuof.conf.d/ \
           --name ismt-ci \
           -v /data:/data \
           saved_model_tool:test \
           /bin/bash'

