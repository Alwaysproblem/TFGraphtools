ARG BASE=reg.docker.alibaba-inc.com/zwf98950/script_worker_base:0.4.5-rc1
FROM ${BASE}

ARG UID=30017
ARG GID=501

RUN groupadd -g ${GID} all && \
    useradd  -g ${GID} -u ${UID} -ms /bin/bash scotty && \
    usermod -a -G all root && \
    mkdir -p /home/scotty/sdk && \
    mkdir -p /home/scotty/Desktop && \
    mkdir -p /home/scotty/workflow && \
    chown scotty:all -R /home/scotty

ARG popsdk=https://artifactory.sourcevertex.net/poplar-sdk-builds/poplar_sdk_centos/versioned/sdk-release-2.3/2.3.1/2021-11-03_16%3A08%3A58_89796d462d/centos_7_6_installer/poplar_sdk-centos_7_6-2.3.1%2B793-89796d462d.tar.gz

RUN yum -y install nic-libs-mellanox-rdma.x86_64 -b test && \
    yum -y clean all  && \
    rm -rf /var/cache

RUN yum makecache fast && \
    rpm --rebuilddb && yum install -y python3 python3-devel && \
    rpm --rebuilddb && yum install -y nic-libs-mellanox-rdma.x86_64 -b test && \
    yum -y clean all && \
    rm -rf /var/cache

WORKDIR /home/scotty/sdk
RUN wget -d --header 'X-JFrog-Art-Api:AKCp8jQd49DG1mvtUS8YA2urpzzaapkmSecr9HaFjqMbWXt6tJD4NWNbY4EkXDZVtLj6ttyor' \
    --header 'Authorization: Bearer' ${popsdk}


COPY ./libstdc++.so.6.0.24 /lib64/libstdc++.so.6.0.24
COPY ./ld /usr/bin/ld

RUN python3 -m pip install --no-cache -U pip && \
    pip install --no-cache pyyaml virtualenv && \
    tar -zxvf /home/scotty/sdk/poplar_sdk-centos_7_6-2.3.1+793-89796d462d.tar.gz && \
    mv /home/scotty/sdk/poplar_sdk-centos_7_6-2.3.1+793-89796d462d/poplar-centos_7_6-2.3.0+1476-0537c534d3 /home/scotty/sdk/ && \
    rm -rf /home/scotty/sdk/poplar_sdk-centos_7_6-2.3.1+793-89796d462d.tar.gz /home/scotty/sdk/poplar_sdk-centos_7_6-2.3.1+793-89796d462d && \
    rm -f /lib64/libstdc++.so.6 && \
    ln -s /lib64/libstdc++.so.6.0.24 /lib64/libstdc++.so.6 && \
    echo "source /home/scotty/sdk/poplar-centos_7_6-2.3.0+1476-0537c534d3/enable.sh" >> ~/.bashrc && \
    echo "source /home/scotty/Desktop/ipu_tool_env/bin/activate" >> ~/.bashrc && \
    echo 'export LD_LIBRARY_PATH=/home/scotty/Desktop/ipu_tool_env/lib64/python3.6/site-packages/tensorflow_core/:$LD_LIBRARY_PATH' >> ~/.bashrc

ARG latestsdk=https://artifactory.sourcevertex.net/poplar-sdk-builds/poplar_sdk_centos/versioned/master/2.6.0-EA.1/2022-05-25_22%3A04%3A45_4cc3a2e115/centos_7_6_installer/poplar_sdk-centos_7_6-2.6.0-EA.1%2B1012-4cc3a2e115.tar.gz
WORKDIR /home/scotty/Desktop
COPY ./tensorflow-1.15.5-cp36-cp36m-linux_x86_64.whl /home/scotty/Desktop/tensorflow-1.15.5-cp36-cp36m-linux_x86_64.whl
COPY ./copy_saved_model_tool.sh /home/scotty/Desktop/copy_saved_model_tool.sh

RUN git clone -b phb-ali-master  https://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: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@github.com/graphcore/ipu_saved_model_tool.git && \
    mkdir -p /home/scotty/Desktop/ipu_tool_env && \
    python3 -m virtualenv ipu_tool_env --python=python3 && \
    /home/scotty/Desktop/ipu_tool_env/bin/python3 -m pip install --no-cache -U pip && \
    /home/scotty/Desktop/ipu_tool_env/bin/python3 -m pip install --no-cache ./tensorflow-1.15.5-cp36-cp36m-linux_x86_64.whl && \
    bash ./copy_saved_model_tool.sh ${latestsdk} && \
    cd ipu_saved_model_tool/toolkits/ali-tf-toolsets && \
    /home/scotty/Desktop/ipu_tool_env/bin/python3 -m pip install --no-cache -e . && \
    rm -rf ./tensorflow-1.15.5-cp36-cp36m-linux_x86_64.whl && \
    rm -rf /home/scotty/sdk/poplar_sdk-centos_7_6-2.3.1+793-89796d462d.tar.gz

ENV TF_POPLAR_FLAGS='--show_progress_bar=true'
WORKDIR /home/scotty/workflow
COPY venv_python3 /usr/local/bin/venv_python3
ENTRYPOINT [ "venv_python3", "/home/scotty/Desktop/ipu_saved_model_tool/toolkits/start_workflow.py" ]
