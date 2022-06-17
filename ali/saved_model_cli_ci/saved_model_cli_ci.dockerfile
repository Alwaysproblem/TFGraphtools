ARG BASE=centos:7.2.1511
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

RUN yum update -y && \
    yum install epel-release -y && \
    yum install -y centos-release-scl-rh && \
    yum install -y https://packages.endpointdev.com/rhel/7/os/x86_64/endpoint-repo.x86_64.rpm && \
    yum install -y https://repo.ius.io/ius-release-el7.rpm && \
    localedef -i en_US -f UTF-8 C.UTF-8 && \
    yum install -y devtoolset-7

ENV LC_ALL="C.UTF-8"

RUN yum update -y && \
    yum install -y librdmacm && \
    yum install -y git wget && \
    yum install -y python36u python36u-libs python36u-devel python36u-pip && \
    yum -y clean all  && \
    rm -rf /var/cache

COPY ./libstdc++.so.6.0.28 /lib64/libstdc++.so.6.0.28
RUN rm -f /lib64/libstdc++.so.6 && \
    ln -s /lib64/libstdc++.so.6.0.28 /lib64/libstdc++.so.6


ARG popsdk=https://artifactory.sourcevertex.net/poplar-sdk-builds/poplar_sdk_centos/versioned/sdk-release-2.3/2.3.1/2021-11-03_16%3A08%3A58_89796d462d/centos_7_6_installer/poplar_sdk-centos_7_6-2.3.1%2B793-89796d462d.tar.gz

# USER scotty
WORKDIR /home/scotty/sdk
RUN wget -d --header 'X-JFrog-Art-Api:AKCp8jQd49DG1mvtUS8YA2urpzzaapkmSecr9HaFjqMbWXt6tJD4NWNbY4EkXDZVtLj6ttyor' \
    --header 'Authorization: Bearer' ${popsdk}

RUN python3 -m pip install --user --no-cache -U pip && \
    python3 -m pip install --user --no-cache virtualenv && \
    tar -zxvf /home/scotty/sdk/poplar_sdk-centos_7_6-2.3.1+793-89796d462d.tar.gz && \
    mv /home/scotty/sdk/poplar_sdk-centos_7_6-2.3.1+793-89796d462d/poplar-centos_7_6-2.3.0+1476-0537c534d3 /home/scotty/sdk/ && \
    rm -rf /home/scotty/sdk/poplar_sdk-centos_7_6-2.3.1+793-89796d462d.tar.gz /home/scotty/sdk/poplar_sdk-centos_7_6-2.3.1+793-89796d462d && \
    echo 'source /opt/rh/devtoolset-7/enable' >> ~/.bashrc && \
    echo "source /home/scotty/sdk/poplar-centos_7_6-2.3.0+1476-0537c534d3/enable.sh" >> ~/.bashrc && \
    echo "source /home/scotty/Desktop/ipu_tool_env/bin/activate" >> ~/.bashrc && \
    echo 'IPUOF_CONFIG_PATH=/etc/ipuof.conf.d/p.conf' >> ~/.bashrc && \
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