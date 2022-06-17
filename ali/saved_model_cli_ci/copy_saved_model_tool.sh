#!/bin/bash

wget -d --header 'X-JFrog-Art-Api:AKCp8jQd49DG1mvtUS8YA2urpzzaapkmSecr9HaFjqMbWXt6tJD4NWNbY4EkXDZVtLj6ttyor' \
    --header 'Authorization: Bearer' $1

poplar_tar=$(echo $1 | sed 's/.*poplar_sdk\([^ ]*\).*/\1/')
poplar_tar="poplar_sdk${poplar_tar/\%2B/+}"
tar -zxvf ${poplar_tar}

wheel=$(find ${poplar_tar/\.tar.gz/''} -name "ipu_tensorflow_addons-1*.whl" )
# cp ${wheel} /root/Desktop/

/home/scotty/Desktop/ipu_tool_env/bin/python3 -m pip install --no-cache ${wheel}

rm -rf poplar_tar*
