_server=yyx01041752@login1.tbcsqa.alibaba.org
_files=$(ssh "${_server}" -t 'ls -al ~/aa/' | awk -F' ' '{print $9}' | grep 'm*')
# server=alibook

for f in ${_files}
do
    _tmp="~/aa/${f}";
    echo "${_tmp}";
    scp ${_server}:${_tmp} $(pwd)
done