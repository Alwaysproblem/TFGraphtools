server=alibook
files="$(echo "$(ssh "${server}" -t 'ls -al aa/')" | awk -F' ' '{print $9}')"
# server=yyx01041752@login1.tbcsqa.alibaba.org

for f in ${files}
do
  if [[ "${f}" != "."  &&  "${f}" != ".." ]]; then
    _tmp="/home/yyx01041752/aa/${f}";
    echo "${_tmp}";
    scp ${server}:${_tmp} .
  fi
done