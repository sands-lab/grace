#!/bin/bash
PREFIX=${1:-"env-tf1.15"}
ENV_ROOT=$PWD
CONDA_TYPE=`type -t conda`
if [ ${CONDA_TYPE}"" == 'function' ]; then
  : # nothing to do
elif [ ${CONDA_TYPE}"" == 'file' ]; then
  CONDA_HOME_PATH=$(dirname "$(dirname "`command -v conda`")")
  . ${CONDA_HOME_PATH}/etc/profile.d/conda.sh
else
  echo "conda not installed or not in path"
  exit 1
fi

conda activate ${ENV_ROOT}/${PREFIX}
HOROVOD_PATH=`python -c "import horovod; print(horovod.__path__[0])"`

echo "will overwrite compression files in ${HOROVOD_PATH}"
read -p "Continue (y/N)? " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    [[ "$0" = "$BASH_SOURCE" ]] && exit 1 || return 1 # handle exits from shell or function but don't exit interactive shell
fi


cp -v patch_files/horovod/tensorflow/__init__.py ${HOROVOD_PATH}/tensorflow/
cp -v patch_files/horovod/tensorflow/keras/__init__.py ${HOROVOD_PATH}/tensorflow/keras/
cp -v patch_files/horovod/_keras/__init__.py ${HOROVOD_PATH}/_keras/
cp -v patch_files/horovod/torch/__init__.py ${HOROVOD_PATH}/torch/
cp -v patch_files/horovod/torch/mpi_ops.py ${HOROVOD_PATH}/torch/

cp -v patch_files/horovod/keras/__init__.py ${HOROVOD_PATH}/keras/
cp -v patch_files/horovod/torch/optimizer.py ${HOROVOD_PATH}/torch/