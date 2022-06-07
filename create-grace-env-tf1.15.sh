#!/usr/bin/env bash
set -e


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

export ENV_PREFIX=$PWD/env-tf1.15
conda env create --prefix $ENV_PREFIX -f environment.yml --force

conda activate $ENV_PREFIX
pip install tensorflow==1.15 # include GPU support
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
conda install -y protobuf

#ln -s $ENV_PREFIX/lib/libcublas.so.10 $ENV_PREFIX/lib/libcublas.so.10.0
#ln -s $ENV_PREFIX/lib/libcudart.so.10.1 $ENV_PREFIX/lib/libcudart.so.10.0
#ln -s $ENV_PREFIX/lib/libcufft.so.10 $ENV_PREFIX/lib/libcufft.so.10.0
#ln -s $ENV_PREFIX/lib/libcurand.so.10 $ENV_PREFIX/lib/libcurand.so.10.0
#ln -s $ENV_PREFIX/lib/libcusolver.so.10 $ENV_PREFIX/lib/libcusolver.so.10.0
#ln -s $ENV_PREFIX/lib/libcusparse.so.10 $ENV_PREFIX/lib/libcusparse.so.10.0

export HOROVOD_CUDA_HOME=$ENV_PREFIX
export HOROVOD_NCCL_HOME=$ENV_PREFIX
export HOROVOD_GPU_OPERATIONS=NCCL
export HOROVOD_WITH_PYTORCH=1
export HOROVOD_WITH_TENSORFLOW=1
pip install --no-cache-dir horovod==0.21.0

# install grace
pip install -e .

pip install -r requirements.txt
