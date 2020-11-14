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


export OMPI_MCA_opal_cuda_support=true
export ENV_PREFIX=$PWD/env-tf1.14

export HOROVOD_CUDA_HOME=$ENV_PREFIX
export HOROVOD_CUDA_INCLUDE=$ENV_PREFIX/include
export HOROVOD_CUDA_LIB=$ENV_PREFIX/lib
export HOROVOD_NCCL_HOME=$ENV_PREFIX
export HOROVOD_GPU_OPERATIONS=MPI
export HOROVOD_WITH_PYTORCH=1
export HOROVOD_WITH_TENSORFLOW=1
export HOROVOD_WITHOUT_MXNET=1
export HOROVOD_WITHOUT_GLOO=1
export HOROVOD_WITH_MPI=1

conda env create --prefix $ENV_PREFIX -f environment.yml --force
conda activate $ENV_PREFIX
pip install --no-cache-dir horovod==0.18.2

# update horovod with grace patch files
cp -v patch_files/horovod/tensorflow/__init__.py $ENV_PREFIX/lib/python3.7/site-packages/horovod/tensorflow/
cp -v patch_files/horovod/tensorflow/keras/__init__.py $ENV_PREFIX/lib/python3.7/site-packages/horovod/tensorflow/keras/
cp -v patch_files/horovod/_keras/__init__.py $ENV_PREFIX/lib/python3.7/site-packages/horovod/_keras/
cp -v patch_files/horovod/torch/__init__.py $ENV_PREFIX/lib/python3.7/site-packages/horovod/torch/
cp -v patch_files/horovod/torch/mpi_ops.py $ENV_PREFIX/lib/python3.7/site-packages/horovod/torch/

# install grace
pip install --user -e .