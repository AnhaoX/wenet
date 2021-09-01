if which conda &>/dev/null; then                                                                                               
  conda_root=`conda info --base`
  . $conda_root/etc/profile.d/conda.sh && conda deactivate && conda activate wenet
fi

export PATH=$PWD:$PWD/tools:$PATH

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8
export PYTHONPATH=../../:$PYTHONPATH
export LC_ALL=C

# The NCCL_SOCKET_IFNAME variable specifies which IP interface to use for nccl
# communication. More details can be found in
# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
#export NCCL_SOCKET_IFNAME=ens4f1
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=COLL
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
#export NCCL_ASYNC_ERROR_HANDLING=1

# Enable this only for debugging, or it will somehow block DDP initialization
# when running on multiple hosts.
#export CUDA_LAUNCH_BLOCKING=1
