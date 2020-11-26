#!/bin/bash

# Configuration
# -gencode=arch=compute_70,code=sm_70 \
# CUDA_GENCODE="\
# -gencode=arch=compute_61,code=sm_61 \
# -gencode=arch=compute_60,code=sm_60 \
# -gencode=arch=compute_52,code=sm_52 \
# -gencode=arch=compute_50,code=sm_50"

CUDA_GENCODE="-arch=sm_30 \
 -gencode=arch=compute_30,code=sm_30 \
 -gencode=arch=compute_50,code=sm_50 \
 -gencode=arch=compute_52,code=sm_52 \
 -gencode=arch=compute_60,code=sm_60 \
 -gencode=arch=compute_61,code=sm_61 \
 -gencode=arch=compute_62,code=sm_62 \
 -gencode=arch=compute_70,code=sm_70 \
 -gencode=arch=compute_70,code=compute_70"

# CUDA_GENCODE="-arch=sm_30 \
#  -gencode=arch=compute_30,code=sm_30 \
#  -gencode=arch=compute_50,code=sm_50 \
#  -gencode=arch=compute_52,code=sm_52 \
#  -gencode=arch=compute_60,code=sm_60 \
#  -gencode=arch=compute_61,code=sm_61 \
#  -gencode=arch=compute_62,code=sm_62 "

# CUDA_GENCODE="-arch=sm_30 \
#  -gencode=arch=compute_30,code=sm_30 \
#  -gencode=arch=compute_50,code=sm_50 \
#  -gencode=arch=compute_52,code=sm_52 \
#  -gencode=arch=compute_60,code=sm_60 \
#  -gencode=arch=compute_61,code=sm_61"


cd src
nvcc -I/usr/local/cuda/include --expt-extended-lambda -O3 -c -o bn.o bn.cu -x cu -Xcompiler -fPIC -std=c++11 ${CUDA_GENCODE}
cd ..
