FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    build-essential \
    python3-dev \
    cmake \
    ninja-build \
    libopenblas-dev \
    nano \
    git xterm xauth openssh-server tmux wget mate-desktop-environment-core

RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV MAX_JOBS=1

COPY . .

# For faster build, use more jobs.
# RUN git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine"
# RUN git clone --recursive "https://github.com/shwoo93/MinkowskiEngine.git"
# RUN cd MinkowskiEngine; python setup.py install --force_cuda --blas=openblas
RUN cd MinkowskiEngine; export CXX=c++; export CUDA_HOME=/usr/local/cuda; python setup.py install --blas=openblas --force_cuda
# Copy the repo

RUN pip install -r requirements.txt
# RUN uv pip install setuptools ninja numpy
# # Build & install
# RUN cd MinkowskiEngine && \
#  python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas