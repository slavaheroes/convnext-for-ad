# Installation of Minkowski Engine for Sparse Convolution


### Install openblas

```bash
sudo apt install libopenblas-dev
```


### Clone Minkowski Engine with Depthwise Convolutions

```bash
git clone https://github.com/shwoo93/MinkowskiEngine.git
```

Change `use_ninja` in `setup.py` to `False`and define environment variables: 

```bash
cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},

export CUDA_HOME=/usr/local/cuda
export MAX_JOBS=1
export LD\_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH
```

Install the Minkowski Engine:
```bash
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
```

## Trouble shooting

For PyTorch==1.13.1+cu117: 

> I modified src/spmm.cu file as shown in https://github.com/NVIDIA/MinkowskiEngine/pull/481 