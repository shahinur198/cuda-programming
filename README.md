https://en.wikipedia.org/wiki/CUDA

https://developer.nvidia.com/cuda-gpus

https://docs.nvidia.com/deeplearning/sdk/cudnn-support-matrix/index.html

https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-nano/

nvidia driver install and check
```
nvidia-settings
```
https://developer.nvidia.com/cuda-80-ga2-download-archive
https://www.youtube.com/watch?v=8Xy1Uqq9Hbk
# cuda 9.0 installation
https://developer.nvidia.com/cuda-90-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604&target_type=deblocal
```
 used method in this post: cuda-sample-build-error. I think you just need substitute the correct gpu driver version in UBUNTU_PKG_NAME = "nvidia-375", which you can find at

    .../NVIDIA_CUDA-8.0_Samples/3_Imaging/cudaDecodeGL/findgllib.mk file

```
path set

```
nano .bashrc
export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```
nstallation Instructions:
```
    sudo dpkg -i cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64.deb
    sudo apt-key add /var/cuda-repo-<version>/7fa2af80.pub
    sudo apt-key add /var/cuda-repo-9-0-local/7fa2af80.pub
    sudo apt-get update
    sudo apt-get install cuda

```
or
```
sudo apt install nvidia-cuda-toolkit
```
path set

```
nano .bashrc
export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```
# set environment variables
```
export PATH=${PATH}:/usr/local/cuda-9.0/bin
export CUDA_HOME=${CUDA_HOME}:/usr/local/cuda:/usr/local/cuda-9.0
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-9.0/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64

for permanent
go to 
sudo nano /etc/environment
and add

PATH=${PATH}:/usr/local/cuda-9.0/bin
CUDA_HOME=${CUDA_HOME}:/usr/local/cuda:/usr/local/cuda-9.0
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-9.0/lib64
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64

```
# how to set environment variables
```
nano .profile
nano .bashrc
sudo nano /etc/environment
```
# cuda verification
```
$ cd /usr/local/cuda-9.0/samples
$ sudo make

$ cd /usr/local/cuda/samples/bin/x86_64/linux/release
$ ./deviceQuery
```
nvcc --version
# cudnn install 
https://developer.nvidia.com/rdp/cudnn-download
https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html
```
    Navigate to your <cudnnpath> directory containing cuDNN Debian file.
    Install the runtime library, for example:

    sudo dpkg -i libcudnn7_7.0.3.11-1+cuda9.0_amd64.deb

    Install the developer library, for example:

    sudo dpkg -i libcudnn7-dev_7.0.3.11-1+cuda9.0_amd64.deb

    Install the code samples and the cuDNN Library User Guide, for example:

    sudo dpkg -i libcudnn7-doc_7.0.3.11-1+cuda9.0_amd64.deb

```
# Verifying cuDNN

To verify that cuDNN is installed and is running properly, compile the mnistCUDNN sample located in the /usr/src/cudnn_samples_v7 directory in the debian file.
```
    Copy the cuDNN sample to a writable path.

    $ cp -r /usr/src/cudnn_samples_v7/ $HOME

    Go to the writable path.

    $ cd  $HOME/cudnn_samples_v7/mnistCUDNN

    Compile the mnistCUDNN sample.

    $ make clean && make

    Run the mnistCUDNN sample.

    $ ./mnistCUDNN

    If cuDNN is properly installed and running on your Linux system, you will see a message similar to the following:

    Test passed!

```
https://medium.com/@zhanwenchen/install-cuda-and-cudnn-for-tensorflow-gpu-on-ubuntu-79306e4ac04e

# gpu view
```
pip3 install gpustat
gpustat -cp

```
# cuda-programming


# Installing MXNet on Ubuntu
http://mxnet.incubator.apache.org/install/index.html
```
pip3 install mxnet-cu90
pip3 install mxnet-cu90mkl
```
Test:
```
import mxnet as mx
a = mx.nd.ones((2, 3), mx.gpu())
b = a * 2 + 1
b.asnumpy()
```
MXNet issu

https://github.com/apache/incubator-mxnet/issues/8671
```
Supported variants:

    mxnet on Windows, Linux, and Mac OSX, with CPU-only without MKL-DNN support.
    mxnet-cu75 on Linux, supports CUDA-7.5.
    mxnet-cu80 on Windows and Linux, supports CUDA-8.0.
    mxnet-cu90 on Windows and Linux, supports CUDA-9.0.
    mxnet-cu91 on Windows and Linux, supports CUDA-9.1.
    mxnet-cu92 on Windows and Linux, supports CUDA-9.2.
    mxnet-mkl on Windows, Linux, and Mac OSX, with CPU-only MKLDNN support.
    mxnet-cu75mkl on Linux, supports CUDA-7.5 and MKLDNN support.
    mxnet-cu80mkl on Windows and Linux, supports CUDA-8.0 and MKLDNN support.
    mxnet-cu90mkl on Windows and Linux, supports CUDA-9.0 and MKLDNN support.
    mxnet-cu91mkl on Windows and Linux, supports CUDA-9.1 and MKLDNN support.
    mxnet-cu92mkl on Windows and Linux, supports CUDA-9.2 and MKLDNN support.

```
MXNet Tutorial:

http://mxnet.incubator.apache.org/tutorials/index.html

MXNet with OpenCv:

https://www.pyimagesearch.com/2017/11/13/how-to-install-mxnet-for-deep-learning/

# Installing minpy on Ubuntu
https://pypi.org/project/minpy/
```
pip3 install minpy
```
Test:
```
import minpy.numpy as np
import minpy.numpy.random as random
from minpy.context import cpu, gpu
import time

n = 100

with cpu():
    x_cpu = random.rand(1024, 1024) - 0.5
    y_cpu = random.rand(1024, 1024) - 0.5

    # dry run
    for i in range(10):
        z_cpu = np.dot(x_cpu, y_cpu)
    z_cpu.asnumpy()

    # real run
    t0 = time.time()
    for i in range(n):
        z_cpu = np.dot(x_cpu, y_cpu)
    z_cpu.asnumpy()
    t1 = time.time()

with gpu(0):
    x_gpu0 = random.rand(1024, 1024) - 0.5
    y_gpu0 = random.rand(1024, 1024) - 0.5

    # dry run
    for i in range(10):
        z_gpu0 = np.dot(x_gpu0, y_gpu0)
    z_gpu0.asnumpy()

    # real run
    t2 = time.time()
    for i in range(n):
        z_gpu0 = np.dot(x_gpu0, y_gpu0)
    z_gpu0.asnumpy()
    t3 = time.time()

print("run on cpu: %.6f s/iter" % ((t1 - t0) / n))
print("run on gpu: %.6f s/iter" % ((t3 - t2) / n))
```
output:
```
run on cpu: 0.011208 s/iter
run on gpu: 0.001077 s/iter

```
# Tensorrt install-guide
https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html

# warning guide
```
/home/nybsys/anaconda3/lib/python3.6/site-packages/h5py/__init__.py:34: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.   from ._conv import register_converters as _register_converters
```
```
pip3 install --upgrade --pre h5py
```

https://www.youtube.com/watch?v=LJvBhmuPNg8
https://www.youtube.com/watch?v=-e4BvZz6H8Q
