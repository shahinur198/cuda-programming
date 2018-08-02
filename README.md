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
