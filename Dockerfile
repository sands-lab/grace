FROM horovod/horovod:0.18.2-tf2.0.0-torch1.3.0-mxnet1.5.0-py3.6-gpu
#FROM horovod/horovod:0.18.2-tf2.0.0-torch1.3.0-mxnet1.5.0-py3.6-cpu
#FROM horovod/horovod:0.18.2-tf2.0.0-torch1.3.0-mxnet1.5.0-py2.7-gpu
#FROM horovod/horovod:0.18.2-tf2.0.0-torch1.3.0-mxnet1.5.0-py2.7-cpu
#FROM <ohter horovod:0.18.2-based image>

COPY . /grace
RUN patch -d /usr/local/lib/python${PYTHON_VERSION}/dist-packages -p0 < /grace/horovod_0.18.2.patch
RUN pip install -e /grace
RUN rm -rf /examples && mv /grace/examples /
