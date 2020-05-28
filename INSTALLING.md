# Installing GRACE

To use GRACE with Horovod 0.18.2, it is required to apply the [patch](horovod_0.18.2.patch) provided. This patch was created
from [this](https://github.com/horovod/horovod/tree/v0.18.2) version of Horovod. After applying the patch it is required
to rebuild and reinstall Horovod. To avoid all this hassle, we provide [these](https://hub.docker.com/r/sandslab/grace) Docker images with the patch applied and
GRACE pre-installed.

To use GRACE via Docker, just pull the latest image with:
```shell script
docker pull sandslab/grace
```

The following assumes you want to install it manually.
GRACE comes as an installable pip package. It is recommended to install it with:
```shell script
git clone https://github.com/sands-lab/grace.git && cd grace
pip3 install --user -e .
```
This way, Python will use the local directory as a package, which allows experimentation with the codebase.

Here is another dirty way to apply the patch (to avoid rebuilding and reinstalling Horvod).
Since only python files are affected, you can apply the patch at the local installation of Horovod. The location of
your installation can be found with this command:
```shell script
python3 -c 'import horovod; print(horovod.__path__)'
```
