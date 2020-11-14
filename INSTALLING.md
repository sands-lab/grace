# Installing GRACE

### Docker
To use GRACE with Horovod 0.18.2, it is required to apply the [patch](horovod_0.18.2.patch) provided. This patch was created
from [this](https://github.com/horovod/horovod/tree/v0.18.2) version of Horovod. After applying the patch it is required
to rebuild and reinstall Horovod. To avoid all this hassle, we provide [these](https://hub.docker.com/r/sandslab/grace) Docker images with the patch applied and
GRACE pre-installed.

To use GRACE via Docker, just pull the latest image with:
```shell script
docker pull sandslab/grace
```

### Conda
If you want to create a new conda environment for GRACE, please use this command:
 ```shell script
git clone https://github.com/sands-lab/grace.git && cd grace
bash ./create-grace-env.sh
```

This will create a conda env in `grace/env-tf1.14`

### Install GARCE 
The following assumes you want to install GARCE manually when you alreday have an enviroment installed with Horovod.
GRACE comes as an installable pip package. It is recommended to install it with:
```shell script
git clone https://github.com/sands-lab/grace.git && cd grace
pip install --user -e .
```
This way, Python will use the local directory as a package, which allows experimentation with the codebase.

Then copy the patch files to Horovod package (to avoid rebuilding and reinstalling Horovod).
```shell script
bash ./update_patch_files.sh
```
