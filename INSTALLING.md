# Installing GRACE


### Conda
If you want to create a new conda environment for GRACE, please use this command:
 ```shell script
git clone https://github.com/sands-lab/grace.git && cd grace
bash ./create-grace-env-tf1.15.sh
```

This will create a conda env in `grace/env-tf1.15`

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
