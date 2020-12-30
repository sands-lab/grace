# GRACE - GRAdient ComprEssion for distributed deep learning

## Context
Powerful computer clusters are used nowadays to train complex deep neural networks (DNN) on large datasets. Distributed training workloads increasingly become communication bound. For this reason, many lossy compression techniques have been proposed to reduce the volume of transferred data, and consequently, accelerate training.

## What is GRACE?
GRACE is a GRAdient ComprEssion framework for distributed deep learning.
It provides a general framework and an API for implementing gradient compression methods.
GRACE includes the implementation of popular compression algorithms surveyed in [Compressed Communication for Distributed Deep Learning: Survey and Quantitative Evaluation](http://hdl.handle.net/10754/662495).
GRACE supports both TensorFlow and Pytorch, and offers easy integration with training scripts. This simplifies
the process of implemeting and comparing across compression methods.

## Install
The `grace_dl` package can be install locally on your nodes. For Horovod support, it is required to apply
the patch provided. To avoid this manual setup, a docker image is provided with GRACE already installed along with a patched version of Horovod 0.18.2. You can find more information [here](INSTALLING.md).

## Concepts
The three main components of this framework are the `Communicator`, `Compressor`, and `Memory` abstract classes.
- `Communicator` implementations define the communication method used by GRACE.
- `Compressor` implementations provide the `compress` and `decompress` methods.
- `Memory` implementations provide the `update` and `compensate` methods.

## Supported frameworks
Currently the supported frameworks are:
- Horovod 0.18.2 ([TensorFlow 1.14 & 2.0](grace_dl/tensorflow))
- Horovod 0.18.2 ([PyTorch 1.3.0](grace_dl/torch))
- [PyTorch distributed](grace_dl/dist).

## Usage
To use GRACE:
- **As a User**: you need to change a few lines of code on your training script. Refer [here](TRAINING.md) for examples.
- **As an Implementer of new Compressors**: you need to implement a `Compressor` class, and optionally a `Memory` or even a `Communicator` class.
Refer [here](IMPLEMENTING.md) for more information.
- **For Benchmarking methods**: you can use the [training scripts](https://github.com/sands-lab/grace-benchmarks) and compare with [our data](https://github.com/sands-lab/grace-data).

## Running
Running a training script is no different with GRACE. See the Training guide for more information.

## Guides
- [Installing GRACE](INSTALLING.md)
- [Implementing GRACE components](IMPLEMENTING.md)
- [Training with GRACE](TRAINING.md)

## Citation

If you find this useful, please cite our work as:

Compressed Communication for Distributed Deep Learning: Survey and Quantitative Evaluation.<br>
H. Xu, C.-Y. Ho, A. M. Abdelmoniem, A. Dutta, E. H. Bergou, K. Karatsenidis, M. Canini, P. Kalnis.<br>
KAUST technical report, Apr 2020.

Here's a BibTeX blurb:

```
@TECHREPORT{grace,
    author = {Xu, Hang and Ho, Chen-Yu and Abdelmoniem, Ahmed M. and Dutta, Aritra and Bergou, El Houcine and Karatsenidis, Konstantinos and Canini, Marco and Kalnis, Panos},
    title = {{Compressed Communication for Distributed Deep Learning: Survey and Quantitative Evaluation}},
    institution = {KAUST},
    note = {http://hdl.handle.net/10754/662495},
    year = {2020},
    month = {Apr} 
}
```

## Publications

* [Compressed Communication for Distributed Deep Learning: Survey and Quantitative Evaluation](http://hdl.handle.net/10754/662495).<br>
  H. Xu, C.-Y. Ho, A. M. Abdelmoniem, A. Dutta, E. H. Bergou, K. Karatsenidis, M. Canini, P. Kalnis.<br>
  KAUST technical report, Apr 2020.

## Mailing lists

There is no official mailing list set up yet. Feel free to reach out by email to any of the authors indicated in the [technical report](http://hdl.handle.net/10754/662495).
