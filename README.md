# GRACE - GRAdient ComprEssion for distributed deep learning

## Context
Powerful computer clusters are used nowadays to train complex deep neural networks (DNN) on large datasets. Distributed training workloads increasingly become communication bound. For this reason, many lossy compression techniques have been proposed to reduce the volume of transferred data, and consequently, accelerate training.

## What is GRACE?
GRACE is a GRAdient ComprEssion framework for distributed deep learning.
It provides a general framework and an API for implementing gradient compression methods.
GRACE includes the implementation of popular compression algorithms surveyed in [GRACE: A Compressed Communication Framework for Distributed Machine Learning](https://sands.kaust.edu.sa/papers/grace.icdcs21.pdf).
GRACE supports both TensorFlow and Pytorch, and offers easy integration with training scripts. This simplifies
the process of implemeting and comparing across compression methods.
![grace_bert_bench](https://user-images.githubusercontent.com/52653060/174394060-8f616ed0-0fca-424d-ab9f-2648423d11c2.jpg)


## Install
The `grace_dl` package can be install locally on your nodes. For Horovod support, it is required to apply
the patch files provided. You can find more information [here](INSTALLING.md).

## Concepts
The three main components of this framework are the `Communicator`, `Compressor`, and `Memory` abstract classes.
- `Communicator` implementations define the communication method used by GRACE.
- `Compressor` implementations provide the `compress` and `decompress` methods.
- `Memory` implementations provide the `update` and `compensate` methods.

## Supported frameworks
Currently the supported frameworks are:
- Horovod 0.21.0 ([TensorFlow 1.15](grace_dl/tensorflow))
- Horovod 0.21.0 ([PyTorch 1.7.0](grace_dl/torch))
- DistributedDataParallel (DDP) ([PyTorch >= 1.4.0](grace_dl/dist))

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

Hang Xu, Chen-Yu Ho, Ahmed M Abdelmoniem, Aritra Dutta, El Houcine Bergou, Konstantinos Karatsenidis, Marco Canini, and Panos Kalnis. GRACE: A Compressed Communication Framework for Distributed Machine Learning. In Proc. of ICDCS, 2021.

Here's a BibTeX blurb:

```
@inproceedings{xu2021grace,
  title={GRACE: A compressed communication framework for distributed machine learning},
  author={Xu, Hang and Ho, Chen-Yu and Abdelmoniem, Ahmed M. and Dutta, Aritra and Bergou, EH and Karatsenidis, Konstantinos and Canini, Marco and Kalnis, Panos},
  booktitle={Proc. of 41st IEEE Int. Conf. Distributed Computing Systems (ICDCS)},
  year={2021}
}
```

## Publications

* [GRACE: A Compressed Communication Framework for Distributed Machine Learning](https://sands.kaust.edu.sa/papers/grace.icdcs21.pdf).<br>
  H. Xu, C.-Y. Ho, A. M. Abdelmoniem, A. Dutta, E. H. Bergou, K. Karatsenidis, M. Canini, P. Kalnis.<br>
  In Proc. of ICDCS, 2021.

## Mailing lists

There is no official mailing list set up yet. Feel free to reach out by email to any of the authors indicated in the [paper](https://sands.kaust.edu.sa/papers/grace.icdcs21.pdf).

## License
See [LICENSE](LICENSE) for the details.
