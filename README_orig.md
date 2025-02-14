# :sparkles: GOOD: A Graph Out-of-Distribution Benchmark :sparkles:

[license-url]: https://github.com/divelab/GOOD/blob/main/LICENSE
[license-image]:https://img.shields.io/badge/license-GPL3.0-green.svg
[contributing-image]:https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat
[contributing-url]:https://good.readthedocs.io/en/latest/contributing.html

[![Documentation Status](https://readthedocs.org/projects/good/badge/?version=latest)](https://good.readthedocs.io/en/latest/?badge=latest)
![Last Commit](https://img.shields.io/github/last-commit/divelab/DIG)
[![License][license-image]][license-url]
[![codecov](https://codecov.io/gh/divelab/GOOD/branch/main/graph/badge.svg?token=W41HSP0XCY)](https://codecov.io/gh/divelab/GOOD)
[![CircleCI](https://circleci.com/gh/divelab/GOOD/tree/main.svg?style=svg)](https://circleci.com/gh/divelab/GOOD/tree/main)
[![GOOD stars](https://img.shields.io/github/stars/divelab/GOOD?style=social)](https://github.com/divelab/GOOD)
[![Contributing][contributing-image]][contributing-url]

[**Documentation**](https://good.readthedocs.io) | [**NeurIPS 2022 Paper**](https://openreview.net/forum?id=8hHg-zs_p-h&referrer=%5Bthe%20profile%20of%20Shurui%20Gui%5D(%2Fprofile%3Fid%3D~Shurui_Gui1)) | [Preprint](https://arxiv.org/abs/2206.08452) | [GOOD version 0](https://github.com/divelab/GOOD/tree/GOODv0) | GOOD version 1
<!-- > We are actively building the document. -->

<!-- [**GOOD: A Graph Out-of-Distribution Benchmark.**](https://arxiv.org/abs/2206.08452) Shurui Gui, Xiner Li, Limei Wang, and Shuiwang Ji. -->

<!-- :fire:**New! The GOOD is now also parts of the software library [DIG](https://github.com/divelab/DIG)! If you wish to use the GOOD datasets with DIG features, you can directly use the [DIG](https://github.com/divelab/DIG) library!** -->

:tada: **We are glad to announce that our GOOD paper is accepted by NeurIPS 2022 Datasets and Benchmarks Track!**

For the original code used in the paper, please check branch [GOOD version 0](https://github.com/divelab/GOOD/tree/GOODv0). All new features, datasets and methods will be updated in this branch.

## Roadmap

### Tutorial
- [x] More flexible pipelines and dataloaders. Please refer to branch [dev](https://github.com/divelab/GOOD/tree/dev) for more details.
- [x] More detailed tutorial for adding new algorithms. Please refer to [Add a new algorithm](#add-a-new-algorithm). 
### Algorithms (plan to include)
- [ ] [Improving Out-of-Distribution Robustness via Selective Augmentation](https://arxiv.org/pdf/2201.00299.pdf)
- [x] [Learning Causally Invariant Representations for Out-of-Distribution Generalization on Graphs](https://arxiv.org/pdf/2202.05441.pdf) [NeurIPS 2022] [[GitHub](https://github.com/LFhase/CIGA)]
  - [x] Method reproduction by its authors. Check [the CIGA branch](https://github.com/divelab/GOOD/tree/CIGA) for your convenience. :smile: [**New!**]
  - [ ] Experiments: hyperparameter sweeping
- [x] [Interpretable and Generalizable Graph Learning via Stochastic Attention Mechanism](https://arxiv.org/abs/2201.12987) [ICML 2022] [[GitHub](https://github.com/Graph-COM/GSAT)]
  - [x] Method reproduction.
  - [ ] Experiments: hyperparameter sweeping
### New features
- [x] Automatic program launcher.
- [x] Automatic hyperparameter sweeping and config updating.
### Leaderboard
- [ ] A new leaderboard is under construction.

## Table of contents

* [Overview](#overview)
* [Why GOOD?](#why-good)
* [Installation](#installation)
* [Quick tutorial](#quick-tutorial)
* [Add a new algorithm](#add-a-new-algorithm)
* [Citing GOOD](#citing-good)
* [License](#license)
* [Contact](#contact)

## Overview

**GOOD** (Graph OOD) is a graph out-of-distribution (OOD) algorithm benchmarking library depending on PyTorch and PyG
to make develop and benchmark OOD algorithms easily.

Currently, GOOD contains 11 datasets with 17 domain selections. When combined with covariate, concept, and no shifts, we obtain 51 different splits.
We provide performance results on 10 commonly used baseline methods (ERM, IRM, VREx, GroupDRO, Coral, DANN, MixupForGraph, DIR, EERM,SRGNN) including 4 graph specific methods with 10 random runs.

The GOOD dataset summaries are shown in the following figure.

![Dataset](/../../blob/main/docs/source/imgs/Datasets.png)

## Why GOOD?

Whether you are an experienced researcher of graph out-of-distribution problems or a first-time learner of graph deep learning, 
here are several reasons to use GOOD as your Graph OOD research, study, and development toolkit.

* **Easy-to-use APIs:** GOOD provides simple APIs for loading OOD algorithms, graph neural networks, and datasets so that you can take only several lines of code to start.
* **Flexibility:** Full OOD split generalization code is provided for extensions and any new graph OOD dataset contributions.
OOD algorithm base class can be easily overwritten to create new OOD methods.
* **Easy-to-extend architecture:** In addition to playing as a package, GOOD is also an integrated and well-organized project ready to be further developed.
All algorithms, models, and datasets can be easily registered by `register` and automatically embedded into the designed pipeline like a breeze!
The only thing the user needs to do is write your own OOD algorithm class, your own model class, or your new dataset class.
Then you can compare your results with the leaderboard.
* **Easy comparisons with the leaderboard:** We provide insightful comparisons from multiple perspectives. Any research and studies can use
our leaderboard results for comparison. Note that this is a growing project, so we will include new OOD algorithms gradually.
Besides, if you hope to include your algorithms in the leaderboard, please contact us or contribute to this project. A big welcome!


## Installation 

### Conda dependencies

GOOD depends on [PyTorch (>=1.6.0)](https://pytorch.org/get-started/previous-versions/), [PyG (>=2.0)](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html), and
[RDKit (>=2020.09.5)](https://www.rdkit.org/docs/Install.html). For more details: [conda environment](/../../blob/main/environment.yml)

> Note that we currently test on PyTorch (==1.10.1), PyG (==2.0.3), RDKit (==2020.09.5); thus we strongly encourage to install these versions.
>
> Attention! Due to a known issue, please install PyG through Pip to avoid incompatibility.

### Pip

#### Installation for Project usages (recommended)

```shell
git clone https://github.com/divelab/GOOD.git && cd GOOD
pip install -e .
```

## Quick Tutorial

### Run an algorithm

It is a good beginning to make it work directly. Here, we provide the CLI `goodtg` (GOOD to go) to 
access the main function located at `GOOD.kernel.main:goodtg`.
Choosing a config file in `configs/GOOD_configs`, we can start a task:

```shell
goodtg --config_path GOOD_configs/GOODCMNIST/color/concept/DANN.yaml
```

### Hyperparameter sweeping

To perform automatic hyperparameter sweeping and job launching, you can use `goodtl` (GOOD to launch):

```shell
goodtl --sweep_root sweep_configs --launcher MultiLauncher --allow_datasets GOODMotif --allow_domains basis --allow_shifts covariate --allow_algs GSAT --allow_devices 0 1 2 3
```

* `--sweep_root` is a config fold located at `configs/sweep_configs`, where we provide a GSAT algorithm hyperparameter sweeping setting example (on GOODMotif dataset, basis domain, and covariate shift). 
  * Each hyperparameter searching range is specified by a list of values. [Example](/../../blob/GOODv1/configs/sweep_configs/GSAT/base.yaml)
  * These hyperparameter configs will be transformed to be CLI argument combinations.
  * Note that hyperparameters in inner config files will overwrite the outer ones.
* `--launcher` denotes the chosen job launcher. Available launchers:
  * `Launcher`: Dummy launcher, only print.
  * `SingleLauncher`: Sequential job launcher. Choose the first device in `--allow_devices`.
  * `MultiLauncher`: Multi-gpu job launcher. Launch on all gpus specified by `--allow_devices`.
* `--allow_XXX` denotes the job scale. Note that for each "allow" combination (e.g. GSAT GOODMotif basis covariate),
there should be a corresponding sweeping config: `GSAT/GOODMotif/basis/covaraite/base.yaml` in the fold specified
by `--sweep_root`.
* `--allow_devices` specifies the gpu devices used to launch jobs.

### Sweeping result collection and config update.

To harvest all fruits you have grown (collect all results you have run), please use `goodtl` with a special launcher `HarvestLauncher`:

```shell
goodtl --sweep_root sweep_configs --final_root final_configs --launcher HarvestLauncher --allow_datasets GOODMotif --allow_domains basis --allow_shifts covariate --allow_algs GSAT
```

* `--sweep_root`: We still need it to specify the experiments that can be harvested.
* `--final_root`: A config store place that will store the best config settings. 
We will update the best configurations (according to the sweeping) into the config files in it.

(Experimental function.)

The output numpy array:
<!-- * Rows: In-distribution train/In-distribution test/Out-of-distribution train/Out-of-distribution test/Out-of-distribution validation -->
* Rows: IID test (last epoch), OOD test (last epoch)/IID test/Out-of-distribution test/Out-of-distribution validation
* Columns: Mean/Std.

### Final runs

It is sometimes not practical to run 10 rounds for hyperparameter sweeping, especially when the searching space is huge.
Therefore, we can generally run hyperparameter sweeping for 2~3 rounds, then perform all rounds after selecting the best hyperparameters.
Now, remove the `--sweep_root`, set `--config_root` to your updated best config saving location, and set the `--allow_rounds`.

```shell
goodtl --config_root final_configs --launcher MultiLauncher --allow_datasets GOODMotif --allow_domains basis --allow_shifts covariate --allow_algs GSAT --allow_devices 0 1 2 3 --allow_rounds 1 2 3 4 5 6 7 8 9 10
```

Note that the results are valid only after 3+ rounds experiments in this benchmark.

### Final result collection

```shell
goodtl --config_root final_configs --launcher HarvestLauncher --allow_datasets GOODMotif --allow_domains basis --allow_shifts covariate --allow_algs GSAT --allow_rounds 1 2 3 4 5 6 7 8 9 10
```

(Experimental function.)

The output numpy array:
* Rows: In-distribution train/In-distribution test/Out-of-distribution train/Out-of-distribution test/Out-of-distribution validation
* Columns: Mean/Std.

You can customize your own launcher at `GOOD/kernel/launchers/`.

## Add a new algorithm

Please follow [this documentation](https://good.readthedocs.io/en/latest/custom.html#practical-steps-to-add-a-new-ood-algorithm) to add a new algorithm.

Any contributions are welcomed! Please refer to [contributing](http://localhost:63342/GOOD/docs/build/contributing.html) for adding your algorithm into GOOD.

## Test

### Dataset regeneration test

This test regenerates all datasets again and compares them with the datasets used in the original training process locates.
Test details can be found at [test_regenerate_datasets.py](/../../blob/main/test/test_reproduce_full/test_regenerate_datasets.py).
For a quick review, we provide a [full regeneration test report](https://drive.google.com/file/d/1jIShh3eBXAQ_oQCFL9AVU3OpUlVprsbo/view?usp=sharing).

### Sampled tests

In order to keep the validity of our code all the time, we link our project with circleci service and provide several 
sampled tests to go through (because of the limitation of computational resources in CI platforms).

## Leaderboard

The original leaderboard results are listed in the paper. And the validation of these results is described [here](/../../tree/GOODv0#reproducibility).

A new leaderboard is under constructed.

## Citing GOOD
If you find this repository helpful, please cite our [paper](https://arxiv.org/abs/2206.08452).
```
@inproceedings{
gui2022good,
title={{GOOD}: A Graph Out-of-Distribution Benchmark},
author={Shurui Gui and Xiner Li and Limei Wang and Shuiwang Ji},
booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
year={2022},
url={https://openreview.net/forum?id=8hHg-zs_p-h}
}
```

## License

The GOOD datasets are under [MIT license](https://drive.google.com/file/d/1xA-5q3YHXLGLz7xV2tT69a9dcVmiJmiV/view?usp=sharing).
The GOOD code are under [GPLv3 license](https://github.com/divelab/GOOD/blob/main/LICENSE).

## Discussion

Please submit [new issues](/../../issues/new) or start [a new discussion](/../../discussions/new) for any technical or other questions.

## Contact

Please feel free to contact [Shurui Gui](mailto:shurui.gui@tamu.edu), [Xiner Li](mailto:lxe@tamu.edu), or [Shuiwang Ji](mailto:sji@tamu.edu)!

