# Spiking-Graph-Mamba
Spiking Graph-Mamba (SGM): an energy-efficient long-range graph learning framework with Spike-Event Surrogate Dynamics Network (SSDN).

This repository contains the implementation of:

- SGM: Spiking Graph-Mamba framework for energy-efficient long-range graph learning.
- SSDN: Spike-Event Surrogate Dynamics Network that converts continuous representations into spike events.

## Method

SGM combines a local topology branch and a global context branch. The local branch uses GatedGCN to capture neighborhood structure, while the global branch uses Graph-Mamba to model long-range dependencies over serialized node sequences. The outputs of both branches are passed through SSDN, converted into spike-event traces, and fused for downstream tasks.

SSDN follows the spike-event objective: instead of only fitting continuous membrane potentials, it preserves threshold-triggered spike traces. The Safety Margin mechanism further improves robustness by keeping active and silent states away from firing boundaries.

## Main Files

```text
main.py                         # Training entry point
configs/Mamba/                  # SGM experiment configs
graphgps/network/gps_model.py   # GPSModel wrapper
graphgps/layer/gps_layer.py     # SGM layer: GatedGCN + Graph-Mamba + spike modules
graphgps/layer/neuron.py        # SSDN, BPTT, and SLTT neuron modules
graphgps/layer/surrogate.py     # Ternary surrogate spike function
graphgps/train/custom_train.py  # Custom training loop
```

## Installation

The code was developed with Python 3.9.

```bash
conda create -n sgm --file requirements_conda.txt
conda activate sgm
```

## Quick Start

Run one of the SGM configs:

```bash
python main.py --cfg configs/Mamba/vocsuperpixels-EX.yaml
```

## Acknowledgements

This code builds on GraphGym, PyTorch Geometric, Mamba-SSM, and SpikingJelly.
