# transformers-deepspeed

This repository includes tests for using [DeepSpeed](https://www.deepspeed.ai/) with [transformers](https://huggingface.co/docs/transformers/v4.16.2/en/main_classes/deepspeed#deepspeed-integration)


## Setup

0. Prepare machine following [Stas Guide](https://github.com/bigscience-workshop/Megatron-DeepSpeed/blob/main/.github/workflows/ci.md). Install Cuda11.3
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.3.1/local_installers/cuda-repo-ubuntu2004-11-3-local_11.3.1-465.19.01-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-3-local_11.3.1-465.19.01-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu2004-11-3-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda-11.3
```

1. Install `deepspeed` and other dependencies

```bash
pip install transformers
pip install git+https://github.com/microsoft/DeepSpeed.git
pip install torch==1.10.2+cu113 -f https://download.pytorch.org/whl/torch_stable.html

```

2. test script
```bash
python -m deepspeed.launcher.runner --num_gpus 2 run_infernence_gpt-neo.py
```

## Commands 

1. check gpus
```bash
nvidia-smi
```
2. check deepspeed
```bash
#ds_report
python -m deepspeed.env_report
```

## Resources

* [DS Documentation Inference](https://deepspeed.readthedocs.io/en/latest/inference-init.html)
* [Supported Model Architectures](https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/module_inject/replace_policy.py)
* [Tutorial: Getting Started with DeepSpeed for Inferencing Transformer based Models](https://www.deepspeed.ai/tutorials/inference-tutorial/)
* [DeepSpeed Launch function for `deepspeed`](https://github.com/microsoft/DeepSpeed/blob/dac9056e13ded1f931171c5f2461761c89fe2595/deepspeed/launcher/launch.py#L90)
