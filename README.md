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

[pre-install-deepspeed-ops](https://www.deepspeed.ai/tutorials/advanced-install/#pre-install-deepspeed-ops)

```bash
pip install torch==1.10.2+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers==4.15.0
# DS_BUILD_OPS=1 pip install git+https://github.com/microsoft/DeepSpeed.git

# deepspeed 
sudo apt install libaio-dev
pip install triton==1.0.0
git clone https://github.com/microsoft/DeepSpeed
cd DeepSpeed
mkdir deepspeed/ops/transformer_inference
DS_BUILD_TRANSFORMER_INFERENCE=1 DS_BUILD_UTILS=1 pip install -e . --global-option="build_ext" --global-option="-j8" --no-cache -v --disable-pip-version-check 2>&1 | tee build.log
cd ..
```
/home/ubuntu/transformers-deepspeed/DeepSpeed/build/lib.linux-x86_64-3.8/deepspeed/ops/transformer_inference/transformer_inference_op.cpython-38-x86_64-linux-gnu.so
2. test script
```bash
python3 -m deepspeed.launcher.runner --num_gpus 2 run_infernence_gpt-neo.py
```

## Commands 

1. check gpus
```bash
nvidia-smi
```
2. check deepspeed
```bash
#ds_report
python3 -m deepspeed.env_report
```

## Working examples

gpt-neo
```bash
python3 -m deepspeed.launcher.runner --num_gpus 2 run_infernence_gpt-neo.py
```

work with ranks
```bash
python3 -m deepspeed.launcher.runner --num_gpus 2 run_infernence_gpt-neo.py
```

memory allocation test
```bash
python3 -m deepspeed.launcher.runner --num_gpus 2 memory_allocation_test.py
```

## WIP: HTTP Example

```bash
python3 -m deepspeed.launcher.runner --num_gpus 2 api.py
```

```bash
curl --request POST \
   --url http://localhost:8500/ \
   --header 'Content-Type: application/json' \
   --data '{
 "inputs":"Deepspeed is"
 }'
```

### IO Example with: `0MQ` and `Starlette

```bash
cd io
python3 server.py
```

```bash
curl --request POST \
  --url http://localhost:8500/ \
  --header 'Content-Type: application/json' \
  --data '{
	"inputs": "Hugging Face can do",
	"paramters": {
		"min_length": 75,
		"max_length": 250
	}
}'
```

## MP Example

```bash
python3 -m deepspeed.launcher.runner --num_gpus 4 run_mp_example.py
python3 -m deepspeed.launcher.runner --num_gpus 4 t5_mp.py
```


## Resources

* [DS Documentation Inference](https://deepspeed.readthedocs.io/en/latest/inference-init.html)
* [Supported Model Architectures](https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/module_inject/replace_policy.py)
* [Tutorial: Getting Started with DeepSpeed for Inferencing Transformer based Models](https://www.deepspeed.ai/tutorials/inference-tutorial/)
* [DeepSpeed Launch function for `deepspeed`](https://github.com/microsoft/DeepSpeed/blob/dac9056e13ded1f931171c5f2461761c89fe2595/deepspeed/launcher/launch.py#L90)
* [Interessting Issue for GPT-J](https://github.com/microsoft/DeepSpeed/issues/1332) 
* [T5 Example](https://github.com/microsoft/DeepSpeed/pull/1711/files) 
* [WIP Examples](https://github.com/microsoft/DeepSpeedExamples/tree/inference/General-TP-examples/inference/huggingface)
* [WIP Examples PR](https://github.com/microsoft/DeepSpeedExamples/pull/144)


