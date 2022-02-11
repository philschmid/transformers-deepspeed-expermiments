# Coming from https://github.com/microsoft/DeepSpeed/issues/1332#issuecomment-1020617723
   
import os
import torch
import deepspeed
import transformers
import time
from deepspeed import module_inject
from transformers import pipeline
from transformers.models.t5.modeling_t5 import T5Block


# Get local gpu rank from torch.distributed/deepspeed launcher
local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

print(
    "***************** Creating model in RANK ({0}) with WORLD_SIZE = {1} *****************"
    .format(local_rank,
            world_size))

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

pipe = pipeline("text2text-generation", model="google/t5-v1_1-small")

# The inpjection_policy shows two things:
#   1. which layer module we need to add Tensor-Parallelism
#   2. the name of several linear layers: a) attention_output (both encoder and decoder), 
#       and b) transformer output

assert pipe.model.config.num_heads % world_size == 0, f"{pipe.model.config.num_heads} % {world_size} = {pipe.model.config.num_heads % world_size} \nTo run the model parallel across the GPUs, the attention_heads require to be divisible by the world_size!This is because the attention computation is partitioned evenly among the parallel GPUs."

pipe.model = deepspeed.init_inference(
    pipe.model,
    mp_size=world_size,
    dtype=torch.float,
    injection_policy={T5Block: ('SelfAttention.o', 'EncDecAttention.o', 'DenseReluDense.wo')}
)

pipe.device = torch.device(f'cuda:{local_rank}')

print(
    f'({local_rank}) after deepspeed: allocated:{round(torch.cuda.memory_allocated()/1000/1000,2)}MB, cached:{round(torch.cuda.memory_allocated()/1000/1000,4)}MB'
)
print("check GPU memory usage")
time.sleep(10)


output = pipe("Is this review positive or negative? Review: this is the best cast iron skillet you will ever buy")

if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    print(output)

