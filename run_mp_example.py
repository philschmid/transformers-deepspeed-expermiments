# Coming from https://github.com/microsoft/DeepSpeed/issues/1332#issuecomment-1020617723
   
import os
import torch
import deepspeed
import transformers

from deepspeed import module_inject
from transformers import pipeline
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoBlock as gpt2_transformer

# Get local gpu rank from torch.distributed/deepspeed launcher
local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

print(
    "***************** Creating model in RANK ({0}) with WORLD_SIZE = {1} *****************"
    .format(local_rank,
            world_size))

# Coming from https://github.com/microsoft/DeepSpeed/issues/1332#issuecomment-1020617723
   
import os
import torch
import deepspeed
import transformers

from deepspeed import module_inject
from transformers import pipeline
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoBlock as gpt2_transformer

# Get local gpu rank from torch.distributed/deepspeed launcher
local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

print(
    "***************** Creating model in RANK ({0}) with WORLD_SIZE = {1} *****************"
    .format(local_rank,
            world_size))

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


assert AutoConfig.from_pretrained("EleutherAI/gpt-j-6B").n_head % world_size == 0, "To run the model parallel across the GPUs, the attention_heads require to be divisible by the world_size!This is because the attention computation is partitioned evenly among the parallel GPUs."

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
# model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")

inp_tokens = tokenizer("DeepSpeed is", return_tensors="pt")

model = deepspeed.init_inference(
            model,
            mp_size=world_size,
            dtype=torch.float,
            replace_method='auto',
            replace_with_kernel_inject=True)
                                 
for token in inp_tokens:
    if torch.is_tensor(inp_tokens[token]):
        inp_tokens[token] = inp_tokens[token].to(f'cuda:{local_rank}')
        
model.cuda().to(f'cuda:{local_rank}')


print(
    f'({local_rank}) after deepspeed: allocated:{round(torch.cuda.memory_allocated()/1000/1000,2)}MB, cached:{round(torch.cuda.memory_allocated()/1000/1000,4)}MB'
)

string = tokenizer.batch_decode(model.generate(**inp_tokens,min_length=50,))[0]
print(string)


