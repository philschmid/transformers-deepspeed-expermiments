import os
import deepspeed
import torch
from transformers import pipeline

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))


generator = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B',
                     device=local_rank)



generator.model = deepspeed.init_inference(generator.model,
                                           mp_size=world_size,
                                           dtype=torch.float,
                                           replace_method='auto',
					   replace_with_kernel_inject=True)


print(f"torch.distributed.is_initialized: {torch.distributed.is_initialized()}")
print(f"torch.distributed.get_rank: {torch.distributed.get_rank()}")
print(f'local_rank: {local_rank}')

if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    print(f'local_rank: {local_rank}')
