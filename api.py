from fastapi import FastAPI, Request, Response, Query
from transformers import pipeline
import deepspeed, torch, os, uvicorn
from pydantic import BaseModel
import time

class Payload(BaseModel):
    inputs: str

app = FastAPI()

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))


generator = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B',
                     device=local_rank)



generator.model = deepspeed.init_inference(generator.model,
                                           mp_size=world_size,
                                           dtype=torch.float,
                                           replace_method='auto',
					                                 replace_with_kernel_inject=True)

@app.post("/gen")
def generate(payload:Payload):
    print(payload)
    return generator(payload.inputs, max_length=100)

# if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
print(f'initiating server on rank: {local_rank}')
uvicorn.run(
    app, 
    host="0.0.0.0", port=8500+local_rank, 
    log_level="info", 
    workers=1
)

# here we lose the second half of the model  