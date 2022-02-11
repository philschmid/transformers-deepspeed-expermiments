import random
import zmq
import time
import os
import deepspeed
import torch
from transformers import pipeline

port = "5555"
# Socket to talk to server
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect(f"tcp://localhost:{port}")
socket.subscribe(b"")

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

generator = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B',
                     device=local_rank)



generator.model = deepspeed.init_inference(generator.model,
                                           mp_size=world_size,
                                           dtype=torch.float,
                                           replace_method='auto',
					   replace_with_kernel_inject=True)




if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    pair_port = "5556"
    pair_socket = context.socket(zmq.PAIR)
    pair_socket.connect(f"tcp://localhost:{pair_port}")
    pair_socket.send(b"READY")


def predict(data):
    # pop inputs for pipeline
    inputs = data.pop("inputs", data)
    parameters = data.pop("parameters", None)

    # pass inputs with all kwargs in data
    if parameters is not None:
        prediction = generator(inputs, **parameters)
    else:
        prediction = generator(inputs)
    return prediction

# Process 5 updates
while True:
    body = socket.recv_pyobj()
    pred = predict(body)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        pair_socket.send_pyobj(pred)

