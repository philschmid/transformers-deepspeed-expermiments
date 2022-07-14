
import re
import time
from flask import Flask                                                         
import threading
from queue import Empty, Queue
import os
from flask import request
import deepspeed
import torch
from transformers import pipeline
import logging
import zmq
import json

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

# Communication
port = "5555"
context = zmq.Context()
# pub
if local_rank == 0:
    pub = context.socket(zmq.PUB)
    pub.bind(f"tcp://*:{port}")
# sub
sub = context.socket(zmq.SUB)
sub.connect(f"tcp://localhost:{port}")
sub.subscribe(b"")

print( f"***************** Creating model in RANK ({local_rank}) with WORLD_SIZE = {world_size} *****************")

#### run command
# torchrun --standalone --nnodes=1 --nproc_per_node=2 flask_pub_sub.py
#### hey example
# hey -n 100 -c 25 -m POST -H 'Content-Type: application/json' -d '{	"inputs": "is multiprocessing a valid"}' http://localhost:5000
# curl --request POST http://localhost:5000 --header 'Content-Type: application/json'  -d '{"inputs": "is multiprocessing a valid"}'
  

data = 'foo'
host_name = "0.0.0.0"
port = 5000
app = Flask(__name__)

global generator
# generator = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B',device=local_rank)
generator = pipeline('text-generation', model='EleutherAI/gpt-neo-125M',device=local_rank)
print("model loaded")
generator.model = deepspeed.init_inference(generator.model,
                                           mp_size=world_size,
                                           dtype=torch.float,
                                           replace_method='auto',
                                           replace_with_kernel_inject=True)
print("deepspeed init done")

def background_predict(data):
    res = generator(payload)
    return res

@app.route('/', methods = ['GET', 'POST', 'DELETE'])
def main():
    try:
        req = request.json["inputs"]
        server = context.socket(zmq.REP)
        port = server.bind_to_random_port(f"tcp://127.0.0.1")
        pub.send_pyobj((port,str(port)))
        response = server.recv_pyobj()
        return json.dumps(response)
    except Exception as e:
        return json.dumps({"error":str(e)})


if __name__ == "__main__":
    if local_rank == 0:
        threading.Thread(target=lambda: app.run(host=host_name, port=port, debug=False, processes=1, use_reloader=False)).start()
    # warm up
    print(generator("i am a test"))
    while True: 
        data = sub.recv_pyobj()
        port, payload =data
        res = background_predict(data)
        if local_rank == 0:
            pair_sender = context.socket(zmq.REQ)
            pair_sender.connect(f"tcp://127.0.0.1:{port}")
            pair_sender.send_pyobj(res)
            pair_sender.disconnect(f"tcp://127.0.0.1:{port}")
