
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

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

print( f"***************** Creating model in RANK ({local_rank}) with WORLD_SIZE = {world_size} *****************")

#### run command
# torchrun --standalone --nnodes=1 --nproc_per_node=2 test.py
#### hey example
# hey http://localhost:23336


data = 'foo'
host_name = "0.0.0.0"
port = 23336
app = Flask(__name__)
global global_q
global_q = Queue(10)

global generator
generator = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B',
                     device=local_rank)

generator.model = deepspeed.init_inference(generator.model,
                                           mp_size=world_size,
                                           dtype=torch.float,
                                           replace_method='auto',
					   replace_with_kernel_inject=True)

print("model loaded")

@app.route('/', methods = ['GET', 'POST', 'DELETE'])
def main():
    req = request.json["inputs"]
    print(req)
    request_queue = Queue()
    global_q.put((request_queue,req))
    pred = request_queue.get()
    return str(pred)


def background_predict(data):
    q, payload= data
    print("start")
    print(payload)
    print(generator)
    res = generator(payload)
    if local_rank == 0:
        q.put(res)
        

if __name__ == "__main__":
    if local_rank == 0:
      threading.Thread(target=lambda: app.run(host=host_name, port=port, debug=True, use_reloader=False)).start()
    while True: 
        data = global_q.get()
        background_predict(data)
        global_q.task_done()
        print("processed")