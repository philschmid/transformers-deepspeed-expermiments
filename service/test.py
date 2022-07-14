import re
import time
from flask import Flask                                                         
import threading
from queue import Empty, Queue
import os
from flask import request


local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

print( f"***************** Creating model in RANK ({local_rank}) with WORLD_SIZE = {world_size} *****************")

#### run command
# torchrun --standalone --nnodes=1 --nproc_per_node=2 test.py
#### hey example
# hey -n 100 -m POST -H 'Content-Type: application/json' -d '{	"inputs": "is multiprocessing a valid"}' http://localhost:5000
# curl --request POST http://localhost:5000 --header 'Content-Type: application/json'  -d '{"inputs": "is multiprocessing a valid"}'
  
data = 'foo'
host_name = "0.0.0.0"
port = 5000
app = Flask(__name__)
global global_q
global_q = Queue(10)


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
    time.sleep(10)
    res = payload.upper()
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