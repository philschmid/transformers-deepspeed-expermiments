from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
import time
import zmq
import uvicorn
import subprocess
import sys

port = "5555"
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind(f"tcp://*:{port}")

server_port = "5556"
pair_socket = context.socket(zmq.PAIR)
pair_socket.bind(f"tcp://*:{server_port}")

# start deepspeed server
# python3 -m deepspeed.launcher.runner --num_gpus 2 io/inference.py
subprocess.Popen([sys.executable,"-u", "-m", "deepspeed.launcher.runner", "--num_gpus", "2", "io/inference.py"])


client_conntected = False
while not client_conntected:
    print("Waiting for clients to connect")
    response = pair_socket.recv()
    if response.decode() == "READY":
        client_conntected = True
        print("clients connected")


async def test(request):
    start = time.time()
    body = await request.json()
    socket.send_pyobj(body)
    response = pair_socket.recv_pyobj()
    return JSONResponse({"result":response,"duration": f"{round(time.time() - start,4)*1000}ms"})


app = Starlette(
    debug=False,
    routes=[
        Route("/", test, methods=["POST"]),
    ],
)

uvicorn.run(app, host="0.0.0.0", port=8500, log_level="info", workers=1)
