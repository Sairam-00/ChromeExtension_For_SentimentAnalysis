import asyncio
import websockets
from ml import YouMLProject

async def echo(websocket):
    async for message in websocket:
        
        await websocket.send(YouMLProject.TextSentiment(message))

async def main():
    async with websockets.serve(echo, "localhost", 8080):
        await asyncio.Future()  # run forever

asyncio.run(main())

#  env\Scripts\activate