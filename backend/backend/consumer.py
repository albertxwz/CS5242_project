from channels.generic.websocket import AsyncWebsocketConsumer
import json
import os
import asyncio
from backend.image_preprocesser import image_to_base64
import os

def list_files(directory):
    image_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            image_list.append(os.path.join(root, file))
    
    return image_list

class ImageConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        print('connect start....')
        await self.accept()
        # await self.run_model_and_send_images()

    async def run_model_and_send_images(self):
        image_list = list_files('data/image')
        image_list.sort()
        cnt = 0
        for step in range(1, 41):  # 假设模型运行1000步
            # 模拟模型运行所需时间
            await asyncio.sleep(0.05)  # 使用asyncio.sleep来模拟异步运行
            if step % 20 == 0:
                image_data = image_to_base64(image_list[cnt])
                await self.send(text_data=json.dumps({
                    'image': image_data,
                    'step': step
                }))
                cnt += 1
            else:
                await self.send(text_data=json.dumps({
                    'step': step
                }))

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        message = json.loads(text_data)
        code = message['code']
        model = message['model']
        print(code, model)
        await self.run_model_and_send_images()
