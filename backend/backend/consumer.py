from channels.generic.websocket import AsyncWebsocketConsumer
import json
import asyncio
from backend.image_preprocesser import image_to_base64

class ImageConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        print('connect start....')
        await self.accept()
        # await self.run_model_and_send_images()

    async def run_model_and_send_images(self):
        for step in range(1, 1001):  # 假设模型运行1000步
            # 模拟模型运行所需时间
            await asyncio.sleep(0.05)  # 使用asyncio.sleep来模拟异步运行
            if step == 100:
                image_data = image_to_base64('data/image/nus_logo.png')
                await self.send(text_data=json.dumps({
                    'image': image_data,
                    'step': step
                }))
            elif step == 200:  
                image_data = image_to_base64('data/image/ntu_logo.webp')
                await self.send(text_data=json.dumps({
                    'image': image_data,
                    'step': step
                }))
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
