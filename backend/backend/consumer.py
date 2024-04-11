from channels.generic.websocket import AsyncWebsocketConsumer
import json
from backend.image_preprocesser import image_to_base64
import os
import threading
import glob
import torch
from backend.model_loader import BaseCompiler, MOECompiler
import time

def producer(code, model):
    '''
    code: The code user input
    model: int, 0: classification; 1: n to n
    '''
    torch.manual_seed(1234)
    if model == 1:
        print("N to N")
        # compiler = BaseCompiler("/home/x/xie77777/codes/markup2im/models/models/all_2/model_e100_lr0.0001.pt.100", "/home/x/xie77777/codes/markup2im/backend/data/dummy")
        # compiler.compile(code)
    if model == 0:
        print("Classification")
        # compiler = MOECompiler()
        # compiler.compile(code)
    time.sleep(80)


def delete_all(path):
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif']
    for extension in image_extensions:
        for image_path in glob.glob(os.path.join(path, extension)):
            os.remove(image_path)
            print(f'Deleted {image_path}')


from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from multiprocessing import Process

class NewFileHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()
        self.new_files = []

    def on_created(self, event):
        if not event.is_directory:
            self.new_files.append(event.src_path)
            print(f'New file created: {event.src_path}')


class ImageConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        print('connect start....')
        await self.accept()

    async def run_model_and_send_images(self, code, model):
        generation = Process(target=producer,args=(code, model,))
        path_to_watch = "data/dummy"
        event_handler = NewFileHandler()
        observer = Observer()
        observer.schedule(event_handler, path_to_watch, recursive=False)
        observer.start()
        generation.start()
        handler_len = 0
        while True:
            if len(event_handler.new_files) != handler_len:
                handler_len = len(event_handler.new_files)
                if handler_len > 1:
                    image_name = event_handler.new_files[-2]
                    image_data = image_to_base64(image_name)
                    await self.send(text_data=json.dumps({
                        'image': image_data,
                        'step': 20 * handler_len,
                        'status': 200
                    }))
            if not generation.is_alive():
                await self.send(text_data=json.dumps({
                     'step': 20 * handler_len,
                     'status': 400
                 }))
                break
        delete_all(path_to_watch)
        

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        message = json.loads(text_data)
        code = message['code']
        model = message['model']
        print(code)
        await self.run_model_and_send_images(code, model)
