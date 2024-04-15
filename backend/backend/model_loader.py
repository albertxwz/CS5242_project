import os
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DConditionModel
from accelerate import Accelerator
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_pipeline(image_decoder, model_path):
    state_dict = torch.load(model_path, map_location='cpu')
    state_dict_new = {}
    for k in state_dict:
        k_out = k.replace('module.', '')
        state_dict_new[k_out] = state_dict[k]
    image_decoder.load_state_dict(state_dict_new)
    
    accelerator = Accelerator(mixed_precision='no')
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, tensor_format="pt")
    pipeline = DDPMPipeline(unet=image_decoder, scheduler=noise_scheduler)
    return pipeline

def encode_text(text_encoder, input_ids, attention_mask, no_grad=True):
    if no_grad:
        with torch.no_grad():
            outputs = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state 
            if attention_mask is not None:
                last_hidden_state = attention_mask.unsqueeze(-1) * last_hidden_state
    else:
        outputs = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state 
        if attention_mask is not None:
            last_hidden_state = attention_mask.unsqueeze(-1) * last_hidden_state
    return last_hidden_state

def create_image_decoder(image_size, color_channels, cross_attention_dim):
    image_decoder = UNet2DConditionModel(
        sample_size=image_size,
        in_channels=color_channels,
        out_channels=color_channels,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "CrossAttnDecoderPositionEncoderPositionDownBlock2D", 
            "CrossAttnDecoderPositionEncoderPositionDownBlock2D",
            "CrossAttnDecoderPositionEncoderPositionDownBlock2D",
            "CrossAttnDecoderPositionEncoderPositionDownBlock2D",
        ), 
        up_block_types=(
            "CrossAttnDecoderPositionEncoderPositionUpBlock2D",
            "CrossAttnDecoderPositionEncoderPositionUpBlock2D",
            "CrossAttnDecoderPositionEncoderPositionUpBlock2D",
            "CrossAttnDecoderPositionEncoderPositionUpBlock2D",
            "UpBlock2D",
            "UpBlock2D" 
          ),
          cross_attention_dim=cross_attention_dim,
          mid_block_type='UNetMidBlock2DCrossAttnDecoderPositionEncoderPosition'
    )
    return image_decoder

class BaseCompiler:
    def __init__(self,
                 model_path: str,
                 output_dir: str,
                 encoder_type: str = "EleutherAI/gpt-neo-125M",
                 image_size: tuple = (64, 320),
                 color_channels: int = 1,
                 save_interval: int = 20,
                ) -> None:
        self.output_dir = output_dir
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_type)
        self.text_encoder = AutoModel.from_pretrained(encoder_type).cuda()
        self.hidden_states = encode_text(self.text_encoder, torch.zeros(1, 1).long().cuda(), None)
        image_decoder = create_image_decoder(image_size, color_channels, self.hidden_states.shape[-1])
        image_decoder = image_decoder.cuda()
        # print(f"[Debug] {image_size}, {color_channels}, {encoder_type}")
        self.pipeline = load_pipeline(image_decoder, model_path)

        self.eos_id = self.tokenizer.encode(self.tokenizer.eos_token)[0]

        self.save_interval = save_interval

    def compile(self, text: str):
        example = self.tokenizer(text, truncation=True, max_length=1024)
        input_ids = torch.LongTensor(example['input_ids'] + [self.eos_id,]).cuda()
        mask = torch.LongTensor(example['attention_mask'] + [1,]).cuda()
        input_ids.unsqueeze_(0)
        mask.unsqueeze_(0)
        encoder_hidden_states = encode_text(self.text_encoder, input_ids, mask)
        swap_step = -1
        t = 0
        for _, pred_images in self.pipeline.run_clean(
            batch_size = input_ids.shape[0],
            generator=torch.manual_seed(0),
            encoder_hidden_states = encoder_hidden_states,
            attention_mask=mask,
            swap_step=swap_step,
            ):
            pred_image = self.pipeline.numpy_to_pil(pred_images)[0]
            if self.save_interval > 0 and t % self.save_interval == 0:
                pred_image.save(os.path.join(self.output_dir, f'_{t:04d}.png'))
            t += 1
        if self.save_interval > -1:
            self.pipeline.numpy_to_pil(pred_images)[0].save(os.path.join(self.output_dir, f'_1000.png'))

model_paths = [
    "../models/math/model_e100_lr0.0001.pt.100", # latex
    "../models/tables/model_e100_lr0.0001.pt.100", # table
    "../models/music/model_e100_lr0.0001.pt.100", # music
    "../models/molecules/model_e100_lr0.0001.pt.100", # chem
]

types = [
    'EleutherAI/gpt-neo-125M',
    'EleutherAI/gpt-neo-125M',
    'EleutherAI/gpt-neo-125M',
    'DeepChem/ChemBERTa-77M-MLM',
]

channels = [1, 1, 1, 3]

image_sizes = [
    (64, 320),
    (64, 64),
    (192, 448),
    (128, 128),
]

class TextMLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(TextMLP, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(200 * embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x) # [batch_size, seq_length, embedding_dim]
        embedded = embedded.view(embedded.shape[0], -1) # Flatten [batch_size, seq_length*embedding_dim]
        out = self.fc1(embedded)
        out = self.relu(out)
        out = self.fc2(out)
        return out

        
class MOECompiler(BaseCompiler):
    def __init__(self, save_interval = 20) -> None:
        self.compilers = [
            BaseCompiler(model_path=model_paths[i], output_dir="data/dummy",
                         image_size=image_sizes[i], encoder_type=types[i], color_channels=channels[i],
                         save_interval=save_interval)
                for i in range(4)
        ]
        #added 
        self.selector = TextMLP(10001, 100, 128, 4)
        self.selector.load_state_dict(torch.load("backend/MOE.pth"))
        self.selector.eval()
    #added
    def get_type(self, text):
        model = self.selector
        result = 0 # set default as math

        with open('backend/tokenizer.json', 'r', encoding='utf-8') as f:
            data = f.read()
            tokenizer = tokenizer_from_json(data)

        sequence = tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=200, padding='post', truncating='post')

        input_tensor = torch.tensor(padded_sequence, dtype=torch.long)
        model.eval()

        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.argmax(output, dim=1)
            result = prediction.item()
        return result
        
    def compile(self, text) -> None:
        model_index = self.get_type(text)# modify
        self.compilers[model_index].compile(text)


# test time
if __name__ == "__main__":
    test_files = ["math", "table", "music", "chemistry"]
    texts = []
    for test_file in test_files:
        with open("backend/test/"+test_file+".txt", "r", encoding="utf-8") as f:
            content = f.read()
            texts.append(content)
    
    results = {}
    
    n2n_compiler = BaseCompiler("../models/all/model_e100_lr0.0001.pt.100", output_dir="", save_interval=-1)
    moe_compiler = MOECompiler(save_interval=-1)

    import pandas as pd
    import time

    data = {
        "original": [None] * 4,
        "N2N": [None] * 4,
        "MOE": [None] * 4,
    }
    
    for i in range(4):
        start = time.time()
        moe_compiler.compilers[i].compile(texts[i])
        data["original"][i] = time.time() - start

    for i in range(4):
        start = time.time()
        moe_compiler.compile(texts[i])
        data["MOE"][i] = time.time() - start
    
    for i in range(2):
        start = time.time()
        n2n_compiler.compile(texts[i])
        data["N2N"][i] = time.time() - start
    
    df = pd.DataFrame(data)
    df.index = test_files
    df.to_csv("backend/test/results.csv")
    # df.to_excel("backend/test/results.xlsx")