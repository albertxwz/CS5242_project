import os
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DConditionModel
from accelerate import Accelerator

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
            if t % self.save_interval == 0:
                pred_image.save(os.path.join(self.output_dir, f'_{t:04d}.png'))
            t += 1
        self.pipeline.numpy_to_pil(pred_images)[0].save(os.path.join(self.output_dir, f'_1000.png'))

model_paths = [
    "/home/x/xie77777/codes/markup2im/models/math/scheduled_sampling/model_e100_lr0.0001.pt.100", # latex
    "/home/x/xie77777/codes/markup2im/models/tables/scheduled_sampling/model_e100_lr0.0001.pt.100", # table
    "/home/x/xie77777/codes/markup2im/models/molecules/scheduled_sampling/model_e100_lr0.0001.pt.100", # chem
    "/home/x/xie77777/codes/markup2im/models/music/scheduled_sampling/model_e100_lr0.0001.pt.100", # music
]

# TODO: complete this part
class MOECompiler(BaseCompiler):
    def __init__(self) -> None:
        self.compilers = [
            BaseCompiler(p, "/home/x/xie77777/codes/markup2im/backend/data/dummy")
                for p in model_paths
        ]
        self.selector = None # classifier

    def compile(self, text) -> None:
        model_index = self.selector(text)
        self.compilers[model_index].compile(text)

# test
if __name__ == "__main__":
    torch.manual_seed(1234)
    compiler = BaseCompiler("/home/x/xie77777/codes/markup2im/models/all_2/model_e100_lr0.0001.pt.100", "/home/x/xie77777/codes/markup2im/backend/data/dummy")
    # compiler.compile("(0,\\frac{a}{2}\\tau(0)+\\frac{b}{2}),")
    # compiler.compile("( 0, \\frac { a } { 2 } \\tau ( 0 ) + \\frac { b } { 2 } ),")
    # compiler.compile("\\hat { N } _ { 3 } = \\sum \\sp f _ { j = 1 } a _ { j } \sp { \\dagger } a _ { j } \\, .")
    compiler.compile("d s ^ { 2 } = e ^ { - 2 k r _ { c } | \\phi | } \\eta _ { \\mu \\nu } d x ^ { \mu } d x ^ { \\nu } - r _ { c } ^ { 2 } d \\phi ^ { 2 },")