from torchvision.transforms.functional import pad
import torch
from torch import nn

class PadToSize(nn.Module):
    '''Pad the given image from the last row and column until its size reaches the goal.
    Image must be a tensor.

    Args:
        output_size: size to be padded
    
    '''
    def __init__(self, output_size):
        super(PadToSize, self).__init__()
        self.output_size = output_size

    def forward(self, img):
        if not isinstance(img, torch.Tensor):
            raise TypeError("image is not a tensor")
        
        _, H, W = img.shape
        if H > self.output_size[0] or W > self.output_size[1]:
            raise ValueError(f"{self.output_size} can not cover ({H}, {W})")

        return pad(img, (0, 0, self.output_size[1] - W, self.output_size[0] - H), 0, "edge")

# test
# if __name__ == "__main__":
#     img = torch.randn((3, 4, 5))
#     img[:, 3] = 0
#     img[:, :, 4] = 0
#     transform = PadToSize((5, 6))
#     print(img)
#     print(transform(img))