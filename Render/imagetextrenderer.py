from Render.baserenderer import BaseRenderer
import numpy as np
import torch
from PIL import Image
class ImageTextRenderer(BaseRenderer):
    def __init__(self):
        super().__init__()
        self.img_columns = []
        self.txt_columns = []

    def is_image(self, value):
        return (
            isinstance(value, Image.Image) or
            isinstance(value, np.ndarray) or
            (torch and isinstance(value, torch.Tensor)) 
        )
    
    def verify_format(self, sample, selected_columns, name):
        for col in selected_columns:
            value = self.get_values(sample, col)

            if self.is_image(value):
                self.img_columns.append(col)

            elif isinstance(value, str):
                self.txt_columns.append(col)

            else:
                print(f"Column '{col}' in dataset '{name}' must be either image-like or text. Found: {type(value).__name__}")
                return False

        return True
    
    def render(self,selected_columns):
        pass