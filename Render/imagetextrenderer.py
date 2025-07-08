from Render.baserenderer import BaseRenderer
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
import textwrap
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
    
    def render(self,sample):
        font = ImageFont.load_default()
        padding = 10
        line_height = 18

        image_value = self.get_values(sample, self.img_columns[0])
        image = image_value.convert("RGB") if hasattr(image_value, "convert") else Image.fromarray(image_value)


        all_text = []
        for col in self.txt_columns:
            value = self.get_values(sample, col)
            if value:
                wrapped = textwrap.wrap(value, width=100)
                all_text.extend(wrapped)


        caption_height = line_height * len(all_text) + 2 * padding
        total_height = image.height + caption_height

        render_image = Image.new("RGB", (image.width, total_height), color="white")
        render_image.paste(image, (0, 0))

        draw = ImageDraw.Draw(render_image)

        y = image.height + padding
        for line in all_text:
            draw.text((padding, y), line, fill="black", font=font)
            y += line_height

        render_image.show()
        return render_image