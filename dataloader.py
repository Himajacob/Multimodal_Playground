import os
import io
import textwrap
import random
from PIL import Image, ImageDraw
from datasets import load_dataset
import matplotlib.pyplot as plt
from Render.imagetextrenderer import ImageTextRenderer
from Render.pixelrenderer import PixelRenderer
from Render.textrenderer import TextRenderer

class DataLoader():
    def __init__(self,dataset_config):
        self.dataset_config = dataset_config
        self._load_dataset()
        
    def _load_dataset(self):
        for config in self.dataset_config:
            path = config["path"]
            name = config["name"]
            dtype = config["type"]
            weights = config.get("weight",None)
            
            ds = load_dataset(path)["train"]
            sample = ds[0]
            columns = config.get("selected_columns", list(sample.keys()))

            if dtype == "image_text":
                renderer = ImageTextRenderer()
            elif dtype == "text":
                renderer = TextRenderer()
            elif dtype == "pixel":
                renderer = PixelRenderer()
            else:
                print(f"unknown dtype {dtype}")

            if self.verify_columns(sample,columns,renderer,name) is not True:
                print(f"invalid column in {name}")
                continue
    
    def verify_columns(self,sample,columns,renderer,name):
        for col in columns:
            if renderer.get_values(sample, col) is None:
                print(f"invalid column in {name}")
                return False
        if renderer.verify_format(sample,columns,name) is not True:
            return False
        return True
        


    def getdata(self):
        # function for sampling the data
        sample = self.ds[random.randint(0, len(self.ds) - 1)]  
        columns = self.selected_columns or list(sample.keys())
        selected_data = {
                col:self.get_values(sample, col) for col in columns
            }

        #obs = self.getrendering(image=sample["image"], caption=sample["sentences"]["raw"])  #coco datser

        # obs = self.getrendering(pixel_values=sample["pixel_values"])         #pixel dataset

        obs = self.getrendering(text=sample["text"])
        output = {
                "json": selected_data,
                "observation": obs
            }
        return output
    
    def getrendering(self,**kwargs):

        # code for coco dataset

        # image = kwargs.get("image")
        # caption = kwargs.get("caption", "")
        # image = image.convert("RGB")
        # height = image.height + 30
        # render_image = Image.new("RGB", (image.width, height), color=(255, 255, 255))
        # render_image.paste(image, (0, 0))
        # draw = ImageDraw.Draw(render_image)
        # draw.text((10, image.height + 5), caption, fill=(0, 0, 0)) 

        #code for pixe dataset

        # pixel_bytes = kwargs.get("pixel_values", {}).get("bytes")
        # byte_data = bytes(pixel_bytes)
        # render_image = Image.open(io.BytesIO(byte_data)).convert("RGB")
        
        text = kwargs.get("text", "")
        lines = textwrap.wrap(text, width=100)
        font_size = 16
        line_height = font_size + 6
        padding = 10
        height = line_height * len(lines) + 2*padding
        render_image = Image.new("RGB", (1000, height), color="white")
        draw = ImageDraw.Draw(render_image)

        h = 10
        for line in lines:
            draw.text((padding, h), line, fill="black")
            h += line_height


        # plt.imshow(render_image)
        # plt.axis("off")
        # plt.show()
        return render_image