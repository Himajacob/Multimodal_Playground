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
        self.datasets = []
        self._load_dataset()

        
    def _load_dataset(self):
        total_weights = 0
        unspecified_indices = []
        for config in self.dataset_config:
            path = config["path"]
            name = config["dataset"]
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
                continue
            
            self.datasets.append({
                "dataset": ds,
                "type": dtype,
                "name": name,
                "columns": columns,
                "renderer": renderer,
                "weight": weights
            })

            if weights is not None:
                total_weights += weights
            else:
                unspecified_indices.append(len(self.datasets) -1)
        self.check_weight(total_weights,unspecified_indices)
    

    def check_weight(self,total_weights,unspecified_indices):
        if (total_weights != 1 and not unspecified_indices) or \
       (len(unspecified_indices) == len(self.datasets)) or \
       (total_weights > 1):
            w = 1/len(self.datasets)
            for data in self.datasets:
                data["weight"] = w
            print("assigning equal weights")
        elif total_weights > 0 and unspecified_indices:
            remaining_weight = max(0.0, 1.0 - total_weights)
            equal_unspecified = remaining_weight / len(unspecified_indices) if unspecified_indices else 0
            for idx in unspecified_indices:
                self.datasets[idx]["weight"] = equal_unspecified
    
    
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
       weights = [d["weight"] for d in self.datasets]
       dataset_entry = random.choices(self.datasets, weights=weights, k=1)[0]
       ds = dataset_entry["dataset"]
       sample = ds[random.randint(0, len(ds) - 1)]
       
       columns = dataset_entry["columns"]
       renderer = dataset_entry["renderer"]
       selected_data = {
        col: renderer.get_values(sample, col)
        for col in columns}
       
       #obs = renderer.render(sample, selected_data)
       obs = None
       return {
        "json": selected_data,
        "observation": obs
        }
    
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