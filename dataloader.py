
import random
from datasets import load_dataset
import matplotlib.pyplot as plt
from Render.imagetextrenderer import ImageTextRenderer
from Render.pixelrenderer import PixelRenderer
from Render.textrenderer import TextRenderer

class DataLoader():
    def __init__(self,dataset_config,render):
        self.dataset_config = dataset_config
        self.render = render
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
            try:
                ds = load_dataset(path)["train"]
            except Exception as e:
                print(f"Failed to load dataset '{name}' from path '{path}': {e}")
                continue
            
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
                continue

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
        if len(self.datasets)>0:
            self.check_weight(total_weights,unspecified_indices)
            self.weights = [d["weight"] for d in self.datasets]
        else:
            print("no datasets found")
    

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
        if self.render:
            if renderer.verify_format(sample, columns, name) is not True:
                return False
        return True
        

    def getdata(self,debug):
       # function for sampling the data
       if len(self.datasets)<1:
           print("no dataset available")
           return False
       dataset_entry = random.choices(self.datasets, weights=self.weights, k=1)[0]
       ds = dataset_entry["dataset"]
       sample = ds[random.randint(0, len(ds) - 1)]
       
       columns = dataset_entry["columns"]
       renderer = dataset_entry["renderer"]
       selected_data = {
        col: renderer.get_values(sample, col)
        for col in columns}
       
       if self.render:
           obs = renderer.render(sample)
       else:
           obs = None
       return {
        "json": selected_data,
        "observation": obs
        }
    
   