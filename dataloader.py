import os
from huggingface_hub import login
import webdataset as wds
from huggingface_hub import HfFileSystem, get_token, hf_hub_url
import random
from PIL import Image, ImageDraw
from datasets import load_dataset

class DataLoader():
    def __init__(self):
        #login(token=os.environ["HUGGINGFACE_TOKEN"])
        self._load_dataset()
        
    def _load_dataset(self):
        self.isweb = False
        if(self.isweb ):
            splits = {'train': '**/train/*.tar', 'test': '**/test/*.tar'}
            fs = HfFileSystem()
            files = [fs.resolve_path(path) for path in fs.glob("hf://datasets/clip-benchmark/wds_mscoco_captions/" + splits["train"])]
            urls = [hf_hub_url(file.repo_id, file.path_in_repo, repo_type="dataset") for file in files]
            urls = f"pipe: curl -s -L -H 'Authorization:Bearer {get_token()}' {'::'.join(urls)}"
            self.ds = wds.WebDataset(urls).decode("pil")
        else:
            self.ds = load_dataset("asuglia/small_coco")["train"]

    def getdata(self):
        if(self.isweb):
            skip=random.randint(1,1000)
            for i, sample in enumerate(self.ds):
                if "jpg" in sample and "txt" in sample:
                    if i >= skip:
                        obs = self.getrendering(sample["jpg"],sample["txt"])
                        output = {
                            "image": sample["jpg"],
                            "text": sample["txt"],
                            "observation": obs
                        }
                        return output
            self._load_dataset()  
            return self.getdata()
        else:
            sample = self.ds[random.randint(0, len(self.ds) - 1)]  
            obs = self.getrendering(sample["image"], sample["sentences"]["raw"])  
            output = {
                "image": sample["image"],
                "text": sample["sentences"]["raw"],
                "observation": obs
            }
            return output
    
    def getrendering(self,image,caption):
        image = image.convert("RGB")
        height = image.height + 30
        render_image = Image.new("RGB", (image.width, height), color=(255, 255, 255))
        render_image.paste(image, (0, 0))
        draw = ImageDraw.Draw(render_image)
        draw.text((10, image.height + 5), caption, fill=(0, 0, 0)) 
        return render_image