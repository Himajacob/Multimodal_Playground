import os
from huggingface_hub import login
import webdataset as wds
from huggingface_hub import HfFileSystem, get_token, hf_hub_url
import random

class DataLoader():
    def __init__(self):
        login(token=os.environ["HUGGINGFACE_TOKEN"])
        self._load_dataset()
        
    def _load_dataset(self):
        splits = {'train': '**/train/*.tar', 'test': '**/test/*.tar'}
        fs = HfFileSystem()
        files = [fs.resolve_path(path) for path in fs.glob("hf://datasets/clip-benchmark/wds_mscoco_captions/" + splits["train"])]
        urls = [hf_hub_url(file.repo_id, file.path_in_repo, repo_type="dataset") for file in files]
        urls = f"pipe: curl -s -L -H 'Authorization:Bearer {get_token()}' {'::'.join(urls)}"
        self.ds = wds.WebDataset(urls).decode("pil")

    def getdata(self):
        skip=random.randint(1,1000)
        for i, sample in enumerate(self.ds):
            if "jpg" in sample and "txt" in sample:
                if i >= skip:
                    return sample
        self._load_dataset()  
        return self.getdata()

