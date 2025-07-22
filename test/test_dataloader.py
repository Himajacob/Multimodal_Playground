import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PIL import Image, ImageDraw
import pytest
from dataloader import DataLoader
from Render.imagetextrenderer import ImageTextRenderer
from Render.textrenderer import TextRenderer
from Render.pixelrenderer import PixelRenderer

@pytest.fixture
def text_config():
    return [
        {
            "type": "text",
            "dataset": "HimaLevenSuprabha/fineweb-reduced",
            "path": "HimaLevenSuprabha/fineweb-reduced",
            "selected_columns": ["text"],
            "weight": 1.0
        }
    ]

def test_text_renderoff(text_config):
    loader = DataLoader(dataset_config=text_config, render=False)
    output = loader.getdata()

    assert "json" in output
    assert "observation" in output
    assert isinstance(output["json"], dict)
    
    for key, val in output["json"].items():
        assert isinstance(val, str)
        assert len(val.strip()) > 0

    assert output["observation"] is None

@pytest.fixture
def pixel_config():
    return [
        {
            "type": "pixel",
            "dataset": "HimaLevenSuprabha/small-rendered-bookcorpus",
            "path": "HimaLevenSuprabha/small-rendered-bookcorpus",
            "selected_columns": ["pixel_values"],
            "weight": 1.0
        }
    ]

def test_pixel_renderoff(pixel_config):
    loader = DataLoader(dataset_config=pixel_config, render=False)
    output = loader.getdata()

    assert "json" in output
    assert "observation" in output
    assert isinstance(output["json"], dict)
    assert "pixel_values" in output["json"]
    
    pixel_data = output["json"]["pixel_values"]
    assert isinstance(pixel_data, dict)
    assert "bytes" in pixel_data
    assert isinstance(pixel_data["bytes"], (bytes, bytearray, list))

    assert output["observation"] is None

@pytest.fixture
def image_text_config():
    return [
        {
            "type": "image_text",
            "dataset": "mscoco",
            "path": "asuglia/small_coco",
            "selected_columns": ["image", "sentences.raw"],
            "weight": 1.0
        }
    ]

def test_imgtext_renderoff(image_text_config):
    loader = DataLoader(dataset_config=image_text_config, render=False)
    output = loader.getdata()

    assert "json" in output
    assert "observation" in output
    assert isinstance(output["json"], dict)
    
    keys = output["json"].keys()
    has_image = any(k for k in keys if "image" in k or "img" in k)
    has_text = any(k for k in keys if "text" in k or "sentences" in k)
    assert has_image or has_text  
    assert output["observation"] is None

# --------Checks if proper json values are returned even if render is off-----------------------------------

def test_text_render_on(text_config):
    loader = DataLoader(dataset_config=text_config, render=True)
    output = loader.getdata()

    obs = output["observation"]
    assert isinstance(obs, Image.Image)
    assert obs.size[0] > 0 and obs.size[1] > 0

def test_pixel_render_on(pixel_config):
    loader = DataLoader(dataset_config=pixel_config, render=True)
    output = loader.getdata()

    obs = output["observation"]
    assert isinstance(obs, Image.Image)
    assert obs.size[0] > 0 and obs.size[1] > 0

def test_imgtext_render_on(image_text_config):
    loader = DataLoader(dataset_config=image_text_config, render=True)
    output = loader.getdata()

    obs = output["observation"]
    assert isinstance(obs, Image.Image)
    assert obs.size[0] > 0 and obs.size[1] > 0

#---------------------checks if render is returning an image------------------------------

def test_invalid_column_is_skipped(capfd):
    config_with_invalid_column = [
        {
            "type": "text",
            "dataset": "HimaLevenSuprabha/fineweb-reduced",
            "path": "HimaLevenSuprabha/fineweb-reduced",
            "selected_columns": ["does_not_exist"],  
            "weight": 1.0
        }
    ]

    loader = DataLoader(dataset_config=config_with_invalid_column, render=False)
    
    assert loader.datasets == []
    captured = capfd.readouterr()
    assert "invalid column in" in captured.out
#------------------checks if dataset has an invalid column----------------

def test_empty_dataset_prints_warning(capfd):
    bad_config = [
        {
            "type": "text",
            "dataset": "dataset",
            "path": "",  
            "selected_columns": ["text"],
            "weight": 1.0
        }
    ]

    try:
        loader = DataLoader(dataset_config=bad_config, render=False)
        result = loader.getdata()
        assert result is False
        captured = capfd.readouterr()
        assert "no datasets found" in captured.out or "no dataset available" in captured.out
    except Exception as e:
        captured = capfd.readouterr()
        assert "no datasets found" in captured.out

def test_empty_dataset_config(capfd):
    empty_config = []

    loader = DataLoader(dataset_config=empty_config, render=False)
    result = loader.getdata()

    assert result is False

    captured = capfd.readouterr()
    assert "no datasets found" in captured.out or "no dataset available" in captured.out
#------------checks what happens when datasets is empty-----------------------------
