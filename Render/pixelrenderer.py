from Render.baserenderer import BaseRenderer
from PIL import Image,ImageDraw
import io
class PixelRenderer(BaseRenderer):
    def __init__(self):
        super().__init__()
        self.columns = []
    
    def verify_format(self, sample, selected_columns, name):
        for col in selected_columns:
            value = self.get_values(sample, col)

            if not isinstance(value, dict):
                print(f"Column '{col}' in dataset '{name}' is not a dict as expected for pixel data.")
                return False

            if "bytes" not in value:
                print(f"Column '{col}' in dataset '{name}' is missing the 'bytes' key.")
                return False

            if not isinstance(value["bytes"], (bytes, bytearray, list)):
                print(f"'bytes' in column '{col}' of dataset '{name}' must be of type bytes, bytearray, or list. Found {type(value['bytes']).__name__}.")
                return False

            self.columns.append(col)

        return True
    def split_horizontal_image(self,img):
        w, h = img.size
        mid = w // 2
        
        left = img.crop((0, 0, mid, h))
        right = img.crop((mid, 0, w, h))
        
        stacked = Image.new("RGB", (mid, h * 2))
        stacked.paste(left, (0, 0))
        stacked.paste(right, (0, h))
        return stacked
    def render(self,sample):
        col = self.columns[0]
        value = self.get_values(sample, col)
        pixel_bytes = value.get("bytes")
        byte_data = bytes(pixel_bytes)
        img = Image.open(io.BytesIO(byte_data)).convert("RGB")
        if img.width > img.height * 1.5:  # Very horizontal? Split it
            img = self.split_horizontal_image(img)
        img.show()
        return img