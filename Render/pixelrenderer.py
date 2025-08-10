from Render.baserenderer import BaseRenderer
from PIL import Image,ImageDraw
import io
import base64
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
    
    def render(self, sample):
        print("[DEBUG] Starting render")
        col = self.columns[0]
        value = self.get_values(sample, col)
        if not value or "bytes" not in value:
            print("[ERROR] No 'bytes' key found.")
            return
        pixel_bytes = value["bytes"]
        if not isinstance(pixel_bytes, (bytes, bytearray)):
            print("[ERROR] 'bytes' field is not bytes type.")
            return
        print("[DEBUG] Byte length:", len(pixel_bytes))
        try:
            img = Image.open(io.BytesIO(pixel_bytes)).convert("L")  # grayscale
            print("[DEBUG] Image loaded:", img.size)
            img.show()
            return img
        except Exception as e:
            print("[ERROR] Failed to open image:", e)