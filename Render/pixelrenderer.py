from baserenderer import BaseRenderer
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
    def render(self,selected_columns):
        pass