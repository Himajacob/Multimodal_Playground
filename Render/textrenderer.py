from Render.baserenderer import BaseRenderer
import textwrap
from PIL import Image,ImageDraw
class TextRenderer(BaseRenderer):
    def __init__(self):
        super().__init__()
        self.columns = []
    
    def verify_format(self, sample, selected_columns,name):
        for col in selected_columns:
            value = self.get_values(sample, col)
            if not isinstance(value,str):
                print(f"[ERROR] Column '{col}' in dataset '{name}' is not of type str. Found type: {type(value).__name__}")
                return False
            self.columns.append(col)
        return True
        
    def render(self,sample):
        max_lines = 30
        font_size = 16
        line_height = font_size + 6
        padding = 10
        lines = []

        for col in self.columns:
            value = self.get_values(sample,col)
            if not value :
                continue
            wrapped_lines = textwrap.wrap(value,width =100)
            for line in wrapped_lines:
                lines.append(line)
                if len(lines)>= max_lines:
                    break
            if len(lines)>= max_lines:
                    break
        
        height = line_height * len(lines) + 2 * padding
        render_image = Image.new("RGB", (500, height), color="white")
        draw = ImageDraw.Draw(render_image)

        h = padding
        for line in lines:
            draw.text((padding, h), line, fill="black")
            h += line_height
        render_image.show()
        return render_image