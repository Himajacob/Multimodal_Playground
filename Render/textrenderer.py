from baserenderer import BaseRenderer
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
        
    def render(self):
        pass