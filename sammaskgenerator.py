from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2

class SAMMaskGenerator():
    def __init__(self):
        sam_checkpoint = "SAM/sam_vit_b_01ec64.pth" 
        model_type = "vit_b"
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.predictor = SamPredictor(self.sam)
    
    def getMaskFromClick(self,image,point= None,label=1,multimask_output=True,debug=False):

        if point is None and debug is True:
            point = self.get_point(image)
        self.predictor.set_image(image)
        input_point = np.array([point])
        input_label = np.array([label])

        masks, scores, _ = self.predictor.predict(
            point_coords = input_point,
            point_labels = input_label,
            multimask_output = multimask_output 
        )

        mask = masks[np.argmax(scores)]
        return mask.astype(np.uint8)
    
    def get_point(self,image):
        print("Click on the object you want to segment...")
        coords = []

        def onclick(event):
            if event.xdata is not None and event.ydata is not None:
                coords.append((int(event.xdata), int(event.ydata)))
                plt.close()
        
        fig, ax = plt.subplots()
        ax.imshow(image)
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
        return coords[0] if coords else None