from ai2thor.controller import Controller
from dataloader import DataLoader
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
class MultimodalController(Controller):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.dataloader=DataLoader()

    def step(self, action, objectId=None,target_object=None,object_mask=None,**kwargs):
        data=None
        if action in [
            "ToggleObjectOn", "ToggleObjectOff",
            "PickupObject", "OpenObject", "CloseObject",
            "PutObject", "DropHandObject"
        ]:
            args = {"action": action, "objectId": objectId, **kwargs}
        else:
            args = {"action": action, **kwargs}
        event = super().step(**args)
        if(action == "ToggleObjectOn" and self.is_targetobject(objectId,target_object)):
            data=self.dataloader.getdata()
            print(data["text"])
            # self.showdata(data["observation"])     
            data["observation"].show() 
        else:
            data=None
        
        return event
    
    def is_targetobject(self,objectId,target_object):
        if objectId is None or target_object is None:
            return False
        obj_type = objectId.split('|')[0] if objectId else ''
        return obj_type == target_object
    
    # def showdata(self,image):
    #     img_np = np.array(image)
    #     img_bgr = cv2.cvtcolor(img_np, cv2.COLOR_RGB2BGR)
    #     cv2.imshow('output', img_bgr)
    #     cv2.waitkey(1)
 