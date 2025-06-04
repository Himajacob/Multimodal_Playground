from ai2thor.controller import Controller
import pdb
class MultimodalController(Controller):
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
            data='testing...'
            print(data)
        else:
            data=None
        
        return event
    
    def is_targetobject(self,objectId,target_object):
        print(target_object)
        print(objectId)
        if objectId is None or target_object is None:
            return False
        print("test")
        obj_type = objectId.split('|')[0] if objectId else ''
        return obj_type == target_object
    

 