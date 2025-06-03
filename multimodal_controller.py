from ai2thor.controller import Controller
import pdb
class MultimodalController(Controller):
    def step(self, action, object_id=None,target_object=None,object_mask=None,**kwargs):
        data=None
        if action in [
            "ToggleObjectOn", "ToggleObjectOff",
            "PickupObject", "OpenObject", "CloseObject",
            "PutObject", "DropHandObject"
        ]:
            args = {"action": action, "objectId": object_id, **kwargs}
        else:
            args = {"action": action, **kwargs}
        event = super().step(**args)
        if(action == "ToggleObjectOn" and self.is_targetobject(object_id,target_object)):
            data='testing...'
            print(data)
        else:
            data=None
        
        return event
    
    def is_targetobject(self,object_id,target_object):
        if object_id is None or target_object is None:
            return False
        if object_id.startswith(target_object):
            return True
        return False


 