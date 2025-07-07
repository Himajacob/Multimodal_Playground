from ai2thor.controller import Controller
from dataloader import DataLoader
from sammaskgenerator import SAMMaskGenerator
import numpy as np
from collections import Counter, OrderedDict
import copy
import cv2

class MultimodalController(Controller):
    def __init__(self,dataset_config,**kwargs):
        super().__init__(**kwargs)
        self.dataloader=DataLoader(dataset_config)
        self.sam_generator = SAMMaskGenerator() 

    def step(self, action, objectId=None,target_object=None,interact_mask=None,debug = False,sam_points=None,sam=False,**kwargs):
        data=None
        if debug is True and sam is False:
            self.get_segmentationmask() 
            interact_mask = self.convert_to_segmentationmask()
        
        if sam:
            frame = np.array(self.last_event.frame)
            mask = self.sam_generator.getMaskFromClick(frame, point=None, debug=True)
            objectId = self.va_interact(interact_mask=mask, debug=True)

        if objectId is None and interact_mask is not None and sam is False:
            objectId = self.va_interact(interact_mask=interact_mask, debug=debug)
        

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
            print(data["json"])  
        else:
            data=None
        
        return event
    
    def is_targetobject(self,objectId,target_object):
        if objectId is None or target_object is None:
            return False
        obj_type = objectId.split('|')[0] if objectId else ''
        return obj_type == target_object
    
    def prune_by_any_interaction(self, instances_ids):
        pruned_instance_ids = []
        for obj in self.last_event.metadata['objects']:
            obj_id = obj['objectId']
            if obj_id in instances_ids:
                if obj['pickupable'] or obj['receptacle'] or obj['openable'] or obj['toggleable'] or obj['sliceable']:
                    pruned_instance_ids.append(obj_id)
        
        ordered_instance_ids = [id for id in instances_ids if id in pruned_instance_ids]
        return ordered_instance_ids
    
    def va_interact(self, interact_mask = None,mask_px_sample = 1, debug = True):
        # ALFRED code
        if type(interact_mask) is str and interact_mask == "NULL":
            raise Exception("NULL Mask")
        elif interact_mask is not None:
            instance_segs = np.array(self.last_event.instance_segmentation_frame)
            color_to_object_id = self.last_event.color_to_object_id

            nz_rows, nz_cols = np.nonzero(interact_mask)
            instance_counter = Counter()
            
            for i in range(0, len(nz_rows), mask_px_sample):
                x, y = nz_rows[i], nz_cols[i]
                instance = tuple(instance_segs[x, y])
                instance_counter[instance] += 1

            if debug:
                print("action_box","instance_counter",instance_counter)
                
            iou_scores = {}
            for color_id, intersection_count in instance_counter.most_common():
                union_count = np.sum(np.logical_or(np.all(instance_segs == color_id, axis=2), interact_mask.astype(bool)))
                iou_scores[color_id] = intersection_count / float(union_count)
            iou_sorted_instance_ids = list(OrderedDict(sorted(iou_scores.items(), key=lambda x: x[1], reverse=True)))

            inv_obj = self.last_event.metadata['inventoryObjects'][0]['objectId'] \
                   if len(self.last_event.metadata['inventoryObjects']) > 0 else None
            all_ids = [color_to_object_id[color_id] for color_id in iou_sorted_instance_ids
                       if color_id in color_to_object_id and color_to_object_id[color_id] != inv_obj]
                
            if debug:
                print("action_box", "all_ids", all_ids)

            instance_ids = [inst_id for inst_id in all_ids if inst_id is not None]
            if debug:
                print("action_box", "instance_ids", instance_ids)
                

            instance_ids = self.prune_by_any_interaction(instance_ids)

            # if debug:
            #         print("action_box", "instance_ids", instance_ids)
            #         instance_seg = copy.copy(instance_segs)
            #         instance_seg[:, :, :] = interact_mask[:, :, np.newaxis] ==1
            #         instance_seg *=225

            #         cv2.imshow('segs',instance_segs)
            #         cv2.imshow('mask', instance_seg)
            #         cv2.imshow('full', self.last_event.frame[:,:,::-1])
            #         cv2.waitKey(0)

            if len(instance_ids) ==0:
                    err ="bad interact mask. Target not found"
                    success = False
                    return success, None, None, err, None
            target_instance_id = instance_ids[0]
            
        else:
            target_instance_id =""
            
        
        return target_instance_id
    
    def get_segmentationmask(self):

        color_to_object_id = self.last_event.color_to_object_id
        objects_metadata = self.last_event.metadata['objects']
        interactable_ids = self.prune_by_any_interaction(color_to_object_id.values())

        for color, object_id in color_to_object_id.items():
            if object_id in interactable_ids:
                obj_type = next((obj['objectType'] for obj in objects_metadata if obj['objectId'] == object_id), "unknown")
                print(f"  - ID: {object_id:<35} | Type: {obj_type:<12} | Segmentation Color: {color}")
    
    def convert_to_segmentationmask(self):
        color_input = input("Enter segmentation color as R,G,B (e.g., 120,90,50): ")
        r, g, b = map(int, color_input.strip().split(','))
        target_color = (r, g, b)
        seg = np.array(self.last_event.instance_segmentation_frame)
        return np.all(seg == target_color, axis=2).astype(np.uint8)

