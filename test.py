from multimodal_controller import MultimodalController
from pynput import keyboard
controller = MultimodalController(path="asuglia/small_coco",
                                  scene="FloorPlan201",
                                  width=1280,
                                  height=720,
                                  renderInstanceSegmentation=True)
event = controller.step(action="Pass")
def on_press(key):
    try:
        if key == keyboard.Key.up:
            controller.step(action="MoveAhead")
            controller.step(action="Pass")
        elif key == keyboard.Key.space:
            # event = controller.step(action="ToggleObjectOn", objectId=targetObjectId, target_object= "Laptop",debug = True)
            event = controller.step(action="ToggleObjectOn", objectId=None, target_object= "Laptop",debug = True,sam=True)
            controller.step(action="Pass")
            for obj in event.metadata['objects']:
                if obj['objectId'] == targetObjectId:
                    print(obj['objectId'], obj['toggleable'], obj['isToggled'])
        elif key == keyboard.Key.down:
            controller.step(action="MoveBack")
            controller.step(action="Pass")
        elif key == keyboard.Key.left:
            controller.step(action="RotateLeft")
            controller.step(action="Pass")
        elif key == keyboard.Key.right:
            controller.step(action="RotateRight")
            controller.step(action="Pass")
        elif key == keyboard.Key.esc:
            print("Exiting and stopping Unity...")
            controller.stop()
            return False  # Exit listener, script ends
    except Exception as e:
        print("Error:", e)



def get_id(object_name):
    for obj in event.metadata['objects']:   
        if obj['objectId'].startswith(object_name):
            print(obj['objectId'], obj['toggleable'], obj['isToggled'])
            if(obj['toggleable']!=True):
                return None
            return obj['objectId']
    return None

targetObjectId = get_id("Laptop")
with keyboard.Listener(on_press=on_press) as listener:
    listener.join()