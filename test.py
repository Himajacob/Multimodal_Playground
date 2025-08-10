from multimodal_controller import MultimodalController
from pynput import keyboard
controller = MultimodalController(dataset_config=[{"type":"image_text",
                                            "dataset":"mscoco",
                                            "path":"asuglia/small_coco",
                                            "selected_columns":["image","sentences.raw"]},
                                            {"type":"text",
                                             "dataset":"fineweb",
                                             "path":"HimaLevenSuprabha/fineweb-reduced",                                  
                                             "selected_columns":["text"]
                                             },
                                            {"type":"pixel",
                                             "dataset":"bookcorpus",
                                             "path":"HimaLevenSuprabha/small-rendered-bookcorpus",
                                             "selected_columns":["pixel_values"]}],
                                  render = True,
                                  targetObject=["Television","Laptop","Mobile"],
                                  scene="FloorPlan_Val3_1",
                                  width=1280,
                                  height=720,
                                  renderInstanceSegmentation=True)
event = controller.step(action="Pass")


def get_object_id_by_type(object_type):
    event = controller.step(action="Pass")
    for obj in event.metadata['objects']:
        if object_type.lower() in obj['objectType'].lower():
            return obj['objectId']
    return None


targetObjectId = get_object_id_by_type("Laptop")
print(targetObjectId)
def on_press(key):
    try:
        if key == keyboard.Key.up:
            controller.step(action="MoveAhead")
            controller.step(action="Pass")
        elif key == keyboard.Key.space:
            
            #event = controller.step(action="ToggleObjectOn", objectId=targetObjectId, debug = False)
            event = controller.step(action="ToggleObjectOn", objectId=None, debug = True,sam=True)
            #event = controller.step(action="ToggleObjectOn", objectId=None,debug = True)
            
            controller.step(action="Pass")
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


with keyboard.Listener(on_press=on_press) as listener:
    listener.join()