from ultralytics import YOLO

from sahi.utils.yolov8 import download_yolov8s_model
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from pathlib import Path
from IPython.display import Image
import glob
import os
from os import listdir
from PIL import Image

# Download YOLOv8 model
#yolov8_model_path = "models/yolov8n-seg.pt"
#download_yolov8s_model(yolov8_model_path)

# Download test images
#download_from_url('https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/small-vehicles1.jpeg', 'demo_data/small-vehicles1.jpeg')
#download_from_url('https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/terrain2.png', 'demo_data/terrain2.png')


# Load a model

model = YOLO('yolov8m-pose.pt')  # load a pretrained model (recommended for training)

#source_webcam_feed = 'D:/webcam/webcam_feed/*.jpg'
source_art_images = 'C:/Users/Administrator/Desktop/pose estimation/dummydata/9.jpg'

'''detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path='yolov8n-pose.pt',
    confidence_threshold=0.2,
    device="cpu" # or 'cuda:0'
)'''

# With an image path
#result = get_prediction(source, detection_model)'''

# Train the model
#results_webcam_feed = model(source = source_webcam_feed,show=False,conf=0.3,save=True, save_conf= True)
results_art = model(source = source_art_images,show=False,conf=0.2,save=True, save_conf= True)
#print(len(results))

#print(type(results))
'''i =0
for r in results_webcam_feed:
    #print(r)
    #print("*********printing keypoints***********")
    #print(r.keypoints)
    print("*********printing boxes***********")
    #print(r.boxes)
    if len(r.boxes.conf) == 0:
        print("no detections found")
        print("no conf")
        print(i)
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])
        #im.show()
        #print(str(im))
        im.save(f'C:/Users/Administrator/Desktop/pose estimation/tileart/{i}.jpg')
        folder_dir = 'C:/Users/Administrator/Desktop/pose estimation/tileart'
        print(f'C:/Users/Administrator/Desktop/pose estimation/tileart/{i}.jpg')
        for images in os.listdir(folder_dir):
            if (images.endswith(f"{i}.png") or images.endswith(f"{i}.jpg") or images.endswith(f"{i}.jpeg")):
                input_image = Image.open(folder_dir + '/' + images)
                tile_width = 500
                tile_height = 500
                width, height = input_image.size
                for y in range(0, height, tile_height):
                    for x in range(0, width, tile_width):
                        box = (x,y,x + tile_width, y + tile_height)
                        tile = input_image.crop(box)
                        tile.save(f'tiles/tile_{x}_{y}_{images}')
                source = 'tiles/*jpg'
                results = model(source,show=False,conf=0.3,save=True, save_conf= True,project="croppingpose", name = images)
                removing_files = glob.glob('tiles/*.jpg')
                for j in removing_files:
                    os.remove(j)
                reconstructed_image = Image.new("RGB", (width, height))
                for y in range(0,height,tile_height):
                    for x in range(0,width,tile_width):
                        tile = Image.open(f'croppingpose/{images}/tile_{x}_{y}_{images}')
                        reconstructed_image.paste(tile,(x,y))
                        reconstructed_image.save(f'croppingpose/{images}/reconstructed_image_{images}')
        i+=1'''

        #result = get_sliced_prediction(source,detection_model,slice_height=256,slice_width=256,overlap_height_ratio=0.2,overlap_width_ratio=0.2)
        #result.export_visuals(export_dir="runs/sahi/", file_name='sahiposejp')
        #results_sahi = model('sahipose.jpeg',show=False,conf=0.3,save=False, save_conf= True)'''


'''import os.path
if os.path.exists('runs/sahi/sahiposejp.png'):
   results_sahi = model('runs/sahi/sahiposejp.png',show=False,conf=0.3,save=True, save_conf= True)
else:
   print("no images")
#results_sahi = model('runs/sahi/sahipose.jpeg',show=False,conf=0.3,save=True, save_conf= True)

#print(result)'''

# Access the object prediction list
'''object_prediction_list = result.object_prediction_list

# Convert to COCO annotation, COCO prediction, imantics, and fiftyone formats
result.to_coco_annotations()[:3]
result.to_coco_predictions(image_id=1)[:3]
result.to_imantics_annotations()[:3]
result.to_fiftyone_detections()[:3]
'''


