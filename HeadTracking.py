from __future__ import with_statement
import cv2
import numpy as np
import onnxruntime
import argparse
import os
import math
from skimage.feature import hog
from skimage import exposure
from numpy import dot
from numpy.linalg import norm
from osnet import OSNet
from osnet import osnet_x1_0,osnet_ibn_x1_0

from data import DataManager
import cv2
import torch
import numpy as np 
from torchvision.transforms import (
    Resize, Compose, ToTensor, Normalize, ColorJitter, RandomHorizontalFlip
)
from data import DataManager
from PIL import Image
import numpy as np
from matplotlib import cm
import re
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)
def build_transforms(
    height,
    width,
    transforms='random_flip',
    norm_mean=[0.485, 0.456, 0.406],
    norm_std=[0.229, 0.224, 0.225],
    **kwargs
):
    """Builds train and test transform functions.

    Args:
        height (int): target image height.
        width (int): target image width.
        transforms (str or list of str, optional): transformations applied to model training.
            Default is 'random_flip'.
        norm_mean (list or None, optional): normalization mean values. Default is ImageNet means.
        norm_std (list or None, optional): normalization standard deviation values. Default is
            ImageNet standard deviation values.
    """
    if transforms is None:
        transforms = []

    if isinstance(transforms, str):
        transforms = [transforms]

    if not isinstance(transforms, list):
        raise ValueError(
            'transforms must be a list of strings, but found to be {}'.format(
                type(transforms)
            )
        )

    if len(transforms) > 0:
        transforms = [t.lower() for t in transforms]

    if norm_mean is None or norm_std is None:
        norm_mean = [0.485, 0.456, 0.406] # imagenet mean
        norm_std = [0.229, 0.224, 0.225] # imagenet std
    normalize = Normalize(mean=norm_mean, std=norm_std)

    print('Building train transforms ...')
    transform_tr = []

    print('+ resize to {}x{}'.format(height, width))
    transform_tr += [Resize((height, width))]

    if 'random_flip' in transforms:
        print('+ random flip')
        transform_tr += [RandomHorizontalFlip()]

    if 'random_crop' in transforms:
        print(
            '+ random crop (enlarge to {}x{} and '
            'crop {}x{})'.format(
                int(round(height * 1.125)), int(round(width * 1.125)), height,
                width
            )
        )
        transform_tr += [Random2DTranslation(height, width)]

    if 'random_patch' in transforms:
        print('+ random patch')
        transform_tr += [RandomPatch()]

    if 'color_jitter' in transforms:
        print('+ color jitter')
        transform_tr += [
            ColorJitter(brightness=0.2, contrast=0.15, saturation=0, hue=0)
        ]

    print('+ to torch tensor of range [0, 1]')
    transform_tr += [ToTensor()]

    print('+ normalization (mean={}, std={})'.format(norm_mean, norm_std))
    transform_tr += [normalize]

    if 'random_erase' in transforms:
        print('+ random erase')
        transform_tr += [RandomErasing(mean=norm_mean)]

    transform_tr = Compose(transform_tr)

    print('Building test transforms ...')
    print('+ resize to {}x{}'.format(height, width))
    print('+ to torch tensor of range [0, 1]')
    print('+ normalization (mean={}, std={})'.format(norm_mean, norm_std))

    transform_te = Compose([
        Resize((height, width)),
        ToTensor(),
        normalize,
    ])

    return transform_tr, transform_te
def get_key_from_value(d, val):

    keys=[]
    for t, v in d.items():
        if (np.array(v) == np.array(val)).all():

            keys.append(t)
            break
         

    if len(keys)>0:

        return keys[0]
    return None
class YOLOv7:
    def __init__(self, path, conf_thres=0.10, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.class_names = ['head']
        # Initialize model
        session_option = onnxruntime.SessionOptions()
        session_option.log_severity_level = 3
        self.session = onnxruntime.InferenceSession(path, sess_options=session_option,providers=[ 'CPUExecutionProvider'] )
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape
        self.input_height = int(self.input_shape[2])
        self.input_width = int(self.input_shape[3])
        
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
        self.has_postprocess = False if len(self.output_names)==1 else True
#         print(self.has_postprocess)

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img.astype(np.float32) / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
        return input_tensor

    def detect(self, image):
        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.session.run(self.output_names, {input_name: input_tensor for input_name in self.input_names})

        if self.has_postprocess:
            boxes, scores, class_ids = self.parse_processed_output(outputs)

        else:
            # Process output data
            boxes, scores, class_ids = self.process_output(outputs)
        
        return boxes, scores, class_ids
    

    def process_output(self, output):
        predictions = np.squeeze(output[0])
        
        # Filter out object confidence scores below threshold
        obj_conf = predictions[:, 4]
        predictions = predictions[obj_conf > self.conf_threshold]
        obj_conf = obj_conf[obj_conf > self.conf_threshold]
        
        # Multiply class confidence with bounding box confidence
        predictions[:, 5:] *= obj_conf[:, np.newaxis]
        
        # Get the scores
        scores = np.max(predictions[:, 5:], axis=1)
        
        # Filter out the objects with a low score
        valid_scores = scores > self.conf_threshold
        predictions = predictions[valid_scores]
        scores = scores[valid_scores]
        
        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 5:], axis=1)
        
        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.conf_threshold,
                                   self.iou_threshold)
        if len(indices)>0:
            indices = indices.flatten()
        return boxes[indices], scores[indices], class_ids[indices]
    
    def parse_processed_output(self, outputs):
        scores = np.squeeze(outputs[0])
        predictions = outputs[1]
        
        # Filter out object scores below threshold
        valid_scores = scores > self.conf_threshold
        predictions = predictions[valid_scores, :]
        scores = scores[valid_scores]
        
        # Extract the boxes and class ids
        class_ids = predictions[:, 1]
        boxes = predictions[:, 2:]
        
        # In postprocess, the x,y are the y,x
        boxes = boxes[:, [1, 0, 3, 2]]
        boxes = self.rescale_boxes(boxes)
        return boxes, scores, class_ids
    
    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]
        
        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)
        
        # Convert boxes to xywh format
        boxes_ = np.copy(boxes)
        boxes_[..., 0] = boxes[..., 0] - boxes[..., 2] * 0.5
        boxes_[..., 1] = boxes[..., 1] - boxes[..., 3] * 0.5
        return boxes_
    
    def rescale_boxes(self, boxes):
        
        # Rescale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes
    
    def draw_detections(self, image, boxes, scores, class_ids):
        for box, score, class_id in zip(boxes, scores, class_ids):
            x, y, w, h = box.astype(int)
            
            # Draw rectangle
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), thickness=2)
            label = self.class_names[class_id]
            label = f'{label} {int(score * 100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
        return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str, default='images/frame604.jpg', help="image path")
    parser.add_argument('--modelpath', type=str, default='models/yolov7_head_0.752_480x640.onnx', help="onnx filepath")
    parser.add_argument('--confThreshold', default=0.10, type=float, help='class confidence')
    parser.add_argument('--nmsThreshold', default=0.1, type=float, help='nms iou thresh')
    args = parser.parse_args()
    
    
    
    modelpath = 'models/yolov7_head_0.752_480x640.onnx'
    confThreshold = 0.10
    nmsThreshold = 0.2
    # Initialize YOLOv7 object detector
    yolov7_detector = YOLOv7(modelpath, conf_thres=confThreshold, iou_thres=nmsThreshold)
    image_path = './Images5/'
    counter = 0
    person_box_store = {}
    model = osnet_ibn_x1_0()
    transform_tr, transform_te = build_transforms(
        224,
        224,
        transforms='random_flip',
        norm_mean=[0.485, 0.456, 0.406],
        norm_std=[0.229, 0.224, 0.225],
    )    

    for imgpath in  sorted_alphanumeric(os.listdir('./Images/')):
        print(imgpath)
    
        
    
        srcimg = cv2.imread('./Images/' + imgpath)
#         print(srcimg.shape)

        # Detect Objects
        boxes, scores, class_ids = yolov7_detector.detect(srcimg)
        boxes = boxes
        scores= scores
        class_ids = class_ids
        
        

        box_list = {}
        print("Length of Boxes is: " , len(boxes))




        if counter == 0:
#             print(srcimg.shape)
            
            peoples = len(boxes)-1
            for t in range(0,len(boxes)):
                box_list["person" + str(t)] = boxes[t]

            final_center_list = {}
            final_center_list2 = {}
            lol = 1
            
            
            
            

            for key in box_list.keys():
                
                temp_key = box_list[key]


                temp_center_img = srcimg[int(temp_key[1]):int(temp_key[1]+temp_key[3]),int(temp_key[0]): int(temp_key[0]+temp_key[2])]
                result = np.array(temp_center_img)
                result = Image.fromarray(np.uint8( result)).convert('RGB')
                result = transform_te(result)
                result = torch.unsqueeze(result, dim=0)
                result = model(result)
                result = result.detach().cpu().numpy()
                result = np.squeeze(result)
                result = np.array(result.tolist())
                
                

                
                
                temp_center_img = result
                 
                 
            
#                 print(temp_center_img)
                final_center_list['c_' + str(key)] = temp_center_img
                final_center_list2['c_' + str(key)] = [temp_key[0], temp_key[1] , temp_key[2], temp_key[3]]    
                lol+=1


            for key in box_list.keys():
                new_key = box_list[key]
                person_box_store[key] = [[str(imgpath), int(new_key[0]), int(new_key[1]), int(new_key[0] + new_key[2]), int(new_key[1] + new_key[3]) ,-1, -1]]
                
               


            counter+=1

        else:

           

            new_temp_box = {}
            
            """
            
            Storing temporary boxes
            
            """
            for box in range(0, len(boxes)):
                new_temp_box['head' + str(box)] = boxes[box]



            
        
            """
            
            Storing Temporary Centers
            
            """
            temp_centers = {}
            temp_centers2 = {}
#             print(srcimg)
            lol =1
            for box in range(0,len(boxes)):
                new_key = new_temp_box['head' + str(box)]


                tmp_image = srcimg[abs(int(new_key[1])):abs(int(new_key[1]+new_key[3])),abs(int(new_key[0])): int(abs(new_key[0])+new_key[2])]

                result2 = np.array(tmp_image)
 
                result2 = Image.fromarray(np.uint8(result2)).convert('RGB')
                result2 = transform_te(result2)
                result2 = torch.unsqueeze(result2, dim=0)
                result2 = model(result2)
                result2 = result2.detach().cpu().numpy()
                result2 = np.squeeze(result2)
                result2 = np.array(result2.tolist())
                temp_centers['temp_c'+ str(box)] = result2
                temp_centers2['temp_c'+ str(box)] =[new_key[0], new_key[1], new_key[2], new_key[3]]
                lol+=1
                
#             print("Temp Centers are : ", temp_centers)
#             break
        
        
            


            coord_mat = []
            coord_mat2 = []
            center_mat = []
            center_mat2 = []
            
            for key in temp_centers.keys():
                t_key = temp_centers[key]
                t_key2 = temp_centers2[key]
                coord_mat.append([t_key])
                coord_mat2.append([t_key2])
            
#             print(len(coord_mat))
            for key in final_center_list.keys():
                center_mat.append(final_center_list[key])
                center_mat2.append(final_center_list2[key])


            coord_ref = {}
            coord_dict = {}
            coord_ref2 = {}
            coord_dict2 = {}             
            
            for length in range(0,len(coord_mat)):
#                 print("Value that goes into ref table: ", coord_mat[length][0])#[0]
#                 break
                coord_ref['head' + str(length)] = coord_mat[length][0]
                coord_ref2['head' + str(length)] = coord_mat2[length]
#             print("Length of heads detected is: ", len(coord_ref.keys()))
            for key in coord_ref.keys():
                coord_dict[key] = new_temp_box[key]
#             print(coord_ref)
            



            p_counter = 0
            new_arr = []
            
        
#             print(imgpath)
#             print("Len of the center mat is ", len(center_mat))
           
            for center in range(len(center_mat)):
#                 break
#                 print("Center is: ", center)
                min_val = 100000
                min_val2 = 0
                


                for coord in range(len(coord_mat)) :
                    
        
                    if len(np.array(center_mat[center]).shape) ==2:
                       center_mat[center] = np.squeeze(np.array(center_mat[center]))


                    new_matrix = [abs(a_i - b_i)**2 for a_i, b_i in zip(coord_mat[coord][0], center_mat[center])]

                    dist1= np.sum(np.sqrt(new_matrix))
                    dist2 = math.hypot(np.squeeze(coord_mat2[coord])[0] - center_mat2[center][0],np.squeeze(coord_mat2[coord])[1] - center_mat2[center][1])**2  # xy
                    dist2 += math.hypot(np.squeeze(coord_mat2[coord])[0]+np.squeeze(coord_mat2[coord])[2] - (center_mat2[center][0]+center_mat2[center][2]), np.squeeze(coord_mat2[coord])[1] -center_mat2[center][1])**2 #x+w,ya
                    dist2 += math.hypot(np.squeeze(coord_mat2[coord])[0] - center_mat2[center][0], (np.squeeze(coord_mat2[coord])[1] +np.squeeze(coord_mat2[coord])[3]) - (center_mat2[center][1] + center_mat2[center][3]))**2 #x, y+h
                    dist2 += math.hypot((np.squeeze(coord_mat2[coord])[0]+np.squeeze(coord_mat2[coord])[2]) - (center_mat2[center][0] +center_mat2[center][2]) , (np.squeeze(coord_mat2[coord])[1] +np.squeeze(coord_mat2[coord])[3]) - (center_mat2[center][1] + center_mat2[center][3]))**2
                    
#                     print("Dist2 is : ",dist2)
                    dist2 = dist2
                    
                    dist =dist2 + dist1 # 
                    
                    
                    
                    if dist < min_val :
                        min_val = dist
                        tmp_near_pt = coord_mat[coord]
                        tmp_near_pt2 = np.squeeze(coord_mat2[coord])                    
                    
#                     print(dist)
                    

   
                key = get_key_from_value(coord_ref, tmp_near_pt)
                    
                    
                    
                    

                if 'person' + str(p_counter) not in person_box_store.keys():
                    

                    person_box_store['person' + str(p_counter)] = [[str(imgpath), int(coord_dict[key][0]), int(coord_dict[key][1]),int(coord_dict[key][0]+ coord_dict[key][2]), int(coord_dict[key][1]+coord_dict[key][3]), -1, -1]]
                    print(f"person{p_counter+1} matches with {key}  and min distance is {min_val}")    
                elif min_val < 5000:
                    person_box_store['person' + str(p_counter)].append([str(imgpath), int(coord_dict[key][0]), int(coord_dict[key][1]),int(coord_dict[key][0]+ coord_dict[key][2]), int(coord_dict[key][1]+coord_dict[key][3]), -1, -1])
    #                     print("Temp near point is: ", tmp_near_pt)
    #                     print("Temp near point first element is: " ,tmp_near_pt)
                    final_center_list['c_person' + str(p_counter) ] = tmp_near_pt
                    final_center_list2['c_person' + str(p_counter) ] = tmp_near_pt2
                    print(f"person{p_counter+1} matches with {key}  and min distance is {min_val}")                    
                p_counter+=1
                
                
# and has coordinates {tmp_near_pt}













        # Draw detections
        dstimg = yolov7_detector.draw_detections(srcimg, boxes, scores, class_ids)




os.mkdir('./ImageReport')
curr_path = os.getcwd()
for keys in person_box_store.keys():
    os.chdir('./ImageReport')
    with open(str(keys)+ '.txt', 'w') as f:
        for _list in person_box_store[keys]:
            k=0
            for _string in _list:
                if (k+ 1) != len(_list):
                    f.write(str(_string) + ',')
                elif (k + 1) == len(_list):
                     f.write(str(_string))
                k+=1
       
            f.write('\n')
    os.chdir(curr_path)
    
import cv2 
import numpy as np 
import pandas as pd 
import os
import glob

person_data = {}
for report in os.listdir('./ImageReport/'):
    data = pd.read_csv('./ImageReport/' + str(report))
    data.columns= ['Frame', 'Left', 'Top', 'Right', 'Bottom', 'GazeX', 'GazeY']
    person_data[str(report).split('.')[0]] = data


os.mkdir('./HeadTrackingResults/')

for img in sorted(os.listdir('./Images')):
#     print(img)
    image = cv2.imread('./Images/' + img)
    f2 = []
    for data_key in person_data.keys():
        f2.append(person_data[data_key].loc[person_data[data_key]['Frame'] == img])

    count = 0
    c = [(255,0,0),(0,255,0),(0,0,255),(255,0,255),(255,255,0),(255,0,255),(255,160,122),(124,252,0)]

    for f in f2: #,f4]:
        color= c[count]
        count+=1

            
        
        if f.shape[0]!= 0:
            image= cv2.rectangle(image, (f['Left'].iloc[0], f['Top'].iloc[0]), (f['Right'].iloc[0],f['Bottom'].iloc[0]), 
                                 color = color,thickness =2)
    curr_path = os.getcwd()
    
    os.chdir('./HeadTrackingResults/')
    cv2.imwrite(str(img),image)
    os.chdir(curr_path)
img_array = []

for filename in glob.glob('./HeadTrackingResults/*.jpg'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    print(size)
    img_array.append(img)

# size = size
out = cv2.VideoWriter('HeadTrackingResult.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()        
                    
        
    
