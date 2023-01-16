from __future__ import with_statement
import cv2
import numpy as np
import onnxruntime
import argparse
import os
import math

def get_key_from_value(d, val):
    keys = [k for k, v in d.items() if v == val]
    if keys:
        return keys[0]
    return None
def iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+ boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3],boxB[1]+ boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs(boxA[2] * boxA[3])
    boxBArea = abs(boxB[2] ) * (boxB[3] )

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou
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
            # top = max(y1, labelSize[1])
            # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255,255,255), cv.FILLED)
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
    image_path = './Images3/'
    counter = 0
    person_box_store = {}

    for imgpath in os.listdir('./Images3/'):
#         print(imgpath)
    
        
    
        srcimg = cv2.imread('./Images3/' + imgpath)

        # Detect Objects
        boxes, scores, class_ids = yolov7_detector.detect(srcimg)
        boxes = boxes
        scores= scores
        class_ids = class_ids
        
        
        new_box_list = []
#         print("Length of the boxes before IOU correction is: ", len(boxes))    
        for b in boxes:
            temp_box = b
            flag_box = 0
            for a in boxes:
#                 print("IOU between boxes is: ", iou(b,a))
                if (np.array(a) != np.array(b)).all() and iou(b,a) >0.05 :
                    flag_box = 1
                    break

            if flag_box ==0 and any(np.array_equal(b, matrix) for matrix in new_box_list) == False :
                new_box_list.append(b)
#         print(new_box_list)            
        new_box_list = np.array(new_box_list)

        boxes = new_box_list
        if len(boxes) <4:
            counter = 0 
#         print("Length of the boxes after IOU correction is: ", len(boxes))            
        

        box_list = {}
#         print("Length of the boxes", len(boxes))




        if counter == 0:
            
            peoples = len(boxes)-1
            for t in range(0,len(boxes)):
                box_list["person" + str(t)] = boxes[t]

            final_center_list = {}
            
            
            

            for key in box_list.keys():
                temp_key = box_list[key]
                final_center_list['c_' + str(key)] = [temp_key[0], temp_key[1] , temp_key[2], temp_key[3]]


            for key in box_list:
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


            print(imgpath)
            
        
            """
            
            Storing Temporary Centers
            
            """
            temp_centers = {}
            
            for box in range(0,len(boxes)):
                new_key = new_temp_box['head' + str(box)]
                temp_centers['temp_c'+ str(box)] = [new_key[0], new_key[1], new_key[2], new_key[3]]
            
        
        
            


            coord_mat = []
            center_mat = []
            
            for key in temp_centers.keys():
                t_key = temp_centers[key]
                coord_mat.append(t_key)
            
#             print(len(coord_mat))
            for key in final_center_list.keys():
                center_mat.append(final_center_list[key])
 

            coord_ref = {}
            coord_dict = {}
            
            for length in range(0,len(coord_mat)):
                coord_ref['head' + str(length)] = coord_mat[length]
            
            for key in coord_ref.keys():
                coord_dict[key] = new_temp_box[key]
            



            p_counter = 0
            new_arr = []
            
        
#             print(imgpath)
            print("Len of the center mat is ", len(center_mat))
    
            for center in center_mat:
                min_val = 1000000
                min_val2 = 0
#                 print("Center is: ", center)
        

                for coord in coord_mat:
#                     print("Coord is: ", coord)
        
                    
                    dist = math.hypot(coord[0] - center[0], coord[1] - center[1])**2  # xy
                    dist += math.hypot(coord[0]+coord[2] - (center[0]+center[2]), coord[1] - center[1])**2 #x+w,y
                    dist += math.hypot(coord[0] - center[0], (coord[1] +coord[3]) - (center[1] + center[3]))**2 #x, y+h
                    dist += math.hypot((coord[0]+coord[2]) - (center[0] +center[2]) , (coord[1] +coord[3]) - (center[1] + center[3]))**2 #x+w, y+h
                
                    dist = np.sqrt(dist)
                    
#                     print("Distance is: ",dist)
                    
                
                    
                    if dist < min_val :
#                         print("Coordinates are: ", coord)
                        min_val = dist
                        tmp_near_pt = coord
                    
                
                
#                 for coord in coord_mat :
        
                    
#                     dist2 = math.hypot(coord[0] - center[0], coord[1] - center[1])**2  # xy
#                     dist2 += math.hypot(coord[0]+coord[2] - (center[0]+center[2]), coord[1] - center[1])**2 #x+w ,y
#                     dist2 += math.hypot(coord[0] - center[0], (coord[1] +coord[3]) - (center[1] + center[3]))**2 # x, y+h
#                     dist2 += math.hypot((coord[0]+coord[2]) - (center[0] +center[2]) , (coord[1] +coord[3]) - (center[1] + center[3]))**2 #x+w, y+h
                
                    
#                     dist2 = np.sqrt(dist2)
#                     if dist2 > min_val2: 
#                         min_val2 = dist2
#                         tmp_near_pt2 = coord
                        
                    
                 

                
# #                 print("MinVal is : ", min_val)
                
# #                 print("Value of peoples is: ", peoples)
# #                 print("Number of boxes found is: ", len(boxes))
# #                 print(peoples)
# #                 print(len(boxes))
#                 if peoples < len(boxes)-1 and min_val2>50:
#                     print("Value of the max val is: ", min_val2)
#                     print(len(boxes))
#                     print(peoples)
#                     peoples+=1
#                     final_center_list['c_person' + str(peoples) ] = tmp_near_pt2
# #                 elif min_val2 <1:
#                     near_pt = tmp_near_pt

        
                key = get_key_from_value(coord_ref, tmp_near_pt)
                    
                    
                    
                    

                if 'person' + str(p_counter) not in person_box_store.keys():
                    

                    person_box_store['person' + str(p_counter)] = [[str(imgpath), int(coord_dict[key][0]), int(coord_dict[key][1]),int( coord_dict[key][0]+ coord_dict[key][2]), int(coord_dict[key][1]+coord_dict[key][3]), -1, -1]]
                    print(f"person{p_counter+1} matches with {key} and has coordinates {tmp_near_pt} and min distance is {min_val}")    
                elif min_val < 50:
                    person_box_store['person' + str(p_counter)].append([str(imgpath), int(coord_dict[key][0]), int(coord_dict[key][1]),int( coord_dict[key][0]+ coord_dict[key][2]), int(coord_dict[key][1]+coord_dict[key][3]), -1, -1])
                    final_center_list['c_person' + str(p_counter) ] = tmp_near_pt
                    print(f"person{p_counter+1} matches with {key} and has coordinates {tmp_near_pt} and min distance is {min_val}")                    
                p_counter+=1
                
                














        # Draw detections
        dstimg = yolov7_detector.draw_detections(srcimg, boxes, scores, class_ids)
        winName = 'Deep learning object detection in ONNXRuntime'
        path_f = os.getcwd()
        os.chdir('./F')

        cv2.imwrite(str(imgpath), dstimg)

        os.chdir(path_f)


# with open('person1.txt', 'w') as f:
#     for _list in person1:
#         k=0
#         for _string in _list:
#             if (k+ 1) != len(_list):
#                 f.write(str(_string) + ',')
#             elif (k + 1) == len(_list):
#                  f.write(str(_string))
#             k+=1
                
#         f.write('\n')




#print(person_box_store.keys())


for keys in person_box_store.keys():
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
    

            
        
    
# 
#     
#         k=0
#         for _string in _list:
#             if (k+ 1) != len(_list):
#                 f.write(str(_string) + ',')
#             elif (k + 1) == len(_list):
#                  f.write(str(_string))
#             k+=1
                
#         f.write('\n')

        
        


# with open('person3.txt', 'w') as f:
#     for _list in person3:
#         k=0
#         for _string in _list:
#             if (k+ 1) != len(_list):
#                 f.write(str(_string) + ',')
#             elif (k + 1) == len(_list):
#                  f.write(str(_string))
#             k+=1
                
#         f.write('\n')
        
# with open('person4.txt', 'w') as f:
#     for _list in person4:
#         k=0
#         for _string in _list:
#             if (k+ 1) != len(_list):
#                 f.write(str(_string) + ',')
#             elif (k + 1) == len(_list):
#                  f.write(str(_string))
#             k+=1
                
#         f.write('\n')        
        
        
        
        
#     cv2.imshow(winName, dstimg)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
