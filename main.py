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
                                   self.iou_threshold).flatten()
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
    image_path = './Images/'
    counter = 0
    person1 = []
    person2 =[]
    person3 = []
    person4 = []     
    for imgpath in os.listdir('./Images/'):
        print(imgpath)
    
        
    
        srcimg = cv2.imread('./Images/' + imgpath)

        # Detect Objects
        boxes, scores, class_ids = yolov7_detector.detect(srcimg)
        boxes = boxes
        scores= scores
        class_ids = class_ids
        
        
#         if len(boxes) <4 :
#             continue
        if len(boxes)==4:
            print(len(boxes))
        
        

        
            if counter == 0:

                person1_box = boxes[0]
                person2_box = boxes[1]
                person3_box = boxes[2]
                person4_box = boxes[3]

                c1 = [person1_box[0] + person1_box[2]/2 , person1_box[1] + person1_box[3]/2] # --- initially randomly split into 3 different set
                c2 = [person2_box[0] + person2_box[2]/2 , person2_box[1] + person2_box[3]/2]
                c3 = [person3_box[0] + person3_box[2]/2 , person3_box[1] + person3_box[3]/2]
                c4 = [person4_box[0] + person4_box[2]/2 , person4_box[1] + person4_box[3]/2]

                person1.append([str(imgpath), int(person1_box[0]), int(person1_box[1]), int(person1_box[0] +person1_box[2]), int(person1_box[1] + person1_box[3]),-1,-1])
                person2.append([str(imgpath), int(person2_box[0]), int(person2_box[1]), int(person2_box[0] +person2_box[2]), int(person2_box[1] + person2_box[3]),-1,-1])             
                person3.append([str(imgpath), int(person3_box[0]), int(person3_box[1]), int(person3_box[0] +person3_box[2]), int(person3_box[1] + person3_box[3]),-1,-1])

                person4.append([str(imgpath), int(person4_box[0]), int(person4_box[1]), int(person4_box[0] +person4_box[2]), int(person4_box[1] + person4_box[3]),-1,-1])             


                counter+=1

            else:

                p1_box = boxes[0]
                p2_box = boxes[1]
                p3_box = boxes[2]
    #             if len(boxes) == 4:
                p4_box = boxes[3]

                print(len(boxes))

    #             if len( np.unique(np.array(boxes))) <3:
    #                 print("whoa")
    #                 print(imgpath)
    #                 break

                temp_c1 = [p1_box[0] + p1_box[2]/2 , p1_box[1] + p1_box[3]/2]
                temp_c2 = [p2_box[0] + p2_box[2]/2 , p2_box[1] + p2_box[3]/2]
                temp_c3 = [p3_box[0] + p3_box[2]/2 , p3_box[1] + p3_box[3]/2 ]
    #             if len(boxes) == 4:
                temp_c4 = [p4_box[0] + p4_box[2]/2 , p4_box[1] + p4_box[3]/2 ]   


                x11 = temp_c1[0] ## ----> center 1
                y11 = temp_c1[1] ## ----> center 2

                x21 = temp_c2[0] 
                y21 = temp_c2[1]

                x31 = temp_c3[0]
                y31 = temp_c3[1]
    #             if len(boxes) == 4:
                x41 = temp_c4[0]
                y41 = temp_c4[1]

                coord_mat = [[x11,y11] , [x21,y21], [x31,y31], [x41,y41]] #---- temp center locations 
                center_mat = [c1, c2, c3, c4] # given centers 

                coord_ref = {'head1': coord_mat[0], 'head2' : coord_mat[1], 'head3' : coord_mat[2],'head4' : coord_mat[3]}
                coord_dict = {'head1': p1_box, 'head2': p2_box, 'head3': p3_box,'head4': p4_box}



                p_counter = 0
                new_arr = [] 
    #             print(imgpath)
                for center in center_mat:
                    min_val = 1000000
    #                 
    #                 print(center)
    #                 if imgpath == '00011332.jpg':
    #                     print(coord_mat)
    #                     print("Center 1 is :" , c1)
    #                     print("Center 2 is :" , c2)
    #                     print("Center 3 is :" , c3)
    # #                     print("Center 4 is :" , c4)

    #                 if imgpath == '00011333.jpg':
    #                     print(coord_mat)
    #                     print("Center 1 is :" , c1)
    #                     print("Center 2 is :" , c2)
    #                     print("Center 3 is :" , c3)
    # #                     print("Center 4 is :" , c4)           

                    for coord in coord_mat:
                        dist = math.hypot(coord[0] - center[0], coord[1] - center[1])
                        if dist < min_val:
                            min_val = dist
                            tmp_near_pt = coord





                    if  p_counter==0:

                        near_pt= tmp_near_pt


                        key = get_key_from_value(coord_ref, tmp_near_pt)
                        new_arr.append(coord_dict[key][0])
                        print(f"person1 matches with {key} and has coordinates {tmp_near_pt} and min distance is {min_val}")


                        person1.append([str(imgpath), int(coord_dict[key][0]), int(coord_dict[key][1]),int(coord_dict[key][0]+ coord_dict[key][2]), int(coord_dict[key][1]+coord_dict[key][3]), -1, -1])



                        c1 = near_pt

                    elif  p_counter ==1:


                        near_pt= tmp_near_pt

                        key = get_key_from_value(coord_ref, tmp_near_pt)
                        new_arr.append(coord_dict[key][0])






                        print(f"person2 matches with {key} and has coordinates {tmp_near_pt} and min distance is {min_val}")
                        person2.append([str(imgpath), int(coord_dict[key][0]), int(coord_dict[key][1]),int(coord_dict[key][0]+ coord_dict[key][2]), int(coord_dict[key][1]+coord_dict[key][3]), -1, -1])


                        c2 = near_pt

                    elif  p_counter ==2:



                        near_pt= tmp_near_pt

                        key = get_key_from_value(coord_ref, tmp_near_pt)
                        new_arr.append(coord_dict[key][0])
                        print(f"person3 matches with {key} and has coordinates {tmp_near_pt} and min distance is {min_val}")

                        person3.append([str(imgpath), int(coord_dict[key][0]), int(coord_dict[key][1]),int(coord_dict[key][0]+ coord_dict[key][2]), int(coord_dict[key][1]+coord_dict[key][3]), -1, -1])

                        c3 = near_pt

                    elif  p_counter ==3:



                        near_pt= tmp_near_pt

                        key = get_key_from_value(coord_ref, tmp_near_pt)
                        new_arr.append(coord_dict[key][0])
    #                     print(f"person4 matches with {key} and has coordinates {tmp_near_pt} and min distance is {min_val}")

                        person4.append([str(imgpath), int(coord_dict[key][0]), int(coord_dict[key][1]),int(coord_dict[key][0]+ coord_dict[key][2]), int(coord_dict[key][1]+coord_dict[key][3]), -1, -1])

                        c4 = near_pt

                    p_counter+=1

                print(np.unique(np.array(new_arr)))

                if len(np.unique(np.array(new_arr))) >4 :
                    print("what the hell!")
                    print(str(imgpath))
                    print(new_arr)














            # Draw detections
            dstimg = yolov7_detector.draw_detections(srcimg, boxes, scores, class_ids)
            winName = 'Deep learning object detection in ONNXRuntime'
            path_f = os.getcwd()
            os.chdir('./F')
            
            cv2.imwrite(str(imgpath), dstimg)
            
            os.chdir(path_f)


with open('person1.txt', 'w') as f:
    for _list in person1:
        k=0
        for _string in _list:
            if (k+ 1) != len(_list):
                f.write(str(_string) + ',')
            elif (k + 1) == len(_list):
                 f.write(str(_string))
            k+=1
                
        f.write('\n')


with open('person2.txt', 'w') as f:
    for _list in person2:
        k=0
        for _string in _list:
            if (k+ 1) != len(_list):
                f.write(str(_string) + ',')
            elif (k + 1) == len(_list):
                 f.write(str(_string))
            k+=1
                
        f.write('\n')

        
        


with open('person3.txt', 'w') as f:
    for _list in person3:
        k=0
        for _string in _list:
            if (k+ 1) != len(_list):
                f.write(str(_string) + ',')
            elif (k + 1) == len(_list):
                 f.write(str(_string))
            k+=1
                
        f.write('\n')
        
with open('person4.txt', 'w') as f:
    for _list in person4:
        k=0
        for _string in _list:
            if (k+ 1) != len(_list):
                f.write(str(_string) + ',')
            elif (k + 1) == len(_list):
                 f.write(str(_string))
            k+=1
                
        f.write('\n')        
        
        
        
        
#     cv2.imshow(winName, dstimg)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
