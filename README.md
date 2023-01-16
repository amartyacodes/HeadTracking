# Head Tracking Model 
**This is a head tracking model which utilizes YOLOv7 to detect the head and Omni Scale Model for Person Reidentification features for tracking the human heads in a video.**
![HeadTrackingResult2 (5)](https://user-images.githubusercontent.com/44440114/212591528-1ee7e928-64ac-48d9-8afd-e1d1e67d2801.gif)

## How to Run the Model?

**STEP1:** Download the model weights from the given folder link :
https://drive.google.com/drive/folders/1tC7iObkkKkmCMiYWT0rzb6VDVkSjg1rB?usp=sharing

NOTE : Download all the models present in the directory and place it in the same directory as current one.

**STEP3:** Place the frames of the video in the "./Images" folder

**STEP2:** Run the command given below
```python
!HeadTracking.py
```

**STEP4:** Head Tracking results will automatically generated with the name of HeadTrackingResult.mp4 file.

## Working Principle

**1.** Initially Detect the Heads for the first frame using YOLOv7 Head Detector model trained on Crowdhuman dataset

**2.** Distribute the bounding boxes into different classes

**3.** For the subsequent frames, again detect the head bounding boxes using the YOLO Head Detector model

**4.** Obtain the distance between corresponding frames and based on nearest distance class assignment, assign the classes of the bounding boxes found in the current frame. For better understanding look at the below figure
![BoxClassAssignment](https://user-images.githubusercontent.com/44440114/212593283-27fb96e3-1469-426d-8772-df610d40f866.jpg)

**5.** Continue the above process till the end 

### NOTE: The only constraint is the number of people in a frame at the start of the video must be less than or equal to the number of people in the rest of the frames. Work is being done for further upgrades

## This work has been done under the guidance of Professor Hajime Nagahara at Institute of Datability Science, Osaka University 
Lab Website: https://www.is.ids.osaka-u.ac.jp/

**My Website:** amartyacodes.github.io


