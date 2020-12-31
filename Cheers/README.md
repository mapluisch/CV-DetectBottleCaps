# Cheers - "Detect The Bottle Caps" (CV20/21)
**Cheers** is a tiny Python3 script, which 

 1. reads some input video file
 2. detects and extracts a still frame
 3. runs inference on the still frame
 4. detects and annotates any matching bottle caps (face-down/face-up/deformed)
 
 I've trained my model using YOLOv4-tiny using ~500 images for training.
 ## Dependencies
 OpenCV-Python
 Python3
 ## Install / Run
Simply clone this repo or download + unzip. Then, simply run
`python3 cheers.py --video some_vid.mp4`
or
`python3 cheers.py --image some_img.jpg`
and you'll get an annotated still frame in ./Output/Annotated Frames/ after approx. 0.2s for input images and approx. 0.5-1.0s for input videos. 
**Cheers** can also be used to live-annotate a passed-in video. You can also specify your own confidence- and non-max-suppresion-thresholds; please take a look at the parameters below.

### Parameters
```
--verbose 				print extensive information per step during runtime
--video					path to video file (to first detect a still frame from and then run inference on)
--image					path to image file (to directly run inference on)
--conf-thres			confidence threshold [0.0-1.0] whether or not a detected object is a bottle cap
--nms-thres				non-max-suppression threshold [0.0-1.0] used to select the best bounding box per detected object
--show-result			immediately show the annotated output frame using cv2.imshow
--save-stillframe		save the detected still frame (default = false)
--live					show live-annotation of the passed in video-file, frame-by-frame
```