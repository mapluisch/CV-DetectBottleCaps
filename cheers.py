#!/usr/bin/env python3
""" 
"Cheers" detects and annotates bottle caps in a given video or image file and returns an annotated output frame.
Given a video-file, it'll either detect a still frame and run inference and annotation on it or do it live, frame-by-frame (--live argument).
Given an image-file, it'll directly run inference and annotation on it.
"""

import os
import csv
import cv2
import sys
import math
import time
import argparse
import numpy as np

# ------------------- Variables -------------------

# -- Whether the program is in competition mode (no fancy argument parser, only csv output) or not
COMPETITION_MODE = False

# -- Video Frame variables
FRAME_POSITION = cv2.CAP_PROP_POS_FRAMES
FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT
START_FRAME_FACTOR = 1/3
END_FRAME_FACTOR = 2/3
LIVE_FRAMESKIP = 0

# -- Relaitve output directories
STILLFRAME_OUTPUT_DIR = '/still frames/'
ANNOTATED_OUTPUT_DIR = '/annotated frames/'

# -- Trained YOLO config directory and filename
CONFIG_DIR = './dnn/configs/'
CONFIG_NAME = 'v4tiny_9000.cfg'
# -- Trained YOLO weights directory and filename
WEIGHTS_DIR = './dnn/weights/'
WEIGHTS_NAME = 'v4tiny_9000.weights'

# -- Inference / Annotation variables
NON_MAX_SUPPRESSION_THRESHOLD = 0.5
CONF_THRESHOLD = 0.5
ANNOTATION_LINE_THICKNESS = 4

# -- Region of Interest min- and max-area relative to input image
ROI_MIN_SIZE = 0.25
ROI_MAX_SIZE = 0.8

# -- Helper variables for detected object classes
CLASS_COLORS = [(255, 101, 59), (19, 201, 225), (53, 159, 7)]
CLASS_NAMES = ["face-down", "face-up", "deformed"]
CSV_CLASS_NAMES = ["BottleCap_FaceDown", "BottleCap_FaceUp", "BottleCap_Deformed"]
CLASS_NAMES_COLORED = ["\u001b[34;1mface-down\u001b[0m", "\u001b[33;1mface-up\u001b[0m", "\u001b[32;1mdeformed\u001b[0m"]

# -------------------------------------------------

# ------------------- Utilities -------------------

def print_verbose(message, forcePrint=False):
    """Prints message to console incl. elapsed runtime in a uniform style."""
    if opt.verbose or forcePrint: print("-- \033[94m[%.3f s]\033[0m %s --" % ((time.time() - start_time), message))

def clean_filename(filename):
    """Removes path- and extension from passed-in filename."""
    try:
        # filter file-path out of passed in filename, e.g. if the specified file is in a different folder
        path_split = filename.split('/')
        # filter file-ending (.mp4, .jpg, ...) out of filename
        extension_split = path_split[len(path_split)-1].split('.')
        return extension_split[0]
    except ValueError:
        print_verbose("could not clean filename")
        return filename

# source: https://stackoverflow.com/questions/61695773/how-to-set-the-best-value-for-gamma-correction
def gamma_correct(image):
    """Gamma-Corrects an input image and returns the corrected version using the HSV model."""
    # bgr-image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    # compute gamma as log(mid*255) / log(mean)
    mid = 0.5
    mean = np.mean(val)
    gamma = math.log(mid*255)/math.log(mean)
    # gamma correction on value channel
    val_gamma = np.power(val, gamma).clip(0,255).astype(np.uint8)
    # combine corrected value channel with original hue and saturation channels
    hsv_gamma = cv2.merge([hue, sat, val_gamma])
    return cv2.cvtColor(hsv_gamma, cv2.COLOR_HSV2BGR)

def init_net_model():
    """Initializes and returns the OpenCV dnn detectionmodel based on the specified config and trained weights."""
    net = cv2.dnn_DetectionModel((CONFIG_DIR + CONFIG_NAME), (WEIGHTS_DIR + WEIGHTS_NAME))
    net.setInputScale(1/500)
    # set image input size to 960 x 540 (same dimensions that I've trained my weights with)
    net.setInputSize((960,540))    
    # set default backend and target; use DNN_BACKEND_INFERENCE_ENGINE for real-time video annotation (OpenVINO)
    # does not work in this project sadly (bound to VM)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT);
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU);
    print_verbose("initialized detection model")
    return net

# -------------------------------------------------

# ------------------- Cheers! ---------------------

def detect_the_bottle_caps():
    """Reads the input and delegates it to the correct sub-functions for still-frame detection and inference. Saves (and shows) the annotated result."""   
    # check for filepath validity first
    if os.path.exists(opt.video) or os.path.exists(opt.image):
        # ----- Live Video Inference -----
        if(opt.live):
            print_verbose("starting live video inference using %s" % opt.video, forcePrint=True)
            video_inference(opt.video)

        # ----- Image Inference -----
        # Still Frame Detection + Inference / Input Image Inference
        else:
            # either detect a still frame from the video or directly use the passed-in --image file
            video_input = os.path.exists(opt.video)

            if(video_input and opt.high_precision): annotated_image = high_precision_detection(opt.video)
            else:    
                if(video_input): frame_index, still_frame = get_still_frame_from_video(opt.video)
                else: still_frame = cv2.imread(opt.image)

                # start inference on the detected/supplied still frame
                print_verbose("inference started")
                classIds, confidences, boxes = image_inference(still_frame)
                print_verbose("inference ended")

                # look for a region of interest, if specified
                # setup blank roi, either unused or overwritten if --find-roi is set
                roi = (0,0,0,0)
                if (opt.find_roi):
                    print_verbose("region of interest search started")
                    roi = get_region_of_interest(still_frame)
                    print_verbose("region of interest search ended")
                
                # annotate still_frame using the detected classIds, confidences and box-positions
                print_verbose("annotation started")
                annotated_image = annotate_image(still_frame, classIds, confidences, boxes, roi)
                print_verbose("annotation ended")

                # create output csv file
                print_verbose("csv-file generator started")
                csv_name = "/" + clean_filename(opt.video) + ".csv"
                with open(opt.output + csv_name, 'w') as csv_file:
                    csv_writer(csv_file, classIds, boxes, roi, frame_index)
                print_verbose("csv-file generator ended")

            # save the annotated result in the annotated output folder
            filename = opt.video if (video_input) else opt.image
            filename = clean_filename(filename)
            output_filepath = opt.output + ANNOTATED_OUTPUT_DIR + filename + "_annotated.jpg"
            cv2.imwrite(output_filepath, annotated_image)
            print_verbose("successfully saved an annotated frame [%s]" % output_filepath, forcePrint=True)

            # directly show the results if --show-result is being used
            if(opt.show_result):
                cv2.imshow("Cheers - Annotated Output", annotated_image)
                cv2.waitKey()
    else:
        print("Please specify a correct file path to your video/image file.")

def detect_the_bottle_caps_competition():
    video_path = sys.argv[1]
    output = sys.argv[2]
    # detect the still frame
    frame_index, still_frame = get_still_frame_from_video(video_path)
    # start inference on the detected/supplied still frame
    print_verbose("inference started")
    classIds, confidences, boxes = image_inference(still_frame)
    print_verbose("inference ended")

    # look for a region of interest, if specified
    # setup blank roi, either unused or overwritten if --find-roi is set
    roi = (0,0,0,0)
    print_verbose("region of interest search started")
    roi = get_region_of_interest(still_frame)
    print_verbose("region of interest search ended")
    
    # create output csv file
    print_verbose("csv-file generator started")
    csv_name = "/" + clean_filename(video_path) + ".csv"
    with open(output + csv_name, 'w') as csv_file:
        csv_writer(csv_file, classIds, boxes, roi, frame_index)
    print_verbose("csv-file generator ended")

def get_still_frame_from_video(video_filepath):
    """Detects and returns a still frame of a given video file."""
    # load OpenCV VideoCapture on specified video
    vid = cv2.VideoCapture(video_filepath)
    print_verbose("opened video file")

    # get total num of frames
    total_frames    = int(cv2.VideoCapture.get(vid, FRAME_COUNT))
    
    # calculate start and end-frame for further inspection based on the total amount of frames
    start_frame     = int(total_frames*START_FRAME_FACTOR)
    end_frame       = int(total_frames*END_FRAME_FACTOR)
    fps             = int(vid.get(cv2.CAP_PROP_FPS))
    print_verbose("calculated frame range")

    # "manually" read and store first frame, so that frame i+1 can check against last_frame (= frame i)
    vid.set(FRAME_POSITION, start_frame)
    ret, last_frame = vid.read()
    print_verbose("grabbed first frame")    
    
    # pre-set detected-still frame to the initially read frame and its absolute difference to the previous one (infinity, since there is no previous frame)
    detected_still_frame = last_frame
    min_absdiff = float('inf')
    frame_index = 0

    # check one frame per every two seconds of video
    print_verbose("looking for a still frame")
    for i in range(start_frame+1, end_frame, fps*2):
        # skip to frame i
        vid.set(FRAME_POSITION, i)
        # read still image from video frame, check ret (bool) if frame is correctly read or not
        ret, frame = vid.read()
        # calculate absolute difference between frame (x-1) and (x)
        diff = cv2.absdiff(last_frame, frame)
        # overwrite detected_still_frame if new lowest difference has been found
        if(diff.sum() < min_absdiff):
            detected_still_frame = frame
            frame_index = i
            min_absdiff = diff.sum()
        # set frame (x-1) to (x) for the next iteration
        last_frame = frame
    print_verbose("detected a still frame")    
    
    # save the detected still frame if --save-stillframe is being used
    if(opt.save_stillframe):
        filename = opt.output + STILLFRAME_OUTPUT_DIR + f'{clean_filename(video_filepath)}_stillframe.jpg'  
        cv2.imwrite(filename, detected_still_frame)
        print_verbose("saved still frame")

    return frame_index, detected_still_frame

def get_region_of_interest(image):
    """Detects a region of interest within the passed-in image and returns the region of interests' (x,y,w,h). 
    Source: https://answers.opencv.org/question/230859/opencv-does-not-detect-the-my-rectangle/"""
    h, w, _ = image.shape
    image_pixels = w*h
    roi_min = image_pixels * ROI_MIN_SIZE
    roi_max = image_pixels * ROI_MAX_SIZE
    
    # convert to grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    edged = cv2.Canny(image, 150, 350)

    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(edged, 255, 1, 1, 11, 2)
    thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    # apply some dilation and erosion to join the gaps - change iteration to detect more or less area's
    thresh = cv2.dilate(thresh,None,iterations = 20)
    thresh = cv2.erode(thresh,None,iterations = 20)

    # Find the contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # find plausible contour for roi
    # region of interests' x,y coordinate, followed by width and height
    roi = (0,0,0,0)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        # check if size fits within region-of-interest profile
        if (w*h) >= roi_min and (w*h) <= roi_max:
            roi = (x,y,w,h)
            if (opt.show_roi): 
                cv2.rectangle(image, (x,y), (x+w,y+h), (30,30,30), 4)
                break
    print_verbose("detected region of interest at position (%s, %s) with a size of (%s, %s)" % (roi[0], roi[1], roi[2], roi[3]))
    return roi

def high_precision_detection(video_filepath):
    """Detects a still frame of a given video file by checking every couple frames of footage and running inference on every generated frame + comparing results for higher-precision output."""
    # load OpenCV VideoCapture on specified video
    vid = cv2.VideoCapture(video_filepath)
    print_verbose("opened video file")

    # init detection model
    net = init_net_model()

    # get total num of frames
    total_frames    = int(cv2.VideoCapture.get(vid, FRAME_COUNT))
    start_frame     = int(total_frames*START_FRAME_FACTOR)
    end_frame       = int(total_frames*END_FRAME_FACTOR)
    fps             = int(vid.get(cv2.CAP_PROP_FPS))
    frameskip       = 5

    # "manually" read and store first frame, so that frame i+1 can check against last_frame (= frame i)
    vid.set(FRAME_POSITION, start_frame)
    ret, last_frame = vid.read()
    print_verbose("grabbed first frame")    
    
    # pre-set best_frame, inference-results and object_count for comparing/overwriting
    best_frame = last_frame
    best_classIds = best_confidences = best_boxes = []
    object_count = 0

    print_verbose("looking for a best-case frame")
    for i in range(start_frame+1, end_frame, frameskip):
        # skip to frame i
        vid.set(FRAME_POSITION, i)
        # read still image from video frame, check ret (bool) if frame is correctly read or not
        ret, frame = vid.read()
        # run inference on gamma-corrected frame
        classIds, confidences, boxes = net.detect(gamma_correct(frame), opt.conf, opt.nms)
        # overwrite best-frame if a new maximum of objects has been detected
        if(len(boxes) > object_count):
            best_frame = frame
            best_classIds = classIds
            best_confidences = confidences
            best_boxes = boxes
            object_count = len(boxes)
    print_verbose("found best-case frame")
    
    if(opt.save_stillframe):
        filename = opt.output + STILLFRAME_OUTPUT_DIR + f'{clean_filename(video_filepath)}_stillframe.jpg'  
        cv2.imwrite(filename, best_frame)
        print_verbose("saved still frame")  

    print_verbose("annotation started")
    # set empty roi
    roi = (0,0,0,0)
    annotate_image(best_frame, best_classIds, best_confidences, best_boxes, roi)
    print_verbose("annotation ended")

    return best_frame

def image_inference(image, net=None, gammaCorrect=True):
    """Runs inference on an input-image to detect all bottle caps per class. Returns an annotated result image with bounding boxes around detected bottle caps."""
    # gamma correct input image
    if gammaCorrect: image = gamma_correct(image)    

    # init neural net-model using my yolov4 config and self-trained weights; skip this step if video-inference passes in a net
    if net is None:
        net = init_net_model()    
    
    print_verbose("starting bottle cap detection")
    # run opencv's detect function on the passed-in image, using the specified confidence and non-max-suppression threshs
    classIds, confidences, boxes = net.detect(image, opt.conf, opt.nms)
    
    # get amount of distinct class objects
    classList = list(classIds)
    bottlecaps_facedown = classList.count(0)
    bottlecaps_faceup = classList.count(1)
    bottlecaps_deformed = classList.count(2)
    print_verbose("detected %s bottle caps in total [%i %s, %i %s, %i %s" % (len(boxes), bottlecaps_facedown, CLASS_NAMES_COLORED[0], bottlecaps_faceup, CLASS_NAMES_COLORED[1], bottlecaps_deformed, CLASS_NAMES_COLORED[2]))
    
    return classIds, confidences, boxes

def annotate_image(image, classIds, confidences, boxes, roi):
    """Annotates a given image using the detected classIds, confidences and box-positions."""
    classList = list(classIds)
    # save each class-total in cap_counts, with class 0 = facedown, 1 = faceup, 2 = deformed
    cap_counts = [classList.count(0), classList.count(1), classList.count(2)]

    roi_defined = (roi != (0,0,0,0))

    for box in range(0, len(boxes)):
        x, y = boxes[box][0:2]
        w, h = boxes[box][2:4]
        # check if detected bounding box lies within detected roi, if roi was specified
        # allow bottle caps that lie on the region-of-interest bounding box; if you don't want to count those, check for x+w > and y+h >
        if (roi_defined):
            if (x < roi[0] or y < roi[1] or x > roi[0]+roi[2] or y > roi[1]+roi[3]): 
                print_verbose("annotation: ignored bottle cap outside of region of interest at position (%s,%s)" % (x,y))
                # remove out-of-bounds cap from class-cap-count
                cap_counts[classIds[box][0]] -= 1
                continue

        color = CLASS_COLORS[classIds[box][0]]
        # draw bounding box
        cv2.rectangle (image, (x,y), (x+w, y+h), color, ANNOTATION_LINE_THICKNESS)
        # draw info header
        header_height = 16
        cv2.rectangle (image, (x,y), (x+w, y+header_height), color, -1)
        cv2.putText(image, CLASS_NAMES[classIds[box][0]], (x,y+header_height-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
        # draw confidence, if --show-conf is being used
        if (opt.show_conf): cv2.putText(image, "%.2f" % confidences[box], (x,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        print_verbose("annotation: bottle cap %s at position (%s,%s) with confidence %.2f" % (CLASS_NAMES_COLORED[classIds[box][0]], x, y, confidences[box]))

    # -- draw top-left verbose info-output --    
    # draw transparent rectangle for better text-readability (source, but adapted by me: https://stackoverflow.com/questions/56472024/how-to-change-the-opacity-of-boxes-cv2-rectangle)
    x, y, w, h = 0, 0, 426, 150
    sub_img = image[y:y+h, x:x+w]
    black_rect = np.ones(sub_img.shape, dtype=np.uint8) * 0
    res = cv2.addWeighted(sub_img, 0.25, black_rect, 0.5, 1.0)
    image[y:y+h, x:x+w] = res
    
    # draw text info and separator line (between total amount and classes)
    cv2.putText(image, str(cap_counts[0]) + " bottle caps %s" % CLASS_NAMES[0], (0,25), cv2.FONT_HERSHEY_DUPLEX, 1, CLASS_COLORS[0], 2)
    cv2.putText(image, str(cap_counts[1]) + " bottle caps %s" % CLASS_NAMES[1], (0,60), cv2.FONT_HERSHEY_DUPLEX, 1, CLASS_COLORS[1], 2)
    cv2.putText(image, str(cap_counts[2]) + " bottle caps %s" % CLASS_NAMES[2], (0,95), cv2.FONT_HERSHEY_DUPLEX, 1, CLASS_COLORS[2], 2)
    cv2.putText(image, str(sum(cap_counts)) + " bottle caps in total", (0,140), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 2)
    cv2.line(image, (0, 108), (424, 108), (175, 175, 175), 2)
    
    return image

def video_inference(video):
    """Reads an input video and runs inference frame-by-frame, immediately showing the results."""
    # read video file
    vid = cv2.VideoCapture(video)
    total_frames = int(cv2.VideoCapture.get(vid, FRAME_COUNT))

    # init detection model
    net = init_net_model()

    # use every 1+LIVE_FRAMESKIP frame to read and run inference on
    for i in range(0, total_frames, 1+LIVE_FRAMESKIP):
        print_verbose("skipped to frame %i" % i)
        # skip to frame i
        vid.set(FRAME_POSITION, i)
        # read still image from video frame, check ret (bool) if frame is correctly read or not
        ret, frame = vid.read()
        print_verbose("read frame %i" % i)
        # if frame is correctly read, run inference and immediately show the result; skip gamma correction for faster output
        if (ret):
            annotated_frame = image_inference(frame, net, False)
            cv2.imshow('Cheers - Video Inference', annotated_frame)
        # press ESC to quit prematurely
        if cv2.waitKey(1) == 27:
            break

def csv_writer(csv_file, classIds, boxes, roi, frame_index):
    # setup csv writer object
    writer = csv.writer(csv_file, quoting=csv.QUOTE_NONNUMERIC)
    # check if a valid roi has been found and passed into this function
    roi_defined = (roi != (0,0,0,0))

    for box in range(0, len(boxes)):
        x, y = boxes[box][0:2]
        w, h = boxes[box][2:4]
        # check if detected bounding box lies within detected roi, if roi was specified
        # allow bottle caps that lie on the region-of-interest bounding box; if you don't want to count those, check for x+w > and y+h >
        if (roi_defined):
            if (x < roi[0] or y < roi[1] or x > roi[0]+roi[2] or y > roi[1]+roi[3]): 
                print_verbose("annotation: ignored bottle cap outside of region of interest at position (%s,%s)" % (x,y))
                continue
            
        row_content = [frame_index, x, y, CSV_CLASS_NAMES[classIds[box][0]]]
        writer.writerow(row_content)
        
if __name__ == "__main__":
    # store start-time to calculate relative run-time in print-messages
    start_time = time.time()


    # init argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true',                                       help='print verbose info during runtime')
    parser.add_argument('--video' , type=str, default='',                                       help='input video file to extract a still frame and scan for bottle caps')
    parser.add_argument('--image' , type=str, default='',                                       help='input image file to directly scan for bottle caps')
    parser.add_argument('--output' , type=str, default='./output',                              help='output folder for still frames, annotated results and the csv-file')
    parser.add_argument('--conf', type=float, default=CONF_THRESHOLD,                           help='object confidence threshold')
    parser.add_argument('--nms', type=float, default=NON_MAX_SUPPRESSION_THRESHOLD,             help='non max suppression threshold')
    parser.add_argument('--show-result', action='store_true',                                   help='display resulting frame after annotation')
    parser.add_argument('--show-roi', action='store_true',                                      help='display detected region of interest')
    parser.add_argument('--show-conf', action='store_true',                                     help='display the confidence values within the annotated result')
    parser.add_argument('--find-roi', action='store_true',                                      help='find a region of interest to only detect bottle caps within the found roi')
    parser.add_argument('--save-stillframe', action='store_true',                               help='save detected still-frames in the corresponding subfolder')
    parser.add_argument('--live', action='store_true',                                          help='live-annotation of the specified input video, frame-by-frame')
    parser.add_argument('--high-precision', action='store_true',                                help='enables comparatively slow, but more precise still-frame-detection and inference')
    
    if COMPETITION_MODE:
        parser.add_argument('video', metavar='V', type=str, help="path to video-file, used for detecting all bottle caps")
        parser.add_argument('output', metavar='O', type=str, help='output folder')

    opt = parser.parse_args()

    # check for "competition mode", where only sys.argv is parsed and the argument parser is skipped
    if (len(sys.argv) > 1):
        # if the first argument is a valid file path, assume that the argument parser should be skipped
        # otherwise, the first argument would be --video e.g.
        if os.path.exists(sys.argv[1]):
            print("competition mode")
            detect_the_bottle_caps_competition()
        elif(opt.video or opt.image):
            detect_the_bottle_caps()

    else: 
        print("\033[93m-- Please specify some --video or --image file for detection. Take a look at all possible input arguments. --\033[0m\n")
        parser.print_help()

# -------------------------------------------------