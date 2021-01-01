import sys
import os
import cv2
import time
import math
import argparse
import numpy as np

VERBOSE = False
FRAME_POSITION = cv2.CAP_PROP_POS_FRAMES
FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT
START_FRAME_FACTOR = 1/3
END_FRAME_FACTOR = 2/3
LIVE_FRAMESKIP = 0

STILLFRAME_FOLDER = './Output/Still Frames/'
ANNOTATED_OUTPUT_FOLDER = './Output/Annotated Frames/'

CONFIG_DIR = './YOLOv4/Config/'
CONFIG_NAME = 'v4tiny_9000.cfg'

WEIGHTS_DIR = './YOLOv4/Weights/'
WEIGHTS_NAME = 'v4tiny_9000.weights'

NON_MAX_SUPRESSION_THRESHOLD = 0.5
CONF_THRESHOLD = 0.5
ANNOTATION_LINE_THICKNESS = 4


CLASS_COLORS = [(255, 101, 59), (19, 201, 225), (53, 159, 7)]
CLASS_NAMES = ["face-down", "face-up", "deformed"]

def print_verbose(message, forcePrint=False):
    if VERBOSE or forcePrint: print("-- [%.3f s] %s --" % ((time.time() - start_time), message))

def clean_filename(filename):
    try:
        # filter file-path out of passed in filename, e.g. if the specified file is in a different folder
        path_split = filename.split('/')
        # filter file-ending (.mp4, .jpg, ...) out of filename
        extension_split = path_split[len(path_split)-1].split('.')
        return extension_split[0]
    except ValueError:
        print_verbose("could not clean filename")
        return filename

def detect_the_bottle_caps():
    if os.path.exists(opt.video) or os.path.exists(opt.image):
        # ----- Live Video Inference -----
        if(opt.live):
            print_verbose("starting live video inference using %s" % opt.video, forcePrint=True)
            video_inference(opt.video)

        # ----- Image Inference -----
        # Still Frame Detection + Inference / Input Image Inference
        else:
            # either detect a still frame from the video or directly use the passed-in --image file
            should_detect_stillframe = os.path.exists(opt.video)
            if(should_detect_stillframe): still_frame = get_still_frame_from_video(opt.video)
            else: still_frame = cv2.imread(opt.image)
            
            # start inference on the detected/supplied still frame
            print_verbose("inference started")
            annotated_image = image_inference(still_frame)
            print_verbose("inference ended")

            # save the annotated result in the annotated output folder
            filename = opt.video if (should_detect_stillframe) else opt.image
            filename = clean_filename(filename)
            output_filepath = ANNOTATED_OUTPUT_FOLDER + filename + "_annotated.jpg"
            cv2.imwrite(output_filepath, annotated_image)
            print_verbose("successfully saved an annotated frame [%s]" % output_filepath, forcePrint=True)

            # directly show the results if --show-result is being used
            if(opt.show_result):
                cv2.imshow("Cheers - Annotated Output", annotated_image)
                cv2.waitKey()
    else:
        print("Please specify a correct file path to your video/image file.")


def get_still_frame_from_video(video_filepath):
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
            min_absdiff = diff.sum()
        # set frame (x-1) to (x) for the next iteration
        last_frame = frame
    print_verbose("detected a still frame")    
    
    # save the detected still frame if --save-stillframe is being used
    if(opt.save_stillframe):
        filename = STILLFRAME_FOLDER + f'{video_filepath}_stillframe.jpg'  
        cv2.imwrite(filename, detected_still_frame)
        print_verbose("saved still frame")

    return detected_still_frame

# source: https://stackoverflow.com/questions/61695773/how-to-set-the-best-value-for-gamma-correction
def gamma_correct(image):
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

def image_inference(image):
    # gamma correct input image
    image = gamma_correct(image)    

    # init neural net-model using my yolov4 config and self-trained weights
    net = cv2.dnn_DetectionModel((CONFIG_DIR + CONFIG_NAME), (WEIGHTS_DIR + WEIGHTS_NAME))
    net.setInputScale(1/500)
    net.setInputSize((960,540))    

    # set preferred backend and target to run the inference on
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT);
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU);
    print_verbose("initialized detection model")
    print_verbose("starting bottle cap detection")
    
    # run opencv's detect function on the passed-in image, using the specified confidence and non-max-supression threshs
    classIds, confidences, boxes = net.detect(image, opt.conf_thres, opt.nms_thres)
    
    # get amount of distinct class objects
    classList = list(classIds)
    bottlecaps_facedown = classList.count(0)
    bottlecaps_faceup = classList.count(1)
    bottlecaps_deformed = classList.count(2)
    print_verbose("detected %s bottle caps in total [%i %s, %i %s, %i %s]" % (len(boxes), bottlecaps_facedown, CLASS_NAMES[0], bottlecaps_faceup, CLASS_NAMES[1], bottlecaps_deformed, CLASS_NAMES[2]))
    
    # draw bounding boxes around detected bottle caps
    print_verbose("annotation started")
    for box in range(0, len(boxes)):
        x, y = boxes[box][0:2]
        w, h = boxes[box][2:4]
        color = CLASS_COLORS[classIds[box][0]]
        # draw bounding box
        cv2.rectangle (image, (x,y), (x+w, y+h), color, ANNOTATION_LINE_THICKNESS)
        # draw info header
        header_height = 16
        cv2.rectangle (image, (x,y), (x+w, y+header_height), color, -1)
        cv2.putText(image, CLASS_NAMES[classIds[box][0]], (x,y+header_height-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
        # draw confidence, if --show-conf is being used
        if (opt.show_conf): cv2.putText(image, "%.2f" % confidences[box], (x,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        print_verbose("annotation: bottle cap %s at position (%s,%s) with confidence %.2f" % (CLASS_NAMES[classIds[box][0]], x, y, confidences[box]))
    print_verbose("annotation ended")
    # -- draw top-left verbose info-output --
    
    
    # draw transparent rectangle for better text-readability (source, but adapted by me: https://stackoverflow.com/questions/56472024/how-to-change-the-opacity-of-boxes-cv2-rectangle)
    x, y, w, h = 0, 0, 426, 150
    sub_img = image[y:y+h, x:x+w]
    black_rect = np.ones(sub_img.shape, dtype=np.uint8) * 0
    res = cv2.addWeighted(sub_img, 0.25, black_rect, 0.5, 1.0)
    image[y:y+h, x:x+w] = res
    
    # draw text info and separator line (between total amount and classes)
    cv2.putText(image, str(bottlecaps_facedown) + " bottle caps face-down", (0,25), cv2.FONT_HERSHEY_DUPLEX, 1, CLASS_COLORS[0], 2)
    cv2.putText(image, str(bottlecaps_faceup) + " bottle caps face-up", (0,60), cv2.FONT_HERSHEY_DUPLEX, 1, CLASS_COLORS[1], 2)
    cv2.putText(image, str(bottlecaps_deformed) + " bottle caps deformed", (0,95), cv2.FONT_HERSHEY_DUPLEX, 1, CLASS_COLORS[2], 2)
    cv2.putText(image, str(bottlecaps_facedown + bottlecaps_faceup + bottlecaps_deformed) + " bottle caps in total", (0,140), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 2)
    cv2.line(image, (0, 108), (424, 108), (175,175,175), 2)

    return image

def video_inference(video):
    # read video file
    vid = cv2.VideoCapture(video)
    total_frames = int(cv2.VideoCapture.get(vid, FRAME_COUNT))

    # use every 1+LIVE_FRAMESKIP frame to read and run inference on
    for i in range(0, total_frames, 1+LIVE_FRAMESKIP):
        # skip to frame i
        vid.set(FRAME_POSITION, i)
        # read still image from video frame, check ret (bool) if frame is correctly read or not
        ret, frame = vid.read()
        # if frame is correctly read, run inference and immediately show the result
        if (ret):
            annotated_frame = image_inference(frame)
            cv2.imshow('Cheers - Video Inference', annotated_frame)          
        # press ESC to quit prematurely
        if cv2.waitKey(1) == 27:
            break

if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true',                                       help='print verbose info during runtime')
    parser.add_argument('--video' , type=str, default='',                                       help='input video file to extract a still frame and scan for bottle caps')
    parser.add_argument('--image' , type=str, default='',                                       help='input image file to directly scan for bottle caps')
    parser.add_argument('--conf-thres', type=float, default=CONF_THRESHOLD,                     help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=NON_MAX_SUPRESSION_THRESHOLD,        help='non max suppression threshold')
    parser.add_argument('--show-result', action='store_true',                                   help='display results')
    parser.add_argument('--show-conf', action='store_true',                                     help='display the confidence values in the annotated result')
    parser.add_argument('--save-stillframe', action='store_true',                               help='save detected still-frames in the corresponding subfolder')
    parser.add_argument('--live', action='store_true',                                          help='live-annotation of the specified input video, frame-by-frame')
    opt = parser.parse_args()
    
    VERBOSE = opt.verbose
    
    if(opt.video or opt.image):
        detect_the_bottle_caps()