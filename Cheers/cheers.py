import sys
sys.path.insert(0, "./yolo_dependencies")
sys.path.insert(0, "./yolov5")
import os
import cv2
import time
import argparse
import numpy as np

DEBUG = False
FRAME_POSITION = cv2.CAP_PROP_POS_FRAMES
FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT
START_FRAME_FACTOR = 1/3
END_FRAME_FACTOR = 2/3
STILLFRAME_FOLDER = './Output/Still Frames/'
ANNOTATED_OUTPUT_FOLDER = './Output/Annotated Frames/'

NON_MAX_SUPRESSION_THRESHOLD = 0.5
CONF_THRESHOLD = 0.5

CLASS_COLORS = [(255, 101, 59), (19, 201, 225), (53, 159, 7)]


def detect_the_bottle_caps():
    if os.path.exists(opt.input) or os.path.exists(opt.input_img):
        # ----- Live Video Inference
        if(opt.video_inference):
            if DEBUG: print("-- starting live video inference using %s --" % opt.input)
            video_inference(video=opt.input)

        # ----- Image Inference
        # ----- Still Frame Detection + Inference / Input Image Inference
        else:
            should_detect_stillframe = os.path.exists(opt.input)
            if(should_detect_stillframe): still_frame = get_still_frame_from_video(opt.input)
            else: still_frame = cv2.imread(opt.input_img)
            if DEBUG: print("-- inference started after %s seconds --" % (time.time() - start_time))
            annotated_image = image_inference(still_frame)
            if DEBUG: print("-- inference ended after %s seconds --" % (time.time() - start_time))
            if(opt.view_img):
                cv2.imshow("Cheers - Annotated Output", annotated_image)
                cv2.waitKey()
            filename = opt.input if (should_detect_stillframe) else opt.input_img
            output_filepath = ANNOTATED_OUTPUT_FOLDER + filename + "_annotated.jpg"
            cv2.imwrite(output_filepath, annotated_image)
            if DEBUG: print("-- saved annotated output image after %s seconds --" % (time.time() - start_time))
    else:
        print("Please specify a correct file path to your video/image file.")


def get_still_frame_from_video(video_filepath):
    # load OpenCV VideoCapture on specified video
    vid = cv2.VideoCapture(video_filepath)
    if DEBUG: print("-- opened video after %s seconds --" % (time.time() - start_time))
    # get total num of frames
    total_frames    = int(cv2.VideoCapture.get(vid, FRAME_COUNT))
    # calculate start and end-frame for further inspection based on the total amount of frames
    # in this case: inspection starts at 1/4th of the videos length and ends at 3/4ths, skipping 50% of "unnecessary" video footage
    start_frame     = int(total_frames*START_FRAME_FACTOR)
    end_frame       = int(total_frames*END_FRAME_FACTOR)
    fps             = int(vid.get(cv2.CAP_PROP_FPS))

    if DEBUG: print("-- calculated frame range after %s seconds --" % (time.time() - start_time))
    # "manually" read and store first frame, so that frame i+1 can check against last_frame (= frame i)
    vid.set(FRAME_POSITION, start_frame)
    ret, last_frame = vid.read()
    if DEBUG: print("-- grabbed first frame after %s seconds --" % (time.time() - start_time))    
    
    detected_still_frame = last_frame
    min_absdiff = float('inf')

    # check one frame per every two seconds of video

    for i in range(start_frame+1, end_frame, fps*2):
        # skip to frame i
        vid.set(FRAME_POSITION, i)
        # read still image from video frame, check ret (bool) if frame is correctly read or not
        ret, frame = vid.read()

        diff = cv2.absdiff(last_frame, frame)
        if(diff.sum() < min_absdiff):
            detected_still_frame = frame
            min_absdiff = diff.sum()

        last_frame = frame
    
    if DEBUG: print("-- detected still frame after %s seconds --" % (time.time() - start_time))    
    
    if(opt.save_stillframe):
        filename = STILLFRAME_FOLDER + f'{video_filepath}_stillframe.jpg'  
        cv2.imwrite(filename, detected_still_frame)
        if DEBUG: print("-- saved still frame after %s seconds --" % (time.time() - start_time))

    return detected_still_frame

def image_inference(image):
    net = cv2.dnn_DetectionModel('./YOLOv4/Config/mpluis2s_yv4.cfg', './YOLOv4/Weights/mpluis2s_yv4.weights')
    net.setInputScale(1/500)
    net.setInputSize((960,540))
    

    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT);
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU);

    classIds, confidences, boxes = net.detect(image, opt.conf_thres, opt.nms_thres)

    # get amount of distinct class objects
    classList = list(classIds)
    bottlecaps_facedown = classList.count(0)
    bottlecaps_faceup = classList.count(1)
    bottlecaps_deformed = classList.count(2)

    for box in range(0, len(boxes)):
        x, y = boxes[box][0:2]
        w, h = boxes[box][2:4]
        color = CLASS_COLORS[classIds[box][0]]
        cv2.rectangle (image, (x,y), (x+w, y+h), color, 2)
    
    # draw transparent rectangle for better text-readability (source, but adapted by me: https://stackoverflow.com/questions/56472024/how-to-change-the-opacity-of-boxes-cv2-rectangle)
    x, y, w, h = 0, 0, 426, 150
    sub_img = image[y:y+h, x:x+w]
    black_rect = np.ones(sub_img.shape, dtype=np.uint8) * 0
    res = cv2.addWeighted(sub_img, 0.25, black_rect, 0.5, 1.0)
    # Putting the image back to its position
    image[y:y+h, x:x+w] = res
    # draw actual text
    cv2.putText(image, str(bottlecaps_facedown) + " bottle caps face-down", (0,25), cv2.FONT_HERSHEY_DUPLEX, 1, CLASS_COLORS[0], 2)
    cv2.putText(image, str(bottlecaps_faceup) + " bottle caps face-up", (0,60), cv2.FONT_HERSHEY_DUPLEX, 1, CLASS_COLORS[1], 2)
    cv2.putText(image, str(bottlecaps_deformed) + " bottle caps deformed", (0,95), cv2.FONT_HERSHEY_DUPLEX, 1, CLASS_COLORS[2], 2)
    # draw separator line
    cv2.line(image, (0, 108), (424, 108), (175,175,175), 2)
    cv2.putText(image, str(bottlecaps_facedown + bottlecaps_faceup + bottlecaps_deformed) + " bottle caps in total", (0,140), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 2)

    return image

def video_inference(video):
    cap = cv2.VideoCapture(video)
    while True:
        flag, frame = cap.read()
        if flag:
            # The frame is ready and already captured
            annotated_frame = image_inference(frame)
            cv2.imshow('video', annotated_frame)
            pos_frame = cap.get(FRAME_POSITION)

        if cv2.waitKey(1) == 27:
            break
        if cap.get(FRAME_POSITION) == cap.get(FRAME_COUNT):
            # If the number of captured frames is equal to the total number of frames,
            # we stop
            break



if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='print some debug info during runtime')
    parser.add_argument('--input' , type=str, default='', help='input video file to scan for bottle caps')
    parser.add_argument('--input-img' , type=str, default='', help='input image file to directly scan for bottle caps')
    parser.add_argument('--conf-thres', type=float, default=CONF_THRESHOLD, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=NON_MAX_SUPRESSION_THRESHOLD, help='non max suppression threshold')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-stillframe', action='store_true', help='save detected still-frames in the corresponding subfolder')
    parser.add_argument('--video-inference', action='store_true', help='use the video input and directly run it through yolov5')
    opt = parser.parse_args()
    
    DEBUG = opt.debug
    
    if(opt.input or opt.input_img):
        detect_the_bottle_caps()
    print("-- total runtime: %s seconds --" % (time.time() - start_time))
