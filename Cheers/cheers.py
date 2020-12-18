import sys
sys.path.insert(0, "./yolo_dependencies")
sys.path.insert(0, "./yolov5")
import os
import cv2
import time
import argparse
from yolov5 import detect

DEBUG = False
FRAME_POSITION = cv2.CAP_PROP_POS_FRAMES
FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT
START_FRAME_FACTOR = 1/3
END_FRAME_FACTOR = 2/3
STILLFRAME_FOLDER = './detected_still_frames/'


def detect_the_bottle_caps():
    if os.path.exists(opt.input) or os.path.exists(opt.input_img):
        if(opt.raw_video_input):
            opt.source = opt.input
            save_img = True
            detect.detect(opt, save_img)
        elif(os.path.exists(opt.input_img)):
            still_frame = get_still_frame_from_video(opt.input)
            opt.source = cv2.imread(opt.input_img)
            save_img = True
            if DEBUG: print("-- yolov5 detection started after %s seconds --" % (time.time() - start_time))
            detect.detect(opt, save_img)
            if DEBUG: print("-- yolov5 detection ended after %s seconds --" % (time.time() - start_time))
        elif(os.path.exists(opt.input)):
            still_frame = get_still_frame_from_video(opt.input)
            opt.source = still_frame
            save_img = True
            if DEBUG: print("-- yolov5 detection started after %s seconds --" % (time.time() - start_time))
            detect.detect(opt, save_img)
            if DEBUG: print("-- yolov5 detection ended after %s seconds --" % (time.time() - start_time))
    else:
        print("Please specify a correct file path to your video file.")


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
    filename = STILLFRAME_FOLDER + f'{video_filepath}_dsf.jpg'  
    cv2.imwrite(filename, detected_still_frame)
    if DEBUG: print("-- saved still frame after %s seconds --" % (time.time() - start_time))

    return filename



if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='print some debug info during runtime')
    parser.add_argument('--input-img' , type=str, default='', help='input image file to directly pass to yolo and scan for bottle caps')
    parser.add_argument('--input' , type=str, default='', help='input video file to scan for bottle caps')
    parser.add_argument('--weights', nargs='+', type=str, default='./weights/v5s_7000epochs.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=960, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.8, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-dir', type=str, default='output', help='directory to save results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--raw-video-input', action='store_true', help='use the video input and directly run it through yolov5')
    opt = parser.parse_args()
    
    DEBUG = opt.debug
    
    if(opt.input):
        detect_the_bottle_caps()
    print("-- total runtime: %s seconds --" % (time.time() - start_time))
