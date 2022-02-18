# Deploy program for the raspberry pi
# Usage: python3 bw-deploy.py [tflite model location]
# Use requirements.txt in this directory

import argparse
import time
import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
from threading import Thread

#start copy
# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True

# end copy

# from https://stackoverflow.com/questions/65824714/process-output-data-from-yolov5-tflite
# for Yolov5 output parsing
def classFilter(classdata):
    classes = []  # create a list
    for i in range(classdata.shape[0]):         # loop through all predictions
        classes.append(classdata[i].argmax())   # get the best classification location
    return classes  # return classes (int)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #tflite model .tflite file
    parser.add_argument(
        '-m',
        '--model',
        default=None,
        help='tflite model to use'
    )
    #labels.txt file assuming it lists all labels, one per line
    parser.add_argument(
        '-l',
        '--labelfile',
        default='labels.txt',
        help='file containing labels'
    )
    parser.add_argument(
        '-t',
        '--threshold',
        default=0.1,
        help='minimum confidence for displaying an object'
    )
    parser.add_argument(
        '-r',
        '--resolution',
        default='??x??',
        help='desired camera resolution (assuming camera supports it)'
    )

    args = parser.parse_args()
    min_threshold = float(args.threshold)
    cam_w, cam_h = args.resolution.split('x')
    cam_w = int(cam_w)
    cam_h = int(cam_h)

    print("Deploy start...")
    print("Using: ")
    print("     - model: " + str(args.model))
    print("     - labelfile: "+str(args.labelfile))
    print("     - threshold: "+str(args.threshold))
    print("     - resolution: "+str(args.resolution))

    #get labels
    labels = list()
    with open(args.labelfile, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    #load model
    interpreter = tflite.Interpreter(model_path=args.model)

    interpreter.allocate_tensors()

    #model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    
    in_height = input_details[0]['shape'][1]
    in_width = input_details[0]['shape'][2]

    floating_model = (input_details[0]['dtype'] == np.float32)

    if floating_model:
        print("type: floating model...")

    input_mean = 127.5
    input_std = 127.5

    # Initialize frame rate calculation
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()

    print("processed inputs...")

    videostream = VideoStream(resolution=(cam_w,cam_h), framerate=30).start()
    print("starting video stream...")
    time.sleep(1)

    # Create window
    cv2.namedWindow('Object detector', cv2.WINDOW_NORMAL)

    #loop infinitely and detect within an image
    try:
        while True:
            #Below based on: EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi
            #and https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py#L117
            # Start timer (for calculating frame rate)
            t1 = cv2.getTickCount()
            print(str(t1) + " reading new image...")

            # Grab frame from video stream
            frame1 = videostream.read()

            # Acquire frame and resize to expected shape [1xHxWx3]
            frame = frame1.copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (in_width, in_height))
            input_data = np.expand_dims(frame_resized, axis=0)

            # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
            if floating_model:
                input_data = (np.float32(input_data) - input_mean) / input_std

            # Perform the actual detection by running the model with the image as input
            interpreter.set_tensor(input_details[0]['index'],input_data)
            interpreter.invoke()

            # Retrieve detection results
            
            #yolo does not output boxes, classes, and scores separately like most other models
            #it creates a table instead that we'll parse to get the same information
            #source: https://stackoverflow.com/questions/65824714/process-output-data-from-yolov5-tflite
            out_table = interpreter.get_tensor(output_details[0]['index'])[0]
        
            #output is [x y w h conf class0, class1, ...]
            boxes = np.squeeze(out_table[..., :4])
            # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
            x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3] #xywh
            xyxy = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]  # xywh to xyxy   [4, 25200]
            scores = np.squeeze(out_table[..., 4:5])
            classes = classFilter(out_table[..., 5:])

            # Loop over all detections and draw detection box if confidence is above minimum threshold
            for i in range(len(scores)):
                if ((scores[i] > min_threshold) and (scores[i] <= 1.0)):

                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                    ymin = int(max(1,(xyxy[1][i] * cam_h)))
                    xmin = int(max(1,(xyxy[0][i] * cam_w)))
                    ymax = int(min(cam_h,(xyxy[3][i] * cam_h)))
                    xmax = int(min(cam_w,(xyxy[2][i] * cam_w)))

                    # optionally print bounding boxes (for debugging)
                    #print("("+str(xmin)+","+str(ymin)+"), ("+str(xmax)+","+str(ymax)+")")
                    
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                    # Draw label
                    object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                    label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                    # also print output to the console (useful when boxes overlap)
                    print(str(object_name) + ":" + str(int(scores[i]*100)) + "%")
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

            # Draw framerate in corner of frame
            cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

            # All the results have been drawn on the frame, so it's time to display it.
            cv2.imshow('Object detector', frame)

            # Calculate framerate
            t2 = cv2.getTickCount()
            time1 = (t2-t1)/freq
            frame_rate_calc= 1/time1

            # Press 'q' to quit
            if cv2.waitKey(1) == ord('q'):
                break

        #handle if we break as well
        cv2.destroyAllWindows()
        videostream.stop()
    
    except KeyboardInterrupt:
        #cleanly handle and end
        cv2.destroyAllWindows()
        videostream.stop()





