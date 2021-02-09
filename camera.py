import cv2
import YOLO_V3 as yolo
import time

print(cv2.__version__)
print("test")
def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

# load yolov3 model
model = yolo.load_model('/Users/evantanuwijaya/Documents/Code/model.h5')
# define the expected input shape for the model
input_w, input_h = 416, 416

cv2.namedWindow("Test Camera")
vc = cv2.VideoCapture(1)

if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False

frame_rate = 1
prev = 0

while rval:
    time_elapsed = time.time() - prev
    # Read from camera
    rval, frame = vc.read()
    if time_elapsed > 1./frame_rate:
        prev = time.time()
        frame = rescale_frame(frame, percent=50)
        # define our new photo
        photo_filename = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # load and prepare image
        image, image_w, image_h = yolo.load_image_pixels(photo_filename, (input_w, input_h))
        # make prediction
        yhat = model.predict(image)
        # summarize the shape of the list of arrays
        # print([a.shape for a in yhat])
        # define the anchors
        #anchors = []
        anchors = [[116, 90, 156, 198, 373, 326], [
            30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]
        # define the probability threshold for detected objects
        class_threshold = 0.7
        boxes = list()
        for i in range(len(yhat)):
            # decode the output of the network
            boxes += yolo.decode_netout(yhat[i][0], anchors[i],
                                class_threshold, input_h, input_w)
        # correct the sizes of the bounding boxes for the shape of the image
        yolo.correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
        # suppress non-maximal boxes
        yolo.do_nms(boxes, 0.7)
        # define the labels
        labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
                "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
                "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
                "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
                "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
                "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
                "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
                "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
        # get the details of the detected objects
        v_boxes, v_labels, v_scores = yolo.get_boxes(boxes, labels, class_threshold)

        # draw what we found
        yolo.draw_boxes(frame, v_boxes, v_labels, v_scores)

    key = cv2.waitKey(20)
    
    if key == 27:
        break

vc.release()
cv2.destroyWindow("Test Camera")
