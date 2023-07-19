import cv2 as cv
import numpy as np
from collections import OrderedDict
import argparse

def read_pretrained_model():
    """
    Read the pretrained MobileNet SSD (Single Shot Detector) model
        
        None

    Return:
        model: the loaded model
        class_names: 90 possible detected classes
    """
    # load the COCO class names
    with open('MobileNet SSD-COCO Model/object_detection_classes_coco.txt', 'r') as f:
        class_names = f.read().split('\n')

    # load the DNN model
    model = cv.dnn.readNet(model='MobileNet SSD-COCO Model/frozen_inference_graph.pb',                        
                           config='MobileNet SSD-COCO Model/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt',
                           framework='TensorFlow')

    return model, class_names


class CentroidTracker():
    """
    A class helps to keep track of objects' apperance and assign unique ID to each object

    REFERENCE:  The CentroidTracker() class was written based on Adrian Rosebrock's blog published on July 23, 2018 and was modified for the purpose of this task.
                Link to Adrian Rosebrock's blog: https://pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
    """
    def __init__(self, maxDisappeared=50):
        # initialize the next unique object ID along with two ordered dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has been marked as "disappeared", respectively
        self.nextObjectID = 1
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

        # store the number of maximum consecutive frames a given object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared


    def calculate_euclidean_dist(self, objectCentroids, inputCentroids):
        """
        Calculate the Euclidean distance between each pair of existing centroids in the previous frames and the detected centroids in 
        the current frame.
        
            objectCentroids: a list of existing centroids in the preivous frames
            inputCentroids: an array of detected centroids in the current frames

        Return:
            D: an array of Euclidean distance between all possible pairs of existing and inputing centroids
        """
        D = []
        
        for obj_centroid in np.array(objectCentroids):
            dist_arr = []
            for input_centroid in inputCentroids:
                dist = np.linalg.norm(obj_centroid - input_centroid)
                dist_arr.append(dist)
            D.append(dist_arr)

        D = np.asarray(D)
        
        return D
    

    def register(self, centroid):
        """
        Register a ID for a new detected object

            centroid: a centroid of a new detected object

        Return:
            None
        """
        # when registering an object we use the next available object ID to store the centroid
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1


    def deregister(self, objectID):
        """
        Deregister for an object that disappears for more than the maximum number allowed an object to disappear

            objectID: an ID of an object

        Return:
            None
        """
        # to deregister an object ID we delete the object ID from both of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]


    def update(self, rects):
        """
        Update the tracking information over time (register new objects/deregister disappered objects/update centroids of current objects)

            rects: Information of the bounding box of the detected object

        Return:
            self.objects: A dictionary with keys as IDs and values as objects' centroids
        """
        # check to see if the list of input bounding box rectangles is empty
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them as disappeared
            for objectID in self.disappeared.keys():
                self.disappeared[objectID] += 1

                # if we have reached a maximum number of consecutive frames where a given object has been marked as missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            # return early as there are no centroids or tracking info to update
            return self.objects

        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        # loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        # if we are currently not tracking any objects take the input centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])

        # otherwise, are are currently tracking objects so we need to try to match the input centroids to existing object centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # compute the distance between each pair of object centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing object centroid
            D = self.calculate_euclidean_dist(np.array(objectCentroids), inputCentroids)

            # in order to perform this matching we must (1) find the smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row with the smallest value as at the *front* of the index list
            rows = D.min(axis=1).argsort()

            # next, we perform a similar process on the columns by finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # in order to determine if we need to update, register, or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()

            # loop over the combination of the (row, column) index tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or column value before, ignore it val
                if row in usedRows or col in usedCols:
                    continue

                # otherwise, grab the object ID for the current row, set its new centroid, and reset the disappeared counter
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                # indicate that we have examined each of the row and column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

            # compute both the row and column index we have NOT yet examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # in the event that the number of object centroids is equal or greater than the number of input centroids
            # we need to check and see if some of these objects have potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    # check to see if the number of consecutive frames the object has been marked "disappeared" for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            # otherwise, if the number of input centroids is greater than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        # return the set of trackable objects
        return self.objects


def estimate_focal_length(est_dist_btw_obj_cam, real_object_width, object_width_in_image):
    """
    Estimate the focal length
        
        est_dist_btw_obj_cam: an estimated distance between the object and camera
        real_object_width: a measured width of detected object in reality
        object_width_in_image: a width of the object displayed in the image (pixels)

    Return:
        focal_length: The estimated focal length
    """
    focal_length = (object_width_in_image * est_dist_btw_obj_cam) / real_object_width
    
    return focal_length


def estimate_distance(focal_length, real_object_width, object_width_in_image):
    """
    Estimate distance between a detected object and the camera

        focal_length: an estimated focal length
        real_object_width: a measured width of detected object in reality
        object_width_in_image: a width of the object displayed in the image (pixels)

    Return:
        distance: the estimated distance between a detected object and the camera
    """
    distance = (real_object_width * focal_length) / object_width_in_image
    
    return distance


def estimate_obj_width_in_pic(cap, model, class_names):
    """
    Estimate the width of an object in a single frame (pixels).

        cap: a video file sequence
        model: a loaded model
        class_names: class names that the loaded model can detect

    Return:
        The estimated width of an object in a single frame (pixels)
    """
    first_time_detected_obj = True
    object_width_first_frame = [] # Widths of all targeted objects detected in the first frame

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break

        image = frame.copy()
        _, image_width, _ = image.shape

        # create blob from image
        blob = cv.dnn.blobFromImage(image=image, size=(300, 300), mean=(104, 117, 123), swapRB=True)
        model.setInput(blob)
        output = model.forward()

        # loop over each of the detections
        for detection in output[0, 0, :, :]:
            # Extract the confidence of the detection
            confidence = detection[2]

            # Get the class_id 
            class_id= detection[1]
            # Map the class id to the class
            class_name = class_names[int(class_id)-1]

            # draw bounding boxes only if the detection confidence os above a certain threshold, else skip
            if class_name == "person" and confidence > 0.4:
                # Get the boudning box coordinates
                x_start = detection[3] * image_width

                # Get the boudning box width
                x_end = detection[5] * image_width
                w = x_end - x_start
                object_width_first_frame.append(w)
                first_time_detected_obj = False # change to False if we found the first frame that contains the target object

        # Break if we have already found the first frame that contains target objectss
        if not first_time_detected_obj:
            break

    object_width_first_frame = np.asarray(object_width_first_frame)

    # Return the median of all detected width
    return np.median(object_width_first_frame)


def task_1(file_name):
    """
    Perform task 1 including extracting moving objects using Gaussian Mixture background modelling, removing noisy detection using morphological operators or 
    majority voting, counting separate moving objects using connected component analysis, and classifying each object (or connected component) into person, 
    car and other by simply using the ratio of width and height of the connected components
        
        file_name: input video file name
    
    Return:
        Output video frame
        The number of objects or connected components 
    """
    # Read video
    cap = cv.VideoCapture(file_name)
    if not cap.isOpened():
        print(f"Unable to open {file_name}")

    # Creates MOG2 Background Subtractor (Gaussian Mixture-based Background/Foreground Segmentation)
    back_ground_sub = cv.createBackgroundSubtractorMOG2()
    
    frame_count = 0 # Number of frames
    KERNEL = np.ones((5,5), np.uint8) # Define a kernel to perform morphological transformations
    CONNECTIVITY = 8 # Connectivity used in the connected component analysis

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Resize the video frame to a size comparable to VGA
        frame_resized = resize_image(frame)

        frame_count += 1

        # Extract foreground (moving pixels)
        foreground_mask = back_ground_sub.apply(frame_resized)
        # Extract background
        background_mask = back_ground_sub.getBackgroundImage()

        # Remove noisy detection using morphological operators, in particular the opening operator
        foreground_mask_noise_rm = cv.morphologyEx(foreground_mask, cv.MORPH_OPEN, KERNEL, iterations=1)
        # Remove noise: if a pixel is less than 250, set it equals to 0
        foreground_mask_noise_rm[np.abs(foreground_mask_noise_rm) < 250] = 0

        # Perform connected component analysis
        output = cv.connectedComponentsWithStats(foreground_mask_noise_rm, CONNECTIVITY, cv.CV_32S)
        numLabels, labels, stats, _ = output

        componentMask = np.uint8(labels)*255
        detected_objects = cv.bitwise_and(frame_resized, frame_resized, mask=componentMask)

        # Classify each object (or connected component) into person, car, and others by simply using the ratio of width and height and area of the connected component
        # RULES TO CLASSIFY OBJECTS (PERSON, CAR, AND OTHERS: 
        #       - CAR: if the ratio between width and height is between 1.30 and 2.30 and the area of the bounding box is greater than or equal to 800
        #       - PERSON: if the ratio between width and height is between 0.60 and 0.90 and the area of the bounding box is greater than or equal to 100
        #       - OTHER: if the ratio does not fall into the above specified range and the area of the boudning box is greater than or equal to 50
        if numLabels == 1:
            # if there is only one label, it means that there is no object/connected component detected in this frame (only the background)
            print(f"Frame {frame_count:04d}: 0 object")
        else:
            total_objects = 0 # Total number of detected objects
            car_counts = 0 # Total number of detected cars
            person_counts = 0 # Total number of detected persons
            others_counts = 0 # Total number of detected other objects

            for i in range (1, numLabels):
                # Get the width, height, and area of the component
                w = stats[i, cv.CC_STAT_WIDTH]
                h = stats[i, cv.CC_STAT_HEIGHT]
                area = stats[i, cv.CC_STAT_AREA]

                # Calcualte the ratio between width and height of the component
                width_height_ratio = round(w/h, 2)

                # Classify objects
                if width_height_ratio >= 1.30 and width_height_ratio <= 2.3 and area >= 800:
                    # If it is classified as a car
                    car_counts += 1
                    total_objects += 1
                elif width_height_ratio >= 0.60 and width_height_ratio <= 0.9 and area >= 100:
                    # If it is classified as a person
                    person_counts += 1
                    total_objects += 1
                else:
                    if area >= 50:
                        # If it is classified as others
                        others_counts += 1
                        total_objects += 1

            # Text processing to display the output in the command window
            total_object_text_display = "object"
            car_text_display = "car"
            person_text_display = "person"
            other_text_display = "other"

            if total_objects > 1:
                total_object_text_display += "s"
                car_text_display += "s" if car_counts > 1 else ""
                person_text_display += "s" if person_counts > 1 else ""
                other_text_display += "s" if others_counts > 1 else ""

            print(f"Frame {frame_count:04d}: {total_objects} {total_object_text_display} ({person_counts} {person_text_display}, {car_counts} {car_text_display}, {others_counts} {other_text_display})")
        
        # Concatenate to display
        first_row = cv.hconcat([frame_resized, background_mask]) # LEFT: the original video frame, RIGHT: the estimated background frame
        foreground_mask = cv.cvtColor(foreground_mask, cv.COLOR_GRAY2BGR)
        second_row = cv.hconcat([foreground_mask, detected_objects]) # LEFT: The detected moving pixels before filtering (in binary mask), RIGHT: detected objects (in the orihinal color)
        stack = cv.vconcat((first_row, second_row))

        cv.imshow("Task 1's Output", stack)

        key = cv.waitKey(20)
        if key == 27:
            break

    cap.release()
    cv.destroyAllWindows()


def person_tracking(file_name):
    """
    Perform task 2 including detecting pedestrians (i.e. persons) using a OpenCV Deep Neural Network (DNN) module and a MobileNet SSD detector pre-trained on 
    the MS COCO dataset, tracking and display the detected pedestrians by providing same labels to the same pedestrians across over times, and selecting up to 
    (3) pedestrians that are most close in space to the camera.

        file_name: input video file name

    Return:
        Output video frame
    """
    # Read video
    cap = cv.VideoCapture(file_name)
    cap_ = cv.VideoCapture(file_name)
    model, CLASS_NAMES = read_pretrained_model()

    # Initialize our centroid tracker and frame dimensions
    ct = CentroidTracker()

    # Assumption distance from camera to a pedestrian measured (in centimeter)
    KNOWN_DISTANCE = 200
    # Assumption width of a pedestrian in the real world (in centimeter). If drawing a box around a person, we can assume that the width of the box equals to the
    # that person's shoulder width. According to https://www.healthline.com/health/average-shoulder-width website, the average shoulder width of American women is 36.7 cm
    # and 41.1 cm is the average for American men's shoulder width. Therefore, let 38.9 which is the average of the average American women and men's shoulder width be the
    # assumption width of a pedestrian in reall world.
    KNOWN_WIDTH = 38.9

    # Estimate a pedestrian width in pixels
    ref_obj_width = estimate_obj_width_in_pic(cap_, model, CLASS_NAMES)
    # Estimate focal length
    focal_length_found = estimate_focal_length(KNOWN_DISTANCE, KNOWN_WIDTH, ref_obj_width)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        image = frame.copy()
        frame_1 = frame.copy()
        frame_2 = frame.copy()
        frame_3 = frame.copy()
        frame_4 = frame.copy()
        
        image_height, image_width, _ = image.shape
        # create blob from image
        blob = cv.dnn.blobFromImage(image=image, size=(300, 300), mean=(104, 117, 123), swapRB=True)
        model.setInput(blob)
        output = model.forward()

        # Intialize necessary variables
        rects = [] # Store location of the bounding boxes of detected pedestrians in each frame
        centroids = [] # Store centroids' location of the bounding boxes of detected pedestrians in each frame
        camera_obj_dist_tracking = [] # Store up to 3 closest distance between pedestrians and the camera
        box_info = [] # Store the locations of the bounding of up to 3 closest pedestrians to the camera

        # loop over each of the detections
        for detection in output[0, 0, :, :]:
            # Extract the confidence of the detection
            confidence = detection[2]

            # Get the class_id 
            class_id= detection[1]
            # Map the class id to the class
            class_name = CLASS_NAMES[int(class_id)-1]

            # Draw bounding boxes only if the detection confidence is above a certain threshold and the detected object is 
            # a person (a.k.a, pedestrians in this assignment), else skip
            if class_name == "person" and confidence > 0.4:
                # Get the boudning box coordinates
                x_start = detection[3] * image_width
                y_start = detection[4] * image_height
                x_end = detection[5] * image_width
                y_end = detection[6] * image_height
                # Get the boudning box width
                w = x_end - x_start
                # h = y_end - y_start

                rects.append(np.array([x_start, y_start, x_end, y_end]))

                # Get the centroid of the bounding box
                cx = int((x_start + x_end) / 2.0)
                cy = int((y_start + y_end) / 2.0)
                
                centroids.append([cx, cy])

                # Estimate the distance between the pedestrians and the camera
                object_dist = estimate_distance(focal_length_found, KNOWN_WIDTH, w)
                
                # If there are less than 3 pedestrians detected in this frame, store the estimated tracking distance and the location of
                # that person. Else, compare the distance of the current detected pedestrians to the longest distance being tracked in the
                # camera_obj_dist_tracking list. If the distance of the current detected pedestrians is less than the longest distance being 
                # tracked in the camera_obj_dist_tracking list, update the list.
                if len(camera_obj_dist_tracking) < 3:
                    camera_obj_dist_tracking.append(object_dist)
                    box_info.append([x_start, y_start, x_end, y_end, cx, cy])
                else:
                    # it is possible to have more than 1 max distances in the list (distance values are the same)
                    index_of_max_dist = [index for index, item in enumerate(camera_obj_dist_tracking) if item == max(camera_obj_dist_tracking)]
                    if object_dist < camera_obj_dist_tracking[index_of_max_dist[0]]:
                        camera_obj_dist_tracking[index_of_max_dist[0]] = object_dist
                        box_info[index_of_max_dist[0]] = [x_start, y_start, x_end, y_end, cx, cy]

                # Draw a bounding box for the detected pedestrian
                cv.rectangle(frame_2, (int(x_start), int(y_start)), (int(x_end), int(y_end)), (0, 255, 0), 2)
                cv.rectangle(frame_3, (int(x_start), int(y_start)), (int(x_end), int(y_end)), (0, 255, 0), 2)
                cv.rectangle(frame_4, (int(x_start), int(y_start)), (int(x_end), int(y_end)), (0, 255, 0), 2)

        # Update the tracking information of detected pedestrians
        objects = ct.update(rects)

        # Display the tracking information in the frame - FRAME 3
        centroids = np.asarray(centroids)
        # If there is no pedestrian detected in the current frame, skip. Else, add labels to the bounding boxes
        if len(centroids) != 0:
            for (objectID, centroid) in objects.items():
                # If the tracked centroids match the detected centroids in this frame, add labels (In some cases, some pedestrians cannot be detected
                # in successive frames, but there tracking information is still being stored since their number of disapperance haven't exceeded the 
                # maximum disapperance times. However, we don't want to draw that label in this current frame because that person cannot be detected
                # in this frame)
                if np.any(np.all(centroid == centroids, axis=1)):
                    index = np.where(centroids == centroid)[0][0]
                    x_start, y_start = rects[index][0], rects[index][1] 
                    text = "ID: {}".format(objectID)
                    cv.putText(frame_3, text, (int(x_start), int(y_start - 5)), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display up to 3 closest pedestrians to the camera - FRAME 4
        dist_order = 1
        for i in range (0, len(camera_obj_dist_tracking)):
            index_of_min_dist = [index for index, item in enumerate(camera_obj_dist_tracking) if item == min(camera_obj_dist_tracking)]
            x_start, y_start, x_end, y_end, cx, cy = box_info[index_of_min_dist[0]]
            # Draw a red box for up to 3 closest pedestrians to the camera. Other detected pedestrians are still being drawn in green
            cv.rectangle(frame_4, (int(x_start), int(y_start)), (int(x_end), int(y_end)), (0, 0, 255), 2)

            # Add the order closeness (1 - closest to the camera, 2 - 2nd closest to the camera, 3rd - 3rd closest to the camera)
            text = "Ord: {}".format(dist_order)
            cv.putText(frame_4, text, (int(x_start), int(y_start - 5)), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            camera_obj_dist_tracking.pop(index_of_min_dist[0])
            box_info.pop(index_of_min_dist[0])
            
            dist_order += 1
        
        # Concatenate to display
        first_row = cv.hconcat([frame_1, frame_2])
        second_row = cv.hconcat([frame_3, frame_4])
        stack = cv.vconcat((first_row, second_row))
        cv.imshow("Task 2's Output", stack)

        key = cv.waitKey(20)
        if key == 27:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Add parameters")
    parser.add_argument('file_name', help="Video file")

    args = parser.parse_args()
    video_name = args.file_name

    person_tracking(video_name)


"""
REFERENCES:
    https://learnopencv.com/deep-learning-with-opencvs-dnn-module-a-definitive-guide/
    https://pyimagesearch.com/2021/02/22/opencv-connected-component-labeling-and-analysis/
    https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html
    https://www.youtube.com/watch?v=rPGfY-QODh8
    https://pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/
    https://pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
"""