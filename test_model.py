# import the necessary packages
from pyimagesearch.object_detection import nms
from pyimagesearch.object_detection import objectdetector
from pyimagesearch.descriptors import hog
from pyimagesearch.utils import conf
import numpy as np
import imutils
import argparse
import pickle
import cv2
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to the configuration file")
ap.add_argument("-i", "--image", required=True, help="path to the image to be classified")
args = vars(ap.parse_args())
 
# load the configuration file
conf = conf.Conf(args["conf"])
 
# load the classifier, then initialize the Histogram of Oriented Gradients descriptor
# and the object detector
model = pickle.loads(open(conf["classifier_path"], "rb").read())
hog = hog.HOG(orientations=conf["orientations"], pixelsPerCell=tuple(conf["pixels_per_cell"]),
	cellsPerBlock=tuple(conf["cells_per_block"]), normalise=conf["normalize"])
od = objectdetector.ObjectDetector(model, hog)

# load the image and convert it to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=min(260, image.shape[1]))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect objects in the image and apply non-maxima suppression to the bounding boxes
(boxes, probs) = od.detect(gray, conf["window_dim"], winStep=conf["window_step"],
	pyramidScale=conf["pyramid_scale"], minProb=conf["min_probability"])
pick = nms.non_max_supression(np.array(boxes), probs, conf["overlap_thresh"])
orig = image.copy()
 
# loop over the original bounding boxes and draw them
for (startX, startY, endX, endY) in boxes:
	cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 0, 255), 2)
 
# loop over the allowed bounding boxes and draw them
for (startX, startY, endX, endY) in pick:
	cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
 
# show the output images
cv2.imshow("Original", orig)
cv2.imshow("Image", image)
cv2.waitKey(0)