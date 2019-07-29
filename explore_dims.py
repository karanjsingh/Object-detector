
#import necessary packages
from __future__ import print_function
from pyimagesearch.utils import conf
from scipy import io
import numpy as np
import argparse
import glob

ap=argparse.ArgumentParser()
ap.add_argument("-c","--conf",required=True,help="Path tothe configuration file")
args=vars(ap.parse_args())

# Load the confoguration file and initialize the list of widths and heights
conf = conf.Conf(args["conf"])
widths = []
heights = [] 

#loop over all annotations path
for p in glob.glob(conf["image_annotations"]+ "/*.mat"):
	#load thebounding box and update widthand height list
	(y, h, x, w) = io.loadmat(p)["box_coord"][0]
	#print("{}".format(y))
	widths.append(w - x)
	heights.append(h - y)
#compute the average of both the width and height lists
(avgWidth ,avgHeight) = (np.mean(widths), np.mean(heights))
print("[info] avg. WIDTH: {:.2f}".format(avgWidth))
print("[info] avg. HEIGHT: {:.2f}".format(avgHeight))
print("[info] aspect ratio: {:.2f}".format(avgWidth/avgHeight))
