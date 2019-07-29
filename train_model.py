# import necessary packages
from __future__ import print_function
from pyimagesearch.utils import dataset
from pyimagesearch.utils import conf
from sklearn.svm import SVC
import numpy as np
import argparse
import pickle

#construct the argument parser
ap=argparse.ArgumentParser()
ap.add_argument("-c","--conf",required=True,
	help="path to configuration file")
ap.add_argument("-n","--hard-negatives",type=int,default=-1,
	help= "flag indicates wether or not hard negatives should be used")
args=vars(ap.parse_args())	
 
# load the configuration file and install dataset
print("[INFO] loading dataset ......")
conf = conf.Conf(args["conf"])
(data,labels) = dataset.load_dataset(conf["features_path"], "features")

#check to see if hard negative flag was supplied
if args["hard_negatives"]>0:
	print("[INFO] loading hard negatives ......")
	(hardData,hardLabels)= dataset.load_dataset(conf["features_path"],"hard_negatives")
	data = np.vstack([data,hardData])
	labels=np.hstack([labels,hardLabels])

# train the classifier (SVM)
print("[INFO] training the classifier....")
model=SVC(kernel="linear",C=conf["C"],probability =True,random_state=42)
model.fit(data,labels)

#dumpthe classifier to file
print("[INFO] dumping classifier .... ")
f=open(conf["classifier_path"],"wb")
f.write(pickle.dumps(model))
f.close()



