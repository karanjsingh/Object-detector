#import necessary packages
import numpy as np
import h5py

## DUMPING THE DATASET
def dump_dataset(data, labels, path, datasetName, writeMethod = 'w'):
	#open database,create dataset,write the data and labels to dataset
	#close the database
	db = h5py.File(path, writeMethod) #opened
	dataset = db.create_dataset(datasetName, (len(data), len(data[0])+1), dtype = "float")
	dataset[0:len(data)] = np.c_[labels,data]
	db.close()

## LOADING THE DATASET
def load_dataset(path, datasetName):
	#open the database, grab the labels and data, then close the database
	db=h5py.File(path, 'r')
	(labels, data) = (db[datasetName][:,0], db[datasetName][:,1:])
	db.close()

	return (data, labels)

##DATA IS THE LIST OF FEATURE VECTORS, WRITTEN TO HDF5 DATASET
##LABELS ASSOCIATE WITH FEATURE VECTORS, REP AS {-1, +1}, WHERE -1 MEANS FEATURE VECTOR NOT BELONG TO OBJ AND +1 IS BELONG TO OBJ
##PATHIS WHERE HDF5 DATASET WILL BE STORED IN DISK
##DATASETNAME IS THE NAME OF DATASET WITHIN THE HDF5 FILE
