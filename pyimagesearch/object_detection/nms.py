## import necessary packages
import numpy as np

def non_max_supression(boxes, probs, overlapThresh):
	#if there are no boxes return empty list
	if len(boxes)==0:
		return []

	##if the bounding boxes are integers , convet it into float 
	## as we will do a lot of divisions
	if boxes.dtype.kind == "i":
		boxes=boxes.astype("float")

	# initialize the list of picked indexes
	pick = []

	# grabthe coordinates of bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]

	#compute the area of bounding box and sort the bounding 
	# boxes by their probabilities
	area = (x2 - x1 + 1) * (y2 - y1 +1)
	idxs = np.argsort(probs)

	#keep loping while some indexes stil remain in indexes list
	while len(idxs)>0:
		#grab last index of index list and add the index to the 
		#picked indes
		last = len(idxs)-1
		i=idxs[last]
		pick.append(i)

		# find the largest (x,y) coord for the start of bb  
		# and the smallest (x,y) coord for end of bb
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# compute width and height of bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		#compute the ratio of overlap
		overlap= (w*h) / area[idxs[:last]]

		# delete all indexes from list that have overlap greater
		# than the provided overlap threshold
		idxs = np.delete(idxs, np.concatenate(([last],np.where(overlap > overlapThresh)[0])))


	#return only the boundingbox that were picked
	return boxes[pick].astype("int")




































