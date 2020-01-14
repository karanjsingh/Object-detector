# import the necessary packages
from . import helpers

class ObjectDetector:
	def __init__(self, model, desc):
		#store the classifier and HOG descriptor
		self.model=model   # SVM model
		self.desc=desc     # descriptor

	def detect(self, image, winDim, winStep=4, pyramidScale=1.5, minProb=0.7):
		# initialize the list of bounding boxes and associated probabilities
		boxes = []
		probs = []

		##loop over image pyramid
		for layer in helpers.pyramid(image, scale= pyramidScale, minSize=winDim):
			# determine the current scale of the pyramid
			scale= image.shape[0]/ float(layer.shape[0])

			#loop over the sliding windows for the current pyramid layer
			for (x,y,window) in helpers.sliding_window(layer,winStep,winDim):
				#grab the dimentions of the window
				(winH,winW) = window.shape[:2]

				#ensure the window dim. match the supplied sliding win. dim.
				if winH == winDim[1] and winW == winDim[0]:
					#extract HOG , classify window contains object or not
						
					features = self.desc.describe(window).reshape(1,-1)
					prob = self.model.predict_proba(features)[0][1]

					#chech to see if the classifier has found an object 
					#with sufficient probability
					if prob > minProb:
						#compute the (x,y)-coordinates of the bounding box
						# using the current scale of the image pyramid
						(startX, startY) = (int(scale*x),int(scale*y))
						endX = int(startX + (scale*winW))
						endY = int(startY + (scale*winH))

						# update the list of bounding boxes andprobabilities
						boxes.append((startX,startY,endX,endY))
						probs.append(prob)

		#return a tuple of the bounding boxes and probabilities
		return(boxes,probs)


