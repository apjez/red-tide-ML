import numpy as np
from bisect import bisect

# Uses linear interpolation to make a ROC use the same false positive points as a reference
def convertROC(fpr, tpr, refFpr):
	tprsAtRef = np.zeros_like(refFpr)

	count = 0

	for ref_fpr in refFpr:

		# Find index to the right of the reference point in the ROC
		ind = bisect(fpr, ref_fpr)
		if(ind > 0 and ind < len(fpr)):
			# Do linear interpolation
			tprAtRef = tpr[ind-1] + (ref_fpr - fpr[ind-1])*((tpr[ind]-tpr[ind-1])/(fpr[ind]-fpr[ind-1]))
		elif(ind > 0):
			tprAtRef = tpr[ind-1]
		else:
			tprAtRef = tpr[ind]

		tprsAtRef[count] = tprAtRef
		count = count + 1

	return tprsAtRef