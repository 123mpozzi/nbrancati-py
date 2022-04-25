import cv2
import numpy as np
import time
from math import ceil
import sys

bins = 256
tolCr = 1
tolCb = 1
hist_size = (bins, bins)
ranges1 = (0, 255)
ranges2 = (0, 255)
ranges = (ranges1, ranges2)

times = []


# Computation of Min and Max of the histogram (5th and 95th percentile)
def calcMinMaxHist(yValues: int, iBins: int, vect: list) -> None:
	flag = 0
	maxVal = 0
	percentage = 0
	app = [0] * bins
	for i in range(yValues[0]):
		app[i] = 0
	for i in range(1, yValues[0]):
		maxVal = maxVal + yValues[i]
	i = 1
	if int(maxVal != 0):
		while flag != 1:
			percentage = percentage + int(yValues[i])

			if ceil((percentage / maxVal) * 100) >= 5:
				flag = 1
				y = yValues[i]
			i = i+1
		vect[0] = i - 1
		i = 1
		flag = 0
		percentage = 0
		while flag != 1:
			percentage = percentage + int(yValues[i])
			if ceil((percentage / maxVal) * 100) >= 95:
				flag = 1
				y = yValues[i]
			i = i+1
		vect[1] = i - 1

		k = 0
		for i in range(vect[0], vect[1] +1):
			if iBins[i] != 0:
				app[k] = iBins[i]
				k = k+1
		app = iBins
		app.sort()
		vect[0] = 255
		vect[1] = 0
		for i in range(k):
			if app[i] != 0:
				vect[0] = app[i]
				break
		for i in range(k - 1, -1, -1):
			if app[i] != 0:
				vect[1] = app[i]
				break
	else:
		vect[0] = 255
		vect[1] = 0

# Computation of the vertices (Y0,CrMax) and (Y1,CrMax) of the trapezium in the YCr subspace
# Computation of the vertices (Y2,CbMin) and (Y3,CbMin) of the trapezium in the YCb subspace
def calculateValueMinMaxY(image, val: float, hist, canal: int) -> list:
	minMax = [0] * 2
	min = 255
	max = 0
	indMax = 0
	indMin = 0
	tmpVal = val
	if canal == 1:
		tol = tolCr
	else:
		tol = tolCb
	indTol = (2 * (tol + 1)) - 1
	app = [0] * bins
	iBins = [0] * bins
	app2 = [0] * bins
	iBins2 = [0] * bins
	for i in range(bins):
		app[i] = 0
		app2[i] = 0
		iBins2[i] = 0
		iBins[i] = 0
	yValue = [0] * indTol
	iBinsVal = [0] * indTol
	for i in range(indTol):
		yValue[i] = [0] * bins
		iBinsVal[i] = [0] * bins
	for j in range(indTol):
		for i in range(bins):
			yValue[j][i] = 0
			iBinsVal[j][i] = 0
	
	height, width, channels = image.shape
	for i in range(height - 1):
		for j in range(width - 1):
			# I(x,y)c ~ ((T*)(img->imageData + img->widthStep*y))[x*N + c]
			spk = image[i, j, canal]
			#spk = image[j * channels + canal, i]
			##spk = (image.imageData + i * image.widthStep)[j * channels + canal]
			if spk >= tmpVal - tol and spk <= tmpVal + tol:
				k = image[i, j, 0]
				#k = (image.imageData + i * image.widthStep)[j * channels + 0]
				bin_val = 0
				# https://docs.huihoo.com/opencv/2.4-documentation/modules/legacy/doc/histograms.html
				#bin_val = cv2.cvQueryHistValue_2D(hist, k, spk)
				bin_val = hist[k, spk]
				if bin_val != 0:
					for l in range(indTol):
						if int(tmpVal - spk + l) == tol:
							yValue[l][k] = bin_val
							iBinsVal[l][k] = k
	for i in range(indTol):
		for k in range(bins):
			app[k] = yValue[i][k]
			iBins[k] = iBinsVal[i][k]
		app = iBins
		app.sort()
		j = 1
		for k in range(bins):
			if app[k] != 0:
				app2[j] = app[k]
				iBins2[j] = iBins[k]
				j = j+1
		app2[0] = j
		minMax[0] = 255
		minMax[1] = 0
		# Computation of Min and Max of the histogram
		calcMinMaxHist(app2, iBins2, minMax)
		if minMax[0] != minMax[1]:
			if minMax[0] != 255:
				indMin = indMin+1
				if minMax[0] < min:
					min = minMax[0]
			if minMax[1] != 0:
				indMax = indMax+1
				if minMax[1] > max:
					max = minMax[1]
	minMax[0] = min
	minMax[1] = max
	return minMax

def calculateHist(plane1):
	# np.histogram flatten the given array itself
	hist, bin_edges =  np.histogram(plane1, bins=bins, range=ranges1)
	return hist

def calculateHist2(plane1, plane2):
	# np.histogram2d expects flat lists of x and y coordinates, not list-of-lists
	plane1 = np.ndarray.flatten(plane1)
	plane2 = np.ndarray.flatten(plane2)
	hist, x_edges, y_edges = np.histogram2d(plane1, plane2, bins=(bins, bins), range=ranges)
	return hist


# TODO: save different outputs images from C code, each one after a significant code block
# and then do the same here and compare images to see where the algo is wrong 
def skin_detect(image_in: str, image_out: str):
	CrMin = 133
	CrMax = 183
	CbMin = 77
	CbMax = 128
	YMin = 0
	YMax = 255

	try:
		source = cv2.imread(image_in, cv2.IMREAD_COLOR)
	except:
		exit('No input image found')
	
	height, width, channels = source.shape
	depth = 8

	minMaxCr = [0] * 2
	minMaxCb = [0] * 2

	# inizio calcolo tempi esecuzione
	time_start = time.time()

	# ALGO
	frame_rgb = source.copy()
	perc = width * height * 0.1 / 100

	frame_ycbcr = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2YCR_CB)

	y_plane, cr_plane, cb_plane = cv2.split(frame_ycbcr)

	histCb = calculateHist(cb_plane)
	histCr = calculateHist(cr_plane)

	max_valCr = 0
	minMaxCr[0] = 255
	minMaxCr[1] = 0
	minMaxCb[0] = 255
	minMaxCb[1] = 0

	print(time.time()-time_start)

	# Computation of Crmax
	for i in range(bins - 1, -1, -1):
		if histCr[i] != 0 and histCr[i] > perc:
			max_valCr = i
			break

	# Computation of Cbmin
	min_valCb = 0
	for i in range(bins):
		if histCb[i] != 0 and histCb[i] > perc:
			min_valCb = i
			break
	
	histYCb = calculateHist2(y_plane, cb_plane)
	histYCr = calculateHist2(y_plane, cr_plane)

	print(time.time()-time_start)
	# Computation of (Y0,CrMax) and (Y1,CrMax) by means of the calculus of percentiles
	if max_valCr != -1:
		if max_valCr > CrMax:
			max_valCr = CrMax
		minMaxCr = calculateValueMinMaxY(frame_ycbcr, max_valCr, histYCr, 1)
		if max_valCr < CrMax:
			CrMax = max_valCr

	# Computation of (Y2,CbMin) and (Y3,CbMin) by means of the calculus of percentiles
	if min_valCb != -1:
		if min_valCb < CbMin:
			min_valCb = CbMin
		minMaxCb = calculateValueMinMaxY(frame_ycbcr, min_valCb, histYCb, 2)
		if min_valCb > CbMin:
			CbMin = min_valCb

	Y0 = 50
	Y1 = 110
	Y2 = 140
	Y3 = 200
	# Store of Y0, Y1
	if max_valCr != -1:
		Y0 = minMaxCr[0]
		Y1 = minMaxCr[1]
	# Store of Y2, Y3
	if min_valCb != -1:
		Y2 = minMaxCb[0]
		Y3 = minMaxCb[1]
	
	bw_final = np.zeros((height, width, 1), np.uint8)
	ACr = 0
	ACb = 0
	B = 256
	bCr = Y1 - Y0
	bCb = Y3 - Y2
	if bCr > bCb:
		maxb = bCr
		minb = bCb
	else:
		maxb = bCb
		minb = bCr
	hCr = CrMax - CrMin
	hCb = CbMax - CbMin
	ACr = ((B + bCr) * hCr) / 2
	ACb = ((B + bCb) * hCb) / 2


	print(time.time()-time_start)

	Y = y_plane
	Cr = cr_plane
	Cb = cb_plane

	# Calculate HCr
	#print(Y >= YMin)
	HCrMask1 = np.logical_and(Y >= YMin, Y < Y0)
	print(HCrMask1)
	HCr1 = np.multiply(HCrMask1, CrMin + hCr * ((Y - YMin) / (Y0 - YMin)))
	print(HCr1)
	HCrMask2 = np.logical_and(Y >= Y0, Y < Y1)
	HCr2 = np.multiply(HCrMask2, CrMax)
	HCrMask3 = np.logical_and(Y >= Y1, Y <= YMax)
	HCr3 = np.multiply(HCrMask3, CrMin + hCr * ((Y - YMax) / (Y1 - YMax)))
	HCr = HCr1 + HCr2 + HCr3

	# Calculate HCb
	HCbMask1 = np.logical_and(Y >= YMin, Y < Y2)
	HCb1 = np.multiply(HCbMask1, CbMin + hCb * ((Y - Y2) / (YMin - Y2)))
	HCbMask2 = np.logical_and(Y >= Y2, Y < Y3)
	HCb2 = np.multiply(HCbMask2, CbMin)
	HCbMask3 = np.logical_and(Y >= Y3, Y <= YMax)
	HCb3 = np.multiply(HCbMask3, CbMin + hCb * ((Y - Y3) / (YMax - Y3)))
	HCb = HCb1 + HCb2 + HCb3


	dCr = Cr - CrMin
	DCr = HCr - CrMin
	DCb = CbMax - HCb

	if ACr > ACb:
		D1Cr = DCr * ACb / ACr
		D1Cb = DCb
	else:
		D1Cr = DCr
		D1Cb = DCb * ACr / ACb
	alpha = np.true_divide(D1Cb, D1Cr)
	print(alpha.any())
	mask1 = D1Cr > 0
	mask2 = np.logical_not(mask1)
	dCbS1 = np.multiply(mask1, np.multiply(dCr, alpha))
	dCbS2 = np.multiply(mask2, 255)
	dCbS = dCbS1 + dCbS2
	CbS = CbMax - dCbS
	print(CbS.any())

	sf = float(minb) / float(maxb)
	print(sf)
	# Condition C.0
	Ivals = (D1Cr + D1Cb) - (dCr + dCbS)
	print(Ivals.any())
	I = np.absolute(Ivals) * sf
	print(I.any())
	#I = abs((D1Cr + D1Cb) - (dCr + dCbS)) * sf
	# Condition C.1
	Jvals = np.multiply(dCbS, np.true_divide((dCbS + dCr), (D1Cb + D1Cr)))
	print(Jvals.any())
	mask3 = (D1Cb + D1Cr) > 0
	#print(mask3)
	print('mask3')
	print(mask3.any())
	mask4 = np.logical_not(mask3)
	print(mask4.any())
	#print(mask4)
	J1 = np.multiply(mask3, Jvals)
	J2 = np.multiply(mask4, 255)
	J = J1 + J2
	print(J.any())
	
	#print('CR')
	#print(Cr)
	# Skin pixels
	#Cr_i = Cr.astype(int)
	#Cb_i = Cb.astype(int)
	Cr_i = Cr
	Cb_i = Cb
	mask5 = Cr_i - Cb_i >= I
	print(mask5.any())
	#CbS_i = CbS.astype(int)
	CbS_i = CbS
	mask6 = np.absolute(Cb_i - CbS_i) <= J
	print('mask6')
	print(mask6.any())
	skinCond = np.logical_and(mask5, mask6)
	print(skinCond.any())
	bw_final = np.multiply(skinCond, 255)
	print(bw_final.any())
	#bw_final = np.multiply(mask6, 255)
	#bw_final = np.multiply(mask5, 255)
	
	#if int(Cr) - int(Cb) >= I and abs(int(Cb) - int(CbS)) <= J:
	#	bw_final[i,j] = 255
	print(time.time()-time_start)

	# fine calcolo tempi di esecuzione
	time_end = time.time()
	time_taken = time_end - time_start
	# salva i tempi di esecuzione in un file
	with open("bench.txt", "w") as text_file:
		text_file.write("{},{}\n".format(image_out, time_taken))

	# cvShowImage("Skin Pixels",bw_final)cvWaitKey(0)
	cv2.imwrite(image_out, bw_final)


if __name__ == "__main__":
	image_in = sys.argv[1]
	image_out = sys.argv[2]
	skin_detect(image_in, image_out)
