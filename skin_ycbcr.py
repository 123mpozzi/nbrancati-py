import cv2
import numpy as np
import time
from math import ceil
import sys
from PIL import Image

bins = 256
tolCr = 1
tolCb = 1
hist_size = (bins, bins)
ranges1 = (0, 255)
ranges2 = (0, 255)
ranges = (ranges1, ranges2)

times = []

# Subtract numpy arrays without wrapping when overflow
# eg. As two uint8 values: 10-20 = 0 instead of 245
def npsubtract(a, b):
  x = a - b
  x[b>a] = 0
  return x


# sort of the histogram
def sortHist(iBins: list, values: list, num: int):
  tmpN = num

  while tmpN >= 0:
    ultimo = -1
    for i in range(tmpN):
      if iBins[i] > iBins[i + 1]:
        tmp = iBins[i]
        iBins[i] = int(iBins[i + 1])
        iBins[i + 1] = int(tmp)
        ultimo = i
        tmp = values[i]
        values[i] = values[i + 1]
        values[i + 1] = tmp
    tmpN = ultimo
  return iBins

# Computation of Min and Max of the histogram (5th and 95th percentile)
def calcMinMaxHist(yValues: int, iBins: list, vect: list) -> None:
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
    #app = iBins
    #app.sort()
    app = sortHist(app, iBins, k - 1)
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

  print("\niBinsVal\n")
  for iyy in range(indTol):
    for iyj in range(bins):
      print(iBinsVal[iyy][iyj], end=" ")

  for i in range(indTol):
    for k in range(bins):
      app[k] = yValue[i][k]
      iBins[k] = iBinsVal[i][k]
    #app = iBins
    #app.sort()
    app = sortHist(app, iBins, k - 1)

    print(f'\napp{len(app)}')
    for ixj in range(len(app)):
      print(app[ixj], end=" ")

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
  #hist, bin_edges =  np.histogram(plane1, bins=bins, range=ranges1)
  #hist = np.bincount(plane1.ravel(),minlength=256) # numpy faster
  hist = cv2.calcHist([plane1],[0],None,[256],[0,256]) # opencv fastest
  return hist

def calculateHist2(plane1, plane2):
  #res = cv2.calcHist([plane1, plane2], [2], None, (bins,bins), ranges)
  #res1 = cv2.calcHist([plane1],[0],None,[256],[0,256])
  #res2 = cv2.calcHist([plane1],[0],None,[256],[0,256])
  img = np.dstack((plane1,plane2))
  res = cv2.calcHist([img], [0, 1], None, [256, 256], [0, 256, 0, 256])
  return res

def calculateHist2A(plane1, plane2):
  scale = 1
  planes = (plane1, plane2)
  hist_img = np.zeros((bins*scale, bins*scale, 3), np.uint8)
  print('PREFLATTEN')
  print(plane1)
  plane1 = np.ndarray.flatten(plane1)
  print('FLATTENDED')
  print(plane1)
  plane2 = np.ndarray.flatten(plane2)
  print(plane2)
  hist, x_edges, y_edges = np.histogram2d(plane1, plane2, bins=(bins, bins), range=ranges)
  print(np.shape(hist))
  return hist

def calculateHist2O(plane1, plane2):
  hist1 = calculateHist(plane1)
  hist2 = calculateHist(plane2)
  print(np.shape(hist1))
  print(np.shape(hist2))
  res = np.dstack((hist1,hist2))
  return (hist1,hist2)
  #return np.concatenate((hist1, hist2, ...), axis=0)

  # np.histogram2d expects flat lists of x and y coordinates, not list-of-lists
  plane1 = np.ndarray.flatten(plane1)
  plane2 = np.ndarray.flatten(plane2)
  hist, x_edges, y_edges = np.histogram2d(plane1, plane2, bins=(bins, bins), range=ranges)
  return hist

def calculateHist2N(plane1, plane2):
  # np.histogram2d expects flat lists of x and y coordinates, not list-of-lists
  plane1 = np.ndarray.flatten(plane1)
  plane2 = np.ndarray.flatten(plane2)
  hist, x_edges, y_edges = np.histogram2d(plane1, plane2, bins=(bins, bins), range=ranges)
  return hist


# TODO: save different outputs images from C code, each one after a significant code block
# and then do the same here and compare images to see where the algo is wrong 
def skin_detect(image_in: str, image_out: str):
  CrMin = float(133)
  CrMax = float(183)
  CbMin = float(77)
  CbMax = float(128)
  YMin = float(0)
  YMax = float(255)

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

  frame_ycrcb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2YCR_CB)

  y_plane, cr_plane, cb_plane = cv2.split(frame_ycrcb)

  histCb = calculateHist(cb_plane)
  histCr = calculateHist(cr_plane)
  print('\nhistCb\n')
  for ix in range(bins):
    print(histCb[ix], end=" ")
  print('\nhistCr\n')
  for ix in range(bins):
    print(histCr[ix], end=" ")
  print('\n')

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
  #print(np.shape(histYCb))
  
  #im = Image.fromarray(histYCb)
  #im = im.convert('RGB')
  #im.save("histYCb.jpeg")

  #print('\histYCb\n')
  #for ix in range(bins):
  #  for ij in range(bins):
  #    print(histYCb[ix][ij], end=" ")
  #print('\n')


  print(time.time()-time_start)
  # Computation of (Y0,CrMax) and (Y1,CrMax) by means of the calculus of percentiles
  if max_valCr != -1:
    if max_valCr > CrMax:
      max_valCr = CrMax
    minMaxCr = calculateValueMinMaxY(frame_ycrcb, max_valCr, histYCr, 1)
    if max_valCr < CrMax:
      CrMax = max_valCr
  print('minmaxCr')
  print(minMaxCr)

  # Computation of (Y2,CbMin) and (Y3,CbMin) by means of the calculus of percentiles
  if min_valCb != -1:
    if min_valCb < CbMin:
      min_valCb = CbMin
    minMaxCb = calculateValueMinMaxY(frame_ycrcb, min_valCb, histYCb, 2)
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
  hCr = float(CrMax - CrMin)
  hCb = float(CbMax - CbMin)
  ACr = ((B + bCr) * hCr) / 2
  ACb = ((B + bCb) * hCb) / 2

  print(f'\nY1 {Y1}')
  print(f'\nY2 {Y2}')
  print(f'\nY3 {Y3}')
  print(f'\nhCr {hCr}')
  print(f'hCb {hCb}')
  print(f'CbMin {CbMin}')
  print(f'\nACr {ACr}')
  print(f'ACb {ACb}')


  print(time.time()-time_start)

  Y = y_plane # max HCb is right with np.int8(y_plane)
  Cr = cr_plane
  Cb = cb_plane

  print(Y)
  print('YYYYYYYYYY')
  print(type(Y))
  datt= (Y - Y2)
  datb = np.clip(Y - Y2, 0, 255).astype(np.uint8)
  # CbMin + hCb * ((Y - Y2) / (YMin - Y2))
  print(datt)
  print(datb)
  print(np.max(datt))
  print(np.min(datt))

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
  #HCr = np.logical_or(HCr1, np.logical_or(HCr2, HCr3))
  print('\n HCR')
  print(np.max(HCr))
  print(np.min(HCr))

  # Calculate HCb
  # arr[arr - subtract_me < threshold] = threshold
  HCbMask1 = np.logical_and(Y >= YMin, Y < Y2)
  HCb1 = np.multiply(HCbMask1, CbMin + hCb * ((np.int8(Y) - Y2) / (YMin - Y2))) # TODO: use cleaner approach to perform color subtraction / saturated subtraction
  HCbMask2 = np.logical_and(Y >= Y2, Y < Y3)
  HCb2 = np.multiply(HCbMask2, CbMin)
  HCbMask3 = np.logical_and(Y >= Y3, Y <= YMax)
  HCb3 = np.multiply(HCbMask3, CbMin + hCb * ((Y - Y3) / (YMax - Y3)))
  HCb = HCb1 + HCb2 + HCb3
  print('\n HCB')
  print(YMin - Y2)
  print(np.max(HCb1))
  print(np.min(HCb1))
  #HCb1 = HCb1 + abs(np.min(HCb1))
  #print(np.min(HCb1))

  #print('NPMAX')
  #print(np.max(np.uint8(HCb3)))
  cv2.imwrite('hcr.png', HCr)
  cv2.imwrite('hcb.png', HCb)

  dCr = Cr - CrMin
  DCr = HCr - CrMin
  DCb = CbMax - HCb

  #cv2.imwrite('DCr.png', DCr)

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

  cv2.imwrite('dCbS.png', dCbS)

  CbS = CbMax - dCbS
  print(CbS.any())

  sf = float(minb) / float(maxb)
  print(sf)
  # Condition C.0
  Ivals = (D1Cr + D1Cb) - (dCr + dCbS)
  print(Ivals.any())
  I = np.absolute(Ivals) * sf
  print(I.any())

  cv2.imwrite('I.png', I)

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
  

  cv2.imwrite('J.png', J)
  cv2.imwrite('Cb.png', Cb)
  cv2.imwrite('CbS.png', CbS)
  cv2.imwrite('Cr.png', Cr)

  #print('CR')
  #print(Cr)
  # Skin pixels
  #Cr_i = Cr.astype(int)
  #Cb_i = Cb.astype(int)
  mask5 = cv2.subtract(Cr, Cb) >= I
  print(mask5.any())
  #CbS_i = CbS.astype(int)
  print(type(Cb.dtype))
  print(type(CbS.dtype))
  mask6 = np.absolute(np.float64(Cb) - CbS).astype(np.uint8) <= J
  print('mask6')
  print(mask6.any())
  skinCond = np.logical_and(mask5, mask6)
  print(skinCond.any())
  bw_final = skinCond * 255
  print(bw_final.any())
  #bw_final = np.multiply(mask6, 255)
  #bw_final = np.multiply(mask5, 255)
  
  cv2.imwrite('mask1.png', mask5*255)
  cv2.imwrite('mask2.png', mask6*255)

  #if int(Cr) - int(Cb) >= I and abs(int(Cb) - int(CbS)) <= J:
  #  bw_final[i,j] = 255
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
