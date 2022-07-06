import os
import sys
import copy
import glob
from matplotlib.pyplot import sci
import numpy as np
import cv2
from tifffile import imwrite, imread
import scipy.io
import scipy.ndimage as spim

def getVideosInFolder(dataFolder, avi_names = ''):
	## dataFolder: (str) location to look for video files
	## avi_names: (str) common name shared by videos, should end right before video number
	 # ex: folder with videos msCam1.avi, msCam2.avi; avi_names = 'msCam'
	 # ex: folder with videos 0.avi, 1.avi; avi_names = ''
	allFiles = os.listdir(dataFolder) # all files in the directory given
	videoFiles = {} # build a dictionary of video files, where the number is the key(ex: {'1':'dir/msCam1.avi', '2':'dir/msCam1.avi', etc)
	numVids = 0
	videoNumList = []
	# print('Data folder:  ' + str(dataFolder))
	for filename in allFiles:
		# print(filename)
		 # Looking for .avi files only
		if '.avi' in filename:
			vidNumEnd = filename.find('.avi')
			if bool(avi_names):
				vidNameEnd = filename.find(avi_names) + len(avi_names)
			else:
				vidNameEnd = 0
			vidNum = filename[vidNameEnd:vidNumEnd]
			videoNumList.append(np.int(vidNum))
			numVids = numVids+1
			videoFiles[str(vidNum)]=  str(dataFolder + '/' + filename)
	# print('Video files found (will be sorted numerically):  ' + str(videoFiles))
	videoNumList = np.sort(videoNumList)
	return videoFiles, videoNumList, numVids

def GetMiniscopeDirs(topdir, animals, TiffDirsOnly=None, miniscope_tiff_name='msCam.tiff', behavcam_tiff_name='behavCam.tiff'):
	# topdir: string under which it will loop to find animal/2022_*/sessionfolder/MiniLFOV or /BehavCam
	# TiffDirsOnly will cause the func to return only the dirs with .tiff files (if TRUE), only the ones without .tiff (if FALSE), or all of them (if NONE)
	# miniscope_tiff_name: tiff name for miniscope
	# miniscope_tiff_name: tiff name for behavcam

	MiniFolders  = []
	BehavFolders = []
	parentBehavFolders = []
	parentMiniscopeFolders = []

	for animal in animals:
		an_dir     = str(topdir + animal)
		year_dir   = str(topdir + animal + '/2022_*')
		dayFolders = glob.glob(year_dir)
		if dayFolders != None:
			for day in dayFolders:
				# sessRecs  = os.listdir(day)
				folders   = list(filter(lambda x: os.path.isdir(f"{day}\\{x}"), os.listdir(day)))
				for singleSess in folders:
					# Miniscope camera
					parent = str(day + '/' + singleSess + '/')
					parent   = parent.replace("\\", "/")

					miniFolder   = str(parent + 'MiniLFOV')
					minitiffOut  = str(miniFolder + '/' + miniscope_tiff_name)
					# Behavior camera
					behavFolder  = str(parent + 'BehavCam')
					behavtiffOut = str(behavFolder + '/' + behavcam_tiff_name)

					exist_mini  = os.path.isfile(minitiffOut)
					exist_behav = os.path.isfile(behavtiffOut)
					if TiffDirsOnly is None: # take all dirs
						MiniFolders.append(miniFolder)						
						parentMiniscopeFolders.append(parent)	
						BehavFolders.append(behavFolder)					
						parentBehavFolders.append(parent)	
					else: # take only dirs with tiff, or no tiffs
						if exist_mini==TiffDirsOnly:
							MiniFolders.append(miniFolder)
							parentMiniscopeFolders.append(parent)	
						if exist_behav==TiffDirsOnly:
							BehavFolders.append(behavFolder)
							parentBehavFolders.append(parent)	
	return MiniFolders, BehavFolders, parentMiniscopeFolders, parentBehavFolders

def scroll_images(image_stack, waitTime=1):
	frameIndx = image_stack.shape[0] - 1
	def onChange(trackbarValue):
		img = image_stack[trackbarValue]
		img = img.astype('uint8')
		cv2.imshow("imageFrame", img)
		pass
	cv2.namedWindow('imageFrame')

	cv2.createTrackbar( 'start', 'imageFrame', 0, frameIndx, onChange )
	onChange(0)
	cv2.waitKey()
	cv2.getTrackbarPos('start','imageFrame')
	cv2.destroyAllWindows()
	# trackbarValue = 0
	# while True:
	# 	img = image_stack[trackbarValue]
	# 	img = img.astype('uint8')
	# 	cv2.imshow("imageFrame", img)
	# 	key = cv2.waitKey(waitTime) & 0xff
	# 	if key == ord("q"):
	# 		cv2.destroyAllWindows()
	# 		break
	return

def load_miniscope_avis(dataFolder, avi_names = '', spatialDownSample=1, temporalDownSample=1, frame_filter=None, verbose=False, show_vid=False):
	videoFiles, videoNumList, numVids = getVideosInFolder(dataFolder, avi_names)
	firstVideoNum = videoNumList[0]

	MaxFramesPerFile = 1100 # used guess the total possible frames and preallocate memory

	# Get a sample frame from the first video
	videoFileName = videoFiles[str(firstVideoNum)]
	cap = cv2.VideoCapture(videoFileName)
	ret, refFrame = cap.read(0)
	frameHeight = refFrame.shape[0]
	frameWidth = refFrame.shape[1]

	cap.release()
	cv2.destroyAllWindows()

	####################
	if show_vid:
		cv2.namedWindow("frame")
	####################
	totalFrames = MaxFramesPerFile*(numVids)#+numFramesFinal;
	frameVect = range(0, totalFrames)
	frameMat = np.empty([totalFrames, frameHeight, frameWidth], dtype='uint8')
	
	# Loop through each video
	frameIndex = 0
	vidCount = 0
	for vidID in videoNumList:
		videoFileName = videoFiles[str(vidID)]
		if verbose: print('Starting ' + videoFileName + '... ')
		cap = cv2.VideoCapture(videoFileName)
		ret = True
		fnum = 0
		vidCount = vidCount + 1
		while ret:
			ret, frame = cap.read(fnum)
			if(ret==True):
				frame = frame[:,:,0].astype('float32')
				if frame_filter is not None:
					frame = spim.median_filter(frame, footprint=frame_filter).astype('uint8')
					# frame = cv2.filter2D(frame, -1, frame_filter)
				# frame = frame[cropROI[0][1]:cropROI[1][1],cropROI[0][0]:cropROI[1][0]]
				frame = frame[::spatialDownSample, ::spatialDownSample] # spatial downsampling
				if show_vid:
					cv2.imshow('frame', frame)
					cv2.waitKey(1)
				frameMat[frameVect[frameIndex], :, :] = np.uint8(frame)
				frameIndex = frameIndex + 1
				fnum = fnum + 1
		cap.release()
		if verbose: print('     done - ' + str(vidCount) + ' of ' + str(numVids))
	if show_vid:
		cv2.destroyAllWindows()
	# downsample the matrix
	frameMat = frameMat[0:frameIndex:temporalDownSample, :, :] # temporal downsampling
	return frameMat


def crop_and_convert_miniscope(dataFolder, tiff_name='msCam.tiff', avi_names = '',
								cropROI=None, circleDef=[], spatialDownSample=1, temporalDownSample=1,
								frame_filter=None, verbose=True, show_vid=False):
	## dataFolder: (str) location to look for video files
	## tiff_name: (str) save name for combined tiff file					
	## avi_names: (str) common name shared by videos, should end right before video number
	 # ex: folder with videos msCam1.avi, msCam2.avi; avi_names = 'msCam'
	 # ex: folder with videos 0.avi, 1.avi; avi_names = ''	## cropROI: (int matrix) [[c1, r1], [c2, r2]] row and column (r/c) start and stop (1->2) indices to crop video							
	## circleDef: [center, radius] of a circular mask, outside of which is set to 0							
	## spatialDownSample: (int) number of pixels to downsample in each dimension							
	## temporalDownSample: (int) number of frames to downsample						
	 # spatialDownSample = 1 # no pixel downsampling
	 # temporalDownSample = 1 # no frame downsampling	
	## frame_filter: matrix for filtering each frame using cv2.filter2D
	videoFiles, videoNumList, numVids = getVideosInFolder(dataFolder, avi_names)
	# videoNumList = [videoNumList[0]]
	firstVideoNum = videoNumList[0]
	tiffStackout = dataFolder + '/' + tiff_name

	MaxFramesPerFile = 1100 # used guess the total possible frames and preallocate memory

	# Get a sample frame from the first video
	videoFileName = videoFiles[str(firstVideoNum)]
	cap = cv2.VideoCapture(videoFileName)
	ret, refFrame = cap.read(0)
	originalHeight = refFrame.shape[0]
	originalWidth = refFrame.shape[1]	
	
	if cropROI is None:
		cropROI = [[0, 0],[refFrame.shape[1], refFrame.shape[0]]] # I know its width by height my bad was 1, 0

	frameCrop = refFrame[cropROI[0][1]:cropROI[1][1], cropROI[0][0]:cropROI[1][0], 0]
	# frameRate = int(cap.get(5))
	frameCrop = frameCrop[::spatialDownSample, ::spatialDownSample]
	frameHeight = frameCrop.shape[0]
	frameWidth = frameCrop.shape[1]

	cap.release()
	cv2.destroyAllWindows()

	####################
	if show_vid:
		cv2.namedWindow("frame")
	####################
	totalFrames = MaxFramesPerFile*(numVids)#+numFramesFinal;
	frameVect = range(0, totalFrames)
	frameMat = np.empty([totalFrames, frameHeight, frameWidth], dtype='uint8')
	
	# define mask
	if bool(circleDef):
		center = circleDef[0]
		radius = circleDef[1]
		Y, X = np.ogrid[:originalHeight, :originalWidth]
		dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
		mask = dist_from_center > radius
	else:
		mask = []
	
	print('     ~~~ Combining  ' + str(numVids) + ' video in ' + dataFolder + ' ~~~')
	# Loop through each video
	frameIndex = 0
	vidCount = 0
	for vidID in videoNumList:
		videoFileName = videoFiles[str(vidID)]
		if verbose: print('Starting ' + videoFileName + '... ')
		cap = cv2.VideoCapture(videoFileName)
		ret = True
		fnum = 0
		vidCount = vidCount + 1
		while ret:
			ret, frame = cap.read(fnum)
			if(ret==True):
				frame = frame[:,:,0].astype('uint8')
				if frame_filter is not None:
					frame = spim.median_filter(frame, footprint=frame_filter) # .astype('uint8')
					# frame = cv2.filter2D(frame, -1, frame_filter)
				if bool(circleDef):
					meanF = 0 # np.mean(np.mean(frame))
					frame[mask] = meanF
				frame = frame[cropROI[0][1]:cropROI[1][1],cropROI[0][0]:cropROI[1][0]]
				frame = frame[::spatialDownSample, ::spatialDownSample] # spatial downsampling
				if show_vid:
					cv2.imshow('frame', frame)
					cv2.waitKey(1)
				frameMat[frameVect[frameIndex], :, :] = frame
				frameIndex = frameIndex + 1
				fnum = fnum + 1
		cap.release()
		if verbose: print('     done - ' + str(vidCount) + ' of ' + str(numVids))
	if show_vid:
		cv2.destroyAllWindows()
	# downsample the matrix
	frameMat = frameMat[0:frameIndex:temporalDownSample, :, :] # temporal downsampling

	# frameMat = load_miniscope_avis(dataFolder, avi_names = '', spatialDownSample=1, temporalDownSample=4, frame_filter=None, verbose=False, show_vid=False)

	# save the parameters used
	saveName = dataFolder + '\Crop_params.mat'
	if frame_filter is None: frame_filter = 'None'
	if mask is None: mask = 'None'

	scipy.io.savemat(saveName, {'cropROI':cropROI, 'spatialDownSample':spatialDownSample,
		'temporalDownSample':temporalDownSample, 'totalFrames':frameIndex, 'tiffStackout':tiffStackout,
		'mask':mask, 'frame_filter':frame_filter})
	
	print('Parameters saved to:  ' + saveName)
	# alternative writing method with TiffWriter
	# tifs = np.array([tifffile.imread(t) for t in fls])
	# with TiffWriter(tiffStackout, bigtiff=True) as tiff:
	# 	for i in range(0, frameMat.shape[0]):
	# 		tiff.save(frameMat[i,:,:], compression='None')
	# tiffStackout = dataFolder + '/alt_' + tiff_name
	imwrite(tiffStackout, frameMat, bigtiff=True)
	print('Done! Movie saved to:  ' + tiffStackout)

def get_all_CropROIs(miniscopeFolders, avi_names='', shape='rect', verbose=True, load_all=False):
	print('     ~~~ Cropping ' + str(len(miniscopeFolders)) + ' folders ~~~')
	def draw_circ_mask(event, x, y, flags, param):
		# grab references to the global variables
		global circMid, circRad, refPt, refNew, rectPt, cropping2, refInput, shape2
		# if the left mouse button was clicked, record the starting
		# (x, y) coordinates and indicate that cropping is being
		# performed

		click = all([event == cv2.EVENT_LBUTTONDOWN])
		Ldrag = all([event == cv2.EVENT_MOUSEMOVE, flags == cv2.EVENT_FLAG_LBUTTON])
		Mdrag = all([event == cv2.EVENT_MOUSEMOVE, flags == cv2.EVENT_FLAG_MBUTTON])
		release = all([event == cv2.EVENT_LBUTTONUP])
		dx = np.abs(refPt[0] - refNew[0])
		dy = np.abs(refPt[1] - refNew[1])
		if click: # event == cv2.EVENT_LBUTTONDOWN: # starting point for circle
			cropping2 = True
			refPt = (x, y)
			refNew = (x, y)
		elif Ldrag: # update as its drawn
			refNew = (x, y)
			# check to see if the left mouse button was released
		elif Mdrag: # move center with middle button
			
			center = (x, y)
			refPt = (center[0]-np.floor(dx/2).astype(int), center[1]-np.floor(dy/2).astype(int))
			refNew= (center[0]+np.floor(dx/2).astype(int), center[1]+np.floor(dy/2).astype(int))
		elif release: # event == cv2.EVENT_LBUTTONUP: # final pos
			# record the ending (x, y) coordinates and indicate that
			# the cropping operation is finished
			refNew = (x, y)
			rectPt = [refPt]
			rectPt.append(refNew)
			cropping2 = False
		# draw a rectangle around the region of interest
		midX = np.mean([refNew[0], refPt[0]]).astype(int)
		midY = np.mean([refNew[1], refPt[1]]).astype(int)
		circMid = (midX, midY)
		rad = np.sqrt( (refNew[0] - refPt[0])**2 + (refNew[1] - refPt[1])**2 )/2
		circRad = rad.astype(int)
		refInput = copy.deepcopy(refFrame)

		if shape2=='rectangle':
			crop_size = (dy, dx)
			cv2.rectangle(refInput, refPt, refNew, (0, 0, 255), 2)	
		elif shape2=='square':
			crop_size = (2*circRad, 2*circRad)
			cv2.rectangle(refInput, refPt, refPt+2*circRad, (0, 0, 255), 2)	# -1*circRad
		elif shape2=='circle':
			crop_size = (2*circRad, 2*circRad)
			cv2.circle(refInput, circMid, circRad, (0, 0, 255), 2)

		font = cv2.FONT_HERSHEY_COMPLEX
		org = (10, 30)
		fontScale = .5 # scale
		thickness = 1 # pix
		instr_str = 'lClick-draw, mClick-move, C-confirm, R-reset'
		cv2.putText(refInput, instr_str, org, font, 
						fontScale, (0,0,255), thickness, cv2.LINE_AA)
		org = (30, refFrame.shape[0]-15)
		fontScale = .35 # scale
		thickness = 1 # pix
		pix_str = 'x: ' + str(x) + '   y: ' + str(y) + ';   ' + str(crop_size[0]) + ' x ' + str(crop_size[1]) + 'pix'
		cv2.putText(refInput, pix_str, org, font, 
						fontScale, (0,0,255), thickness, cv2.LINE_AA)
		cv2.imshow("refFrame", refInput)
		# return refFrame

	roiDict = {} # dictionary for rois
	maskDict = {} # dictionary for circles
	for sessionLoop in range(0, len(miniscopeFolders)):
		if True:
			dataFolder = miniscopeFolders[sessionLoop]
			if load_all==True:
				tiff_name = str(dataFolder + '/behavCam.tiff')
				if os.path.isfile(tiff_name):
					print('Reading in existing tiff: ' + tiff_name)
					frameMat = imread(str(dataFolder + '/behavCam.tiff'))
				else:
					print('Loading all avis for reference image, may take a while')
					frameMat = load_miniscope_avis(dataFolder, avi_names = '', spatialDownSample=1, temporalDownSample=4, frame_filter=None, verbose=False, show_vid=False)
				refFrame = np.max(frameMat, axis=0).astype('uint8')
				refFrame = refFrame - np.min(refFrame)
				refFrame = 255*(refFrame/np.max(refFrame))
				refFrame[refFrame>255] = 255
				refFrame = np.uint8(refFrame)
				# b = np.min(frameMat, axis=0).astype('uint8')
				# refFrame = a-b
			else: # 
				videoFiles, videoNumList, numVids = getVideosInFolder(dataFolder, avi_names)
				videoFileName = videoFiles[str(videoNumList[0])]
				cap = cv2.VideoCapture(videoFileName)
				for i in range(0, 10):
					ret, refFrame = cap.read(0) # with trigger start the 0 frame is black?	
			clone = copy.deepcopy(refFrame)

			cv2.namedWindow("refFrame")
			cv2.setMouseCallback("refFrame", draw_circ_mask)

			global refPt, refNew, rectPt, shape2, circMid, circRad
			if shape=='rect' or shape=='r' or shape=='rectangle':
				shape2 = 'rectangle'
			elif shape=='sq' or shape=='s' or shape=='square':
				shape2 = 'square'
			elif shape=='circ' or shape=='c' or shape=='circle':
				shape2 = 'circle'
			refPt = (0,0)
			rectPt = [(0,0), (0,0)]
			refNew = (0,0)
			# keep looping until the 'q' key is pressed
			while True:
				# display the image and wait for a keypress
				# refInput = copy.deepcopy(refFrame)
				cv2.imshow('refFrame', refFrame)
				key = cv2.waitKey(0) & 0xFF
				# if the 'r' key is pressed, rceset the cropping region
				if key == ord("r"):
					refFrame = copy.deepcopy(clone)
					refPt = (0,0)
					rectPt = [(0,0), (0,0)]
					refNew = (0,0)
				# if the 'c' key is pressed, break from the loop
				elif key == ord("c"):
					break
			refROI = [[0, 0],[0, 0]]
			if shape2=='rectangle':
				refROI[0][0] = (np.min([rectPt[0][0], rectPt[1][0]]))
				refROI[0][1] = (np.min([rectPt[0][1], rectPt[1][1]]))
				refROI[1][0] = (np.max([rectPt[0][0], rectPt[1][0]]))
				refROI[1][1] = (np.max([rectPt[0][1], rectPt[1][1]]))
				circMid = None
			elif shape2=='circle': 
				refROI[0][0] = circMid[0] - circRad
				refROI[0][1] = circMid[1] - circRad
				refROI[1][0] = circMid[0] + circRad
				refROI[1][1] = circMid[1] + circRad
			elif shape2=='square':
				refROI[0][0] = refPt[0] #- circRad
				refROI[0][1] = refPt[1] #- circRad
				refROI[1][0] = refPt[0] + 2*circRad
				refROI[1][1] = refPt[1] + 2*circRad
			refROI[0][0] = np.max([refROI[0][0], 1])
			refROI[0][1] = np.max([refROI[0][1], 1])
			refROI[1][0] = np.min([refROI[1][0], refFrame.shape[1]])
			refROI[1][1] = np.min([refROI[1][1], refFrame.shape[0]])

			if refROI[1][0]==0 and refROI[1][1]==0:
				refROI = ([1, 1], [refFrame.shape[1], refFrame.shape[0]])
			if verbose: print('Cropping ROI:  ' + str(refROI))
			roiDict[miniscopeFolders[sessionLoop]] = refROI

			if shape2=='rectangle' or shape2=='square':
				maskDict[miniscopeFolders[sessionLoop]] = []
			else:
				maskDict[miniscopeFolders[sessionLoop]] = [circMid, circRad]
			print(miniscopeFolders[sessionLoop])
	cv2.destroyAllWindows()
	return roiDict , maskDict