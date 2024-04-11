# from heapq import merge
import os
import sys
import copy
import glob
import json
# from this import d
import time
# from matplotlib.pyplot import sci
import numpy as np
import cv2
from tifffile import imwrite, imread
import scipy.io
import scipy.ndimage as spim

def getVideosInFolder(dataFolder, avi_names = '', avi2load=None):
	## dataFolder: (str) location to look for video files
	## avi_names: (str) common name shared by videos, should end right before video number
	 # ex: folder with videos msCam1.avi, msCam2.avi; avi_names = 'msCam'
	 # ex: folder with videos 0.avi, 1.avi; avi_names = ''
	allFiles = os.listdir(dataFolder) # all files in the directory given
	videoDict = {} # build a dictionary of video files, where the number is the key(ex: {'1':'dir/msCam1.avi', '2':'dir/msCam1.avi', etc)
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
			vidTemp = filename[vidNameEnd:vidNumEnd]
			if avi2load is not None:
				if np.any(np.isin(vidTemp, avi2load) == True):
					vidNum = filename[vidNameEnd:vidNumEnd]
					videoNumList.append(np.int(vidNum))
					numVids = numVids+1
					videoDict[str(vidNum)]=  str(dataFolder + '/' + filename)
			else:
				vidNum = filename[vidNameEnd:vidNumEnd]
				videoNumList.append(np.int(vidNum))
				numVids = numVids+1
				videoDict[str(vidNum)]=  str(dataFolder + '/' + filename)
	# print('Video files found (will be sorted numerically):  ' + str(videoDict))
	videoNumList = np.sort(videoNumList)
	return videoDict, videoNumList, numVids

def GetMiniscopeDirs(topdir, animals, TiffDirsOnly=None, miniscope_tiff_name='msCam.tiff', behavcam_tiff_name='behavCam.tiff', scope_name='miniscope', behavcam_name='BehavCam'):
	# topdir: string under which it will loop to find animal/2022_*/sessionfolder/MiniLFOV or /BehavCam
	# TiffDirsOnly will cause the func to return only the dirs with .tiff files (if TRUE), only the ones without .tiff (if FALSE), or all of them (if NONE)
	# miniscope_tiff_name: tiff name for miniscope
	# miniscope_tiff_name: tiff name for behavcam

	MiniFolders  = []
	BehavFolders = []
	parentBehavFolders = []
	parentMiniscopeFolders = []

	for animal in animals:
		# an_dir     = str(topdir + animal)
		year_dir   = str(topdir + animal + '/202*')
		# year_dir   = str(topdir + animal + '/2022_*')
		dayFolders = glob.glob(year_dir)
		if dayFolders != None:
			for day in dayFolders:
				if os.path.isdir(day):
				# sessRecs  = os.listdir(day)
					folders   = list(filter(lambda x: os.path.isdir(f"{day}\\{x}"), os.listdir(day)))
					for singleSess in folders:
						# Miniscope camera
						parent = str(day + '/' + singleSess + '/')
						parent   = parent.replace("\\", "/")

						miniFolder   = str(parent + scope_name)
						minitiffOut  = str(miniFolder + '/' + miniscope_tiff_name)
						# Behavior camera
						behavFolder  = str(parent + behavcam_name)
						behavtiffOut = str(behavFolder + '/' + behavcam_tiff_name)

						exist_folder  = os.path.isdir(miniFolder)
						exist_mini  = os.path.isfile(minitiffOut)
						exist_behav = os.path.isfile(behavtiffOut)
						if TiffDirsOnly is None: # take all dirs
							if exist_folder == True:
								MiniFolders.append(miniFolder)						
							parentMiniscopeFolders.append(parent)	
							BehavFolders.append(behavFolder)					
							parentBehavFolders.append(parent)	
						else: # take only dirs with tiff, or no tiffs
							if exist_mini==TiffDirsOnly:
								if exist_folder == True:
									MiniFolders.append(miniFolder)
								parentMiniscopeFolders.append(parent)	
							if exist_behav==TiffDirsOnly:
								BehavFolders.append(behavFolder)
								parentBehavFolders.append(parent)	
	return MiniFolders, BehavFolders, parentMiniscopeFolders, parentBehavFolders

def scroll_images(image_stack, resize_val=1):
	frameIndx, h, w = image_stack.shape
	def onChange(trackbarValue):
		if trackbarValue == frameIndx:
			ind = -1
		else:
			ind = trackbarValue
		# print(str(ind))
		img = cv2.resize(image_stack[ind].astype('float32'), (resize_val*w, resize_val*h), interpolation=cv2.INTER_NEAREST)
		img = img.astype('uint8')
		cv2.imshow("imageFrame", img)
		pass
	cv2.namedWindow('imageFrame')

	cv2.createTrackbar( 'start', 'imageFrame', 0, frameIndx-1, onChange )
	onChange(0)
	cv2.waitKey()
	cv2.getTrackbarPos('start','imageFrame')
	cv2.destroyAllWindows()
	return

def scroll_images_RGB(r_stack, g_stack, b_stack, resize_val=1):
	frameIndx, h, w = r_stack.shape
	def onChange(trackbarValue):
		if trackbarValue == frameIndx:
			ind = -1
		else:
			ind = trackbarValue
		# print(str(ind))
		r = cv2.resize(r_stack[ind].astype('float32'), (resize_val*w, resize_val*h), interpolation=cv2.INTER_NEAREST)
		g = cv2.resize(g_stack[ind].astype('float32'), (resize_val*w, resize_val*h), interpolation=cv2.INTER_NEAREST)
		b = cv2.resize(b_stack[ind].astype('float32'), (resize_val*w, resize_val*h), interpolation=cv2.INTER_NEAREST)
		img = np.dstack((b,g,r))
		img = img.astype('uint8')
		cv2.imshow("imageFrame", img)
		pass
	cv2.namedWindow('imageFrame')

	cv2.createTrackbar( 'start', 'imageFrame', 0, frameIndx-1, onChange )
	onChange(0)
	cv2.waitKey()
	cv2.getTrackbarPos('start','imageFrame')
	cv2.destroyAllWindows()
	return

def load_miniscope_avis(dataFolder, avi_names = '', cropROI=None, spatialDownSample=1, temporalDownSample=1, frame_filter_size=None, verbose=False, show_vid=False, avi2load=None, getRGB=False):
	videoDict, videoNumList, numVids = getVideosInFolder(dataFolder, avi_names, avi2load)
	firstVideoNum = videoNumList[0]
	metaFile = open(dataFolder + '/metaData.json')
	metaData = json.load(metaFile)
	metaFile.close()
	FramesPerFile = metaData['framesPerFile'] # used guess the total possible frames and preallocate memory

	# Get a sample frame from the first video
	videoFileName = videoDict[str(firstVideoNum)]
	cap = cv2.VideoCapture(videoFileName)
	ret, refFrame = cap.read(0)
	frameWidth, frameHeight = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
	# FramesPerFile = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	
	####################
	cv2.destroyAllWindows()
	if show_vid:
		cv2.namedWindow("frame")
	####################
	totalFrames = FramesPerFile*(numVids)#+numFramesFinal;
	frameVect = range(0, totalFrames)
	if cropROI is None:
		cropROI = [[0,0],[0,0]]
		cropROI[0][1] = 0
		cropROI[1][1] = frameHeight
		cropROI[0][0] = 0
		cropROI[1][0] = frameWidth
	elif cropROI == []:
		cropROIs, maskROIs = get_all_CropROIs([dataFolder], avi_names=avi_names, shape='rect', verbose=verbose)
		cropROI = cropROIs[dataFolder]



	refFrame = refFrame[cropROI[0][1]:cropROI[1][1],cropROI[0][0]:cropROI[1][0]]
	refFrame = refFrame[::spatialDownSample, ::spatialDownSample]
	frameHeight = refFrame.shape[0]
	frameWidth = refFrame.shape[1]
	frameMat_r = np.empty([totalFrames, frameHeight, frameWidth], dtype='uint8')
	if getRGB is True:
		frameMat_g = np.empty([totalFrames, frameHeight, frameWidth], dtype='uint8')
		frameMat_b = np.empty([totalFrames, frameHeight, frameWidth], dtype='uint8')
	else:
		frameMat_g = None
		frameMat_b = None

	cap.release()

	# Loop through each video
	frameIndex = 0
	vidCount = 0
	for vidID in videoNumList:
		videoFileName = videoDict[str(vidID)]
		if verbose: print('Starting ' + videoFileName + '... ')
		cap = cv2.VideoCapture(videoFileName)
		ret = True
		fnum = 0
		vidCount = vidCount + 1
		while ret:
			ret, frame = cap.read(fnum)
			if(ret==True):
				frame_R = frame[:,:,0].astype('float32')
				if frame_filter_size is not None:
					frame_R = cv2.medianBlur(frame_R, frame_filter_size).astype('uint8')
				frame_R = frame_R[cropROI[0][1]:cropROI[1][1],cropROI[0][0]:cropROI[1][0]]
				frame_R = frame_R[::spatialDownSample, ::spatialDownSample] # spatial downsampling
				if show_vid:
					cv2.imshow('frame', frame_R/255)
					cv2.waitKey(1)
				frameMat_r[frameVect[frameIndex], :, :] = np.uint8(frame_R)

				if getRGB is True:
					frame_G = frame[:,:,1].astype('float32')
					if frame_filter_size is not None:
						frame_G = cv2.medianBlur(frame_G, frame_filter_size).astype('uint8')
					frame_G = frame_G[cropROI[0][1]:cropROI[1][1],cropROI[0][0]:cropROI[1][0]]
					frame_G = frame_G[::spatialDownSample, ::spatialDownSample] # spatial downsampling
					frameMat_g[frameVect[frameIndex], :, :] = np.uint8(frame_G)

					frame_B = frame[:,:,2].astype('float32')
					if frame_filter_size is not None:
						frame_B = cv2.medianBlur(frame_B, frame_filter_size).astype('uint8')
					frame_B = frame_B[cropROI[0][1]:cropROI[1][1],cropROI[0][0]:cropROI[1][0]]
					frame_B = frame_B[::spatialDownSample, ::spatialDownSample] # spatial downsampling
					frameMat_b[frameVect[frameIndex], :, :] = np.uint8(frame_B)

				frameIndex = frameIndex + 1
				fnum = fnum + 1
		cap.release()
		if verbose: print('     done - ' + str(vidCount) + ' of ' + str(numVids))
	if show_vid:
		cv2.destroyAllWindows()
	# downsample the matrix
	frameMat_r = frameMat_r[0:frameIndex:temporalDownSample, :, :] # temporal downsampling
	if getRGB is True:
		frameMat_g = frameMat_g[0:frameIndex:temporalDownSample, :, :] # temporal downsampling
		frameMat_b = frameMat_b[0:frameIndex:temporalDownSample, :, :] # temporal downsampling
	return frameMat_r, frameMat_g, frameMat_b


def crop_and_convert_miniscope(dataFolder, tiff_name='msCam.tiff', avi_names = '',
								cropROI=None, circleDef=[], spatialDownSample=1, temporalDownSample=1,
								frame_filter_size=None, verbose=True, show_vid=False, multicore=1, dff=False):
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
	## frame_filter_size: positive odd int, pixel width of midean blur filter cv2.medianBlur
	
	videoDict, videoNumList, numVids = getVideosInFolder(dataFolder, avi_names)
	videoList = [videoDict[str(0)]]
	for key in np.arange(1, len(videoDict)):
		videoList.append(videoDict[str(key)])
	
	if os.path.isabs(tiff_name) == False:
		tiffStackout = dataFolder + '/' + tiff_name
	else:
		tiffStackout = tiff_name
	
	if cropROI is None:
		cap = cv2.VideoCapture(videoDict['0'])
		width, height = (
				int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
				int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
			)
		cropROI = [[0, 0],[width, height]] # I know its width by height my bad was 1, 0
	elif cropROI == []:
		cropROIs, maskROI = get_all_CropROIs([dataFolder], avi_names=avi_names, shape='rect', verbose=verbose)
		cropROI = cropROIs[dataFolder]

	# cap.release()
	# cv2.destroyAllWindows()

	start_time = time.time()
	####################
	if show_vid:
		cv2.namedWindow("frame")
	####################
	# define mask
	if bool(circleDef):
		center = circleDef[0]
		radius = circleDef[1]
		Y, X = np.ogrid[:originalHeight, :originalWidth]
		dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
		mask = dist_from_center > radius
	else:
		mask = []
	if multicore > 1:
		print('     ~~~ Combining  ' + str(numVids) + ' video in ' + dataFolder + ' ~~~')
		print('     ~~~ MULTIPROCESSING ~~~')
		import multiprocessing as mp
		from miniscope_tiffs import miniscope_avi_multiprocessing, miniscope_multiprocessing_combine
		from functools import partial
		num_processes = multicore # mp.cpu_count()
		print("Number of CPU: " + str(num_processes))
		# frameMat_multi = multi_process_avis()
		print("Video processing using {} processes...".format(num_processes))
		# Parallel execution across multiple videos
		p = mp.Pool(num_processes)
		frameMat_multi = p.map(miniscope_avi_multiprocessing, videoList)
		frameMat = miniscope_multiprocessing_combine(frameMat_multi, videoDict, cropROI, frame_filter_size, tds=temporalDownSample, sds=spatialDownSample)
		frameIndex = frameMat.shape[0]
	else:
		print('     ~~~ Combining  ' + str(numVids) + ' video in ' + dataFolder + ' ~~~')
		frameMat = load_miniscope_avis(dataFolder=dataFolder, avi_names=avi_names, cropROI=cropROI, spatialDownSample=spatialDownSample, temporalDownSample=temporalDownSample, frame_filter_size=frame_filter_size, verbose=verbose, show_vid=show_vid)
		frameIndex = frameMat.shape[0]
		# load_miniscope_avis(dataFolder, avi_names = '', spatialDownSample=1, temporalDownSample=1, frame_filter_size=None, verbose=False, show_vid=False)
	
	end_time = time.time()
	total_processing_time = end_time - start_time
	print("Time taken: {}".format(total_processing_time))
	
	# save the parameters used
	if tiff_name != 'test.tiff':
		saveName = dataFolder + '\Crop_params.mat'
		if frame_filter_size is None: frame_filter_size = 'None'
		if mask is None: mask = 'None'
		scipy.io.savemat(saveName, {'cropROI':cropROI, 'spatialDownSample':spatialDownSample,
			'temporalDownSample':temporalDownSample, 'totalFrames':frameIndex, 'tiffStackout':tiffStackout,
			'mask':mask, 'frame_filter_size':frame_filter_size})
		print('Parameters saved to:  ' + saveName)
	else:
		print('TESTING - Crop params not saved')
	
	# alternative writing method with TiffWriter
	# tifs = np.array([tifffile.imread(t) for t in fls])
	# with TiffWriter(tiffStackout, bigtiff=True) as tiff:
	# 	for i in range(0, frameMat.shape[0]):
	# 		tiff.save(frameMat[i,:,:], compression='None')
	# tiffStackout = dataFolder + '/alt_' + tiff_name
	if dff==True:
		print('Subtracting min and scaling!')
		frameMat = np.uint8(2*(frameMat - np.min(frameMat, axis=0))  + 30)
	imwrite(tiffStackout, frameMat, bigtiff=True)
	print('Done! Movie saved to:  ' + tiffStackout)
	return tiffStackout

def crop_and_convert_colorAVI(dataFolder, aviOUT_name='test.avi', avi_names = '',
								cropROI=None, circleDef=[], spatialDownSample=1, temporalDownSample=1,
								frame_filter_size=None, verbose=True, show_vid=False, multicore=1):
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
	## frame_filter_size: positive odd int, pixel width of midean blur filter cv2.medianBlur
	videoDict, videoNumList, numVids = getVideosInFolder(dataFolder, avi_names)
	videoList = [videoDict[str(0)]]
	for key in np.arange(1, len(videoDict)):
		videoList.append(videoDict[str(key)])
	
	if os.path.isabs(aviOUT_name) == False:
		vidname = dataFolder + '/' + aviOUT_name
	else:
		vidname = aviOUT_name
	
	if cropROI is None:
		cap = cv2.VideoCapture(videoDict['0'])
		width, height = (
				int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
				int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
			)
		cropROI = [[0, 0],[width, height]] # I know its width by height my bad was 1, 0
	elif cropROI == []:
		cropROIs, _ = get_all_CropROIs([dataFolder], avi_names=avi_names, shape='rect', verbose=verbose)
		cropROI = cropROIs[dataFolder]

	# cap.release()
	# cv2.destroyAllWindows()

	start_time = time.time()
	####################
	if show_vid:
		cv2.namedWindow("frame")
	####################
	# define mask
	if bool(circleDef):
		center = circleDef[0]
		radius = circleDef[1]
		Y, X = np.ogrid[:originalHeight, :originalWidth]
		dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
		mask = dist_from_center > radius
	else:
		mask = []
	if (multicore > 1) or (multicore==True):
		print('     ~~~ Combining  ' + str(numVids) + ' video in ' + dataFolder + ' ~~~')
		print('     ~~~ MULTIPROCESSING ~~~')
		import multiprocessing as mp
		from miniscope_tiffs import miniscope_avi_multiprocessing, miniscope_multiprocessing_combine
		from functools import partial
		if multicore == True:
			num_processes = mp.cpu_count()
		else:
			num_processes = multicore # mp.cpu_count()
		print("Number of CPU: " + str(num_processes))
		# frameMat_multi = multi_process_avis()
		print("Video processing using {} processes...".format(num_processes))
		# Parallel execution across multiple videos
		p = mp.Pool(num_processes)
		# frameMat_multi = p.map(partial(miniscope_avi_multiprocessing, fileDir=dataFolder), np.arange(numVids+1))
		partial(miniscope_avi_multiprocessing_RGB, getRGB=True)
		frameMat_mR, frameMat_mG, frameMat_mB = p.map(miniscope_avi_multiprocessing_RGB, videoList)
		frameMat_R = miniscope_multiprocessing_combine(frameMat_mR, videoDict, cropROI, frame_filter_size, tds=temporalDownSample, sds=spatialDownSample)
		frameMat_G = miniscope_multiprocessing_combine(frameMat_mG, videoDict, cropROI, frame_filter_size, tds=temporalDownSample, sds=spatialDownSample)
		frameMat_B = miniscope_multiprocessing_combine(frameMat_mB, videoDict, cropROI, frame_filter_size, tds=temporalDownSample, sds=spatialDownSample)
		frameIndex = frameMat_R.shape[0]
	else:
		print('     ~~~ Combining  ' + str(numVids) + ' video in ' + dataFolder + ' ~~~')
		frameMat_R, frameMat_G, frameMat_B = load_miniscope_avis(dataFolder=dataFolder, avi_names=avi_names, cropROI=cropROI, spatialDownSample=spatialDownSample,
								 temporalDownSample=temporalDownSample, frame_filter_size=frame_filter_size, verbose=verbose, show_vid=show_vid, getRGB=True)
		frameIndex = frameMat_R.shape[0]
		# load_miniscope_avis(dataFolder, avi_names = '', spatialDownSample=1, temporalDownSample=1, frame_filter_size=None, verbose=False, show_vid=False)
	
	end_time = time.time()
	total_processing_time = end_time - start_time
	print("Time taken: {}".format(total_processing_time))
	
	# save the parameters used
	if aviOUT_name != 'test.avi':
		saveName = dataFolder + '\Crop_params.mat'
		if frame_filter_size is None: frame_filter_size = 'None'
		if mask is None: mask = 'None'
		scipy.io.savemat(saveName, {'cropROI':cropROI, 'spatialDownSample':spatialDownSample,
			'temporalDownSample':temporalDownSample, 'totalFrames':frameIndex, 'vidname':vidname,
			'mask':mask, 'frame_filter_size':frame_filter_size})
		print('Parameters saved to:  ' + saveName)
	else:
		print('TESTING - Crop params not saved')
	
	[nf, h, w] = frameMat_R.shape
		
	fps = 30
	# out = cv2.VideoWriter(vidname, cv2.VideoWriter_fourcc(*'XVID'), fps, (h, w), False)
	out = cv2.VideoWriter(vidname, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
	# out = cv2.VideoWriter(aviname, -1, fps, (h, w), True)
	for ind in np.arange(0, nf):
		# data = np.random.randint(0, 256, (h,w,3), dtype='uint8')
		r = frameMat_R[ind, :,:]
		g = frameMat_G[ind, :,:]
		b = frameMat_B[ind, :,:]
		data = np.dstack((r, g, b))
		out.write(data)
		# cv2.imshow('frame',data)
		# if cv2.waitKey(1) & 0xFF == ord('q'):
		#     break
	out.release()

	print('Done! Movie saved to:  ' + vidname)
	return frameMat_R, frameMat_G, frameMat_B

def combine_multi_folders(dataFolderList, output_folder, tiff_name='msCam.tiff', avi_names = '',
								cropROI=None, circleDef=[], spatialDownSample=1, temporalDownSample=1,
								frame_filter_size=None, verbose=True, show_vid=False):
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
	## frame_filter_size: matrix for filtering each frame using cv2.filter2D
	# videoDict = {}
	# videoNumList = []
	# numVids = []
	ind = 0
	metaFile = open('metaData.json')
	metaData = json.load(metaFile)
	metaFile.close()
	MaxFramesPerFile = metaData['framesPerFile'] # used guess the total possible frames and preallocate memory
	timeStampFname = 'timeStamps.csv'
	OriBNOFname = 'headOrientation.csv'

	for dataFolder in dataFolderList:
		print(dataFolder)
		# dataFolder = dataFolder_sub + '/MiniLFOV/'
		if ind == 0:
			videoDict, videoNumList, numVids = getVideosInFolder(dataFolder, avi_names)
			firstVideoNum = videoNumList[0]
			tiffStackout = output_folder + '/' + tiff_name

			tsFile = output_folder  + '/' + timeStampFname
			oriFile = output_folder + '/' + OriBNOFname 
			# Get a sample frame from the first video
			videoFileName = videoDict[str(firstVideoNum)]
			cap = cv2.VideoCapture(videoFileName)
			ret, refFrame = cap.read(0)
			originalHeight = refFrame.shape[0]
			originalWidth = refFrame.shape[1]	
			
			if cropROI is None:
				cropROI = [[0, 0],[refFrame.shape[1], refFrame.shape[0]]] # I know its width by height my bad was 1, 0
			elif cropROI == []:
				cropROIs, maskROI = get_all_CropROIs([dataFolder], avi_names=avi_names, shape='rect', verbose=verbose)
				cropROI = cropROIs[dataFolder]

			frameCrop = refFrame[cropROI[0][1]:cropROI[1][1], cropROI[0][0]:cropROI[1][0], 0]
			# frameRate = int(cap.get(5))
			frameCrop = frameCrop[::spatialDownSample, ::spatialDownSample]
			frameHeight = frameCrop.shape[0]
			frameWidth = frameCrop.shape[1]

			cap.release()
			cv2.destroyAllWindows()
		else:
			sub_videoFiles, sub_videoNumList, sub_numVids = getVideosInFolder(dataFolder, avi_names)
			for v in sub_videoNumList:
				newnum = ind*100 + v # str(ind) + '_' + str(v)
				print(newnum)
				videoNumList = np.append(videoNumList, newnum)
				videoDict[str(newnum)] = sub_videoFiles[str(v)]
			numVids = numVids + sub_numVids
		ind = ind+1
		
	print(videoDict)
	###################
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
	print(videoNumList)
	for vidID in videoNumList:
		videoFileName = videoDict[str(vidID)]
		if verbose: print('Starting ' + videoFileName + '... ')
		cap = cv2.VideoCapture(videoFileName)
		ret = True
		fnum = 0
		vidCount = vidCount + 1
		while ret:
			ret, frame = cap.read(fnum)
			if(ret==True):
				frame = frame[:,:,0].astype('uint8')
				if frame_filter_size is not None:
					frame = cv2.medianBlur(frame, frame_filter_size).astype('uint8')
					# scypy image median filter is >100x slower :(
					# frame = spim.median_filter(frame, footprint=frame_filter_size) # .astype('uint8')
					# frame = cv2.filter2D(frame, -1, frame_filter_size)
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

	# save the parameters used
	saveName = output_folder + '\Crop_params.mat'
	if frame_filter_size is None: frame_filter_size = 'None'
	if mask is None: mask = 'None'

	scipy.io.savemat(saveName, {'cropROI':cropROI, 'spatialDownSample':spatialDownSample,
		'temporalDownSample':temporalDownSample, 'totalFrames':frameIndex, 'tiffStackout':tiffStackout,
		'mask':mask, 'frame_filter_size':frame_filter_size})
	
	print('Parameters saved to:  ' + saveName)
	# alternative writing method with TiffWriter
	# tifs = np.array([tifffile.imread(t) for t in fls])
	# with TiffWriter(tiffStackout, bigtiff=True) as tiff:
	# 	for i in range(0, frameMat.shape[0]):
	# 		tiff.save(frameMat[i,:,:], compression='None')
	# tiffStackout = dataFolder + '/alt_' + tiff_name
	imwrite(tiffStackout, frameMat, bigtiff=True)
	print('Done! Movie saved to:  ' + tiffStackout)

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
		rr = refPt+2*circRad
		# print(refInput.shape)
		# print(refPt)
		# print(refNew)
		# print(refPt+2*circRad)
		cv2.rectangle(refInput, refPt, (rr[0], rr[1]), (0, 0, 255), 2)	
		# cv2.rectangle(refInput, refPt, refPt+2*circRad, (0, 0, 255), 2)	# -1*circRad
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

def get_all_CropROIs(miniscopeFolders, avi_names='', shape='rect', verbose=True, load_all=False):
	print('     ~~~ Cropping ' + str(len(miniscopeFolders)) + ' folders ~~~')
	from miniscope_tiffs import draw_circ_mask
	global refPt, refNew, rectPt, shape2, circMid, circRad, refFrame
	roiDict = {} # dictionary for rois
	maskDict = {} # dictionary for circles
	for sessionLoop in range(0, len(miniscopeFolders)):
		if True:
			dataFolder = miniscopeFolders[sessionLoop]
			videoDict, videoNumList, numVids = getVideosInFolder(dataFolder, avi_names)
			videoList = [videoDict[str(0)]]
			for key in np.arange(1, len(videoDict)):
				videoList.append(videoDict[str(key)])
			
			if load_all==True:
				tiff_name = str(dataFolder + '/behavCam.tiff')
				if os.path.isfile(tiff_name):
					print('Reading in existing tiff: ' + tiff_name)
					frameMat = imread(str(dataFolder + '/behavCam.tiff'))
				else:
					print('Loading all avis for reference image, may take a while')
					multicore = True
					if multicore > 1:
						print('     ~~~ Combining  ' + str(numVids) + ' video in ' + dataFolder + ' ~~~')
						print('     ~~~ MULTIPROCESSING ~~~')
						import multiprocessing as mp
						from miniscope_tiffs import miniscope_avi_multiprocessing, miniscope_multiprocessing_combine
						from functools import partial
						# from process_video_multiprocessing import miniscope_avi_multiprocessing, miniscope_multiprocessing_combine
						
						num_processes = multicore # mp.cpu_count()
						print("Number of CPU: " + str(num_processes))
						# frameMat_multi = multi_process_avis()
						print("Video processing using {} processes...".format(num_processes))
						# Parallel execution across multiple videos
						p = mp.Pool(num_processes)
						partial(miniscope_avi_multiprocessing, getRGB=False)
						frameMat_multi, _, _ = p.map(miniscope_avi_multiprocessing, videoList)
						frameMat = miniscope_multiprocessing_combine(frameMat_multi, videoDict, cropROI, frame_filter_size, tds=temporalDownSample, sds=spatialDownSample)
					else:
						print('     ~~~ Combining  ' + str(numVids) + ' video in ' + dataFolder + ' ~~~')
						frameMat, _, _ = load_miniscope_avis(dataFolder, avi_names = '', spatialDownSample=1, temporalDownSample=4, frame_filter_size=None, verbose=False, show_vid=False)
				refFrame = np.max(frameMat, axis=0).astype('uint8')
				refFrame = refFrame - np.min(refFrame)
				refFrame = 255*(refFrame/np.max(refFrame))
				refFrame[refFrame>255] = 255
				refFrame = np.uint8(refFrame)
				# b = np.min(frameMat, axis=0).astype('uint8')
				# refFrame = a-b
			else: # 
				videoDict, videoNumList, numVids = getVideosInFolder(dataFolder, avi_names)
				videoFileName = videoDict[str(videoNumList[0])]
				cap = cv2.VideoCapture(videoFileName)
				ret = False
				i = 0
				while ret == False:
					ret, refFrame = cap.read(0) # with trigger start the 0 frame is black?	
					i = i+1
					if i>500:
						break
			clone = copy.deepcopy(refFrame)

			cv2.namedWindow("refFrame")
			cv2.setMouseCallback("refFrame", draw_circ_mask)

			
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

def crop_tiff(frameMat, shape='rect'):
	from miniscope_tiffs import draw_circ_mask
	global refPt, refNew, rectPt, shape2, circMid, circRad, refFrame
	# refFrame = np.mean(frameMat, axis=0)
	refFrame = frameMat[0]
	clone = copy.deepcopy(refFrame)
	# print(refFrame.shape)
	cv2.namedWindow("refFrame")
	cv2.setMouseCallback("refFrame", draw_circ_mask)

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
		print(rectPt)
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

	if shape2=='rectangle' or shape2=='square':
		mask = []
	else:
		mask = [circMid, circRad]
	# print(miniscopeFolders[sessionLoop])
	cv2.destroyAllWindows()
	frameMat = frameMat[:, refROI[0][1]:refROI[1][1], refROI[0][0]:refROI[1][0]]
	return refROI , mask, frameMat

def miniscope_avi_multiprocessing(aviFileName): # aviNum, fileDir):
	'''
	Args:
		aviFileName   -   full path of avi file to be read in
	Returns:
		numpy aray of avi data (frames x hpixels x wpixels)
	based off the method described by Sabina Pokhrel at:
	https://towardsdatascience.com/faster-video-processing-in-python-using-parallel-computing-25da1ad4a01
	'''
	# fname = fileDir + '/' + str(aviNum) + '.avi'
	cap = cv2.VideoCapture(aviFileName)
	#     cap.set(cv2.CAP_PROP_POS_FRAMES)
	# get height, width and frame count of the video
	width, height = (
			int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
			int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		)
	no_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	frameMat_r = np.empty([no_of_frames, height, width], dtype='uint8')
	fnum = 0
	ret = True
	try:
		while ret:
			ret, refFrame = cap.read(fnum)
			if not ret:
				break
			frame = refFrame[:,:,0].astype('uint8')
			frameMat_r[fnum, :, :] = frame
			fnum = fnum + 1   
	except:
		# Release resources
		cap.release()
	frameMat_r = frameMat_r[0:fnum, :, :] 

	# Release resources
	cap.release()
	return frameMat_r

def miniscope_avi_multiprocessing_RGB(aviFileName, getRGB=True): # aviNum, fileDir):
	'''
	Args:
		aviFileName   -   full path of avi file to be read in
	Returns:
		numpy aray of avi data (frames x hpixels x wpixels)
	based off the method described by Sabina Pokhrel at:
	https://towardsdatascience.com/faster-video-processing-in-python-using-parallel-computing-25da1ad4a01
	'''
	# fname = fileDir + '/' + str(aviNum) + '.avi'
	cap = cv2.VideoCapture(aviFileName)
	#     cap.set(cv2.CAP_PROP_POS_FRAMES)
	# get height, width and frame count of the video
	width, height = (
			int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
			int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		)
	no_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	frameMat_r = np.empty([no_of_frames, height, width], dtype='uint8')
	if getRGB is True:
		frameMat_g = np.empty([no_of_frames, height, width], dtype='uint8')
		frameMat_b = np.empty([no_of_frames, height, width], dtype='uint8')
	fnum = 0
	ret = True
	try:
		while ret:
			ret, refFrame = cap.read(fnum)
			if not ret:
				break
			frame = refFrame[:,:,0].astype('uint8')
			frameMat_r[fnum, :, :] = frame
			if getRGB is True:
				frame = refFrame[:,:,1].astype('uint8')
				frameMat_g[fnum, :, :] = frame
				frame = refFrame[:,:,2].astype('uint8')
				frameMat_b[fnum, :, :] = frame
			fnum = fnum + 1   
	except:
		# Release resources
		cap.release()
	frameMat_r = frameMat_r[0:fnum, :, :] 
	if getRGB is True:
		frameMat_g = frameMat_g[0:fnum, :, :] 
		frameMat_b = frameMat_b[0:fnum, :, :] 
	else:
		frameMat_g = None
		frameMat_b = None
	# Release resources
	cap.release()
	return frameMat_r, frameMat_g, frameMat_b

def miniscope_multiprocessing_combine(frameMat_multi, videoDict, cropROI, frame_filter_size, tds=1, sds=1):
	frames_per_file, height, width = (frameMat_multi[0].shape)
	totalAvis = len(frameMat_multi)-1
	avi_nums = np.arange(totalAvis)
	totalFrames = frames_per_file*(totalAvis+1)
	frameMat = np.empty([totalFrames, height, width], dtype='uint8')
	for aviNum in avi_nums:
		f1 = aviNum*frames_per_file
		f2 = (aviNum+1)*frames_per_file   
		ff = frameMat_multi.pop(0)
		frameMat[f1:f2, :, :] = ff
	
	# get the last video files, that might have fewer frames than the rest
	no_of_frames_last = frameMat_multi[0].shape[0]
	f1 = (totalAvis)*frames_per_file
	f2 = (totalAvis)*frames_per_file + no_of_frames_last
	frameMat[f1:f2, :, :] = frameMat_multi.pop(0)

	# temporal downsampling
	frameMat = frameMat[0:f2:tds]
	# crop to ROI
	frameMat = frameMat[:, cropROI[0][1]:cropROI[1][1],cropROI[0][0]:cropROI[1][0]]
	if frame_filter_size is not None:
		import scipy.ndimage as spim
		for ind, frame in enumerate(frameMat):
			frame = cv2.medianBlur(frame, frame_filter_size).astype('uint8')
			frameMat[ind,:,:] = frame
	# spatial downsampling
	frameMat = frameMat[:, ::sds, ::sds]
	return frameMat