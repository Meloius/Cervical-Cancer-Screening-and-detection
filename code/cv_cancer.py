import cv2 as cv
import numpy as np
import time

def nothing(x):
    pass
"""
cap = cv2.VideoCapture(0)
cv2.namedWindow('frame')

fourcc = cv2.cv.CV_FOURCC(*'XVID')
#out = cv2.VideoWriter("cervix.avi", fourcc, 30, (640, 360))


# create trackbars for color change
#cv2.createTrackbar('R','image',0,255,nothing)
#cv2.createTrackbar('G','image',0,255,nothing)
#cv2.createTrackbar('B','image',0,255,nothing)

# create switch for ON/OFF functionality
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'frame',0,1,nothing)


#	capturing video frames

fps = 30
size = (int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))/2, int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))/2)

out = cv2.VideoWriter("cervix.avi", fourcc, fps, size)

success, frame = cap.read()
numFramesRemaining = 10 * fps - 1

while success and numFramesRemaining > 0:
    	#ret, frame = cap.read()
    	cv2.imshow("frame", frame)
    
    	out.write(frame)
    
    	# get current positions of four trackbars
    	r = cv2.getTrackbarPos('R','frame')
    	g = cv2.getTrackbarPos('G','frame')
    	b = cv2.getTrackbarPos('B','frame')
    	s = cv2.getTrackbarPos(switch,'frame')

    	if r == 0:
        	frame[:] = 0
    	else:
        	frame[:] = [b,g,r]

	success, frame = cap.read()
	numFramesRemaining -= 1

    	key = cv2.waitKey(25)
    	if key == 27:
        	break

out.release()
cap.release()
cv2.destroyWindow("frame")

"""
class CaptureManager(object):

	def __init__(self, channel, previewWindowManager = None, shouldMirrorPreview = False):
		"""
		capture -->	open / allow / output camera feed
		channel -->	camera source
			0 	->	in-built / webcam
			1 	-> 	usb cam
		 
		"""
		self.previewWindowManager = previewWindowManager
		self.shouldMirrorPreview = shouldMirrorPreview
		
		self._channel = channel
		self._capture = cv.VideoCapture(self._channel)
		
		self._enteredFrame = False
		self._frame = None
		
		self._imageFilename = None
		self._videoFilename = None
		self._videoEncoding = None
		self._videoWriter = None
		    
		self._startTime = None
		self._framesElapsed = long(0)
		self._fpsEstimate = None
		
	@property
	def isWritingImage(self):
		return self._imageFilename is not None
    
	@property
	def isWritingVideo(self):
		return self._videoFilename is not None
		
	def writeImage(self, filename):
		"""Write the next exited frame to an image file."""
		self._imageFilename = filename
		#cv.imwrite(self._imageFilename)
    
	def startWritingVideo(self, filename, encoding = cv.cv.CV_FOURCC('M','J','P','G')):
		"""Start writing exited frames to a video file."""
		self._videoFilename = filename
		self._videoEncoding = encoding
    
	def stopWritingVideo(self):
		"""Stop writing exited frames to a video file."""
		self._videoFilename = None
		self._videoEncoding = None
		self._videoWriter = None


	@property
	def frame(self):
		if self._enteredFrame and self._frame is None:
			_, self._frame = self._capture.retrieve(channel = self._channel)
		return self._frame
        
	def enterFrame(self):
		"""Capture the next frame, if any."""
        
		# But first, check that any previous frame was exited.
		#assert not self._enteredFrame, \
		#		'previous enterFrame() had no matching exitFrame()'
        
		if self._capture is not None:
			self._enteredFrame = self._capture.grab()	

	def exitFrame(self):
		"""Draw to the window. Write to files. Release the frame."""
        
		# Check whether any grabbed frame is retrievable.
		# The getter may retrieve and cache the frame.
		if self.frame is None:
			self._enteredFrame = False
			return
        
		# Update the FPS estimate and related variables.
		if self._framesElapsed == 0:
			self._startTime = time.time()
		else:
			timeElapsed = time.time() - self._startTime
			self._fpsEstimate =  self._framesElapsed / timeElapsed
		self._framesElapsed += 1
        
		# Draw to the window, if any.
		if self.previewWindowManager is not None:
			if self.shouldMirrorPreview:
				mirroredFrame = np.fliplr(self._frame).copy()
				self.previewWindowManager.show(mirroredFrame)
			else:
				self.previewWindowManager.show(self._frame)
        
		# Write to the image file, if any.
		if self.isWritingImage:
			cv.imwrite(self._imageFilename, self._frame)
			self._imageFilename = None
        
		# Write to the video file, if any.
		self._writeVideoFrame()
        
		# Release the frame.
		self._frame = None
		self._enteredFrame = False

	def _writeVideoFrame(self):

	        if not self.isWritingVideo:
        	    return
        
        	if self._videoWriter is None:
        		print('pass')
            		fps = self._capture.get(cv.cv.CV_CAP_PROP_FPS)
            		if fps <= 0.0:
                		# The capture's FPS is unknown so use an estimate.
                		if self._framesElapsed < 20:
                    			# Wait until more frames elapse so that the
                    			# estimate is more stable.
                    			return
                		else:
                    			fps = self._fpsEstimate
            				size = (int(self._capture.get(
                        				cv.cv.CV_CAP_PROP_FRAME_WIDTH)),
                    				int(self._capture.get(
                        				cv.cv.CV_CAP_PROP_FRAME_HEIGHT)))
            				self._videoWriter = cv.VideoWriter(
                				self._videoFilename, self._videoEncoding,
               					fps, size)
        
        	self._videoWriter.write(self._frame)
	
class WindowManager(object):
	
	def __init__(self, windowName, keypressCallback = None):
	
		self.keypressCallback = keypressCallback
		self._windowName = windowName
		self._isWindowCreated = False
		
	@property
	def isWindowCreated(self):
		return self._isWindowCreated
    
	def createWindow(self):
		cv.namedWindow(self._windowName)
		self._isWindowCreated = True
		#self._isWindowCreated, self._frame = self._capture.read()
    
	def show(self, frame):
		cv.imshow(self._windowName, frame)
		#out.write(self._frame) 
    
	def destroyWindow(self):
		cv.destroyWindow(self._windowName)
		self._isWindowCreated = False
    
	def processEvents(self):
		#out.release() #CLOSE VIDEO WRITER	
		#cap.release() #CLOSE THE CAM FEED
		keycode = cv.waitKey(25)
		if self.keypressCallback is not None and keycode != -1:
			# Discard any non-ASCII info encoded by GTK.
			keycode &= 0xFF
			self.keypressCallback(keycode)



class CervicalCancer(object):

	"""
	We blend everything together here.
	
		capture manager - ( images & video output )
		window manager  - windows created
		keyInterrupt   - GUI buttons
	
	"""

	def __init__(self):
		self._windowManager = WindowManager('AIR labs Diagnostics', self.keyInterrupt)
		self._captureManager = CaptureManager(0, self._windowManager, True)
	
	
	def main(self):
		self._windowManager.createWindow()
		#print(1)
		while self._windowManager.isWindowCreated:
			self._captureManager.enterFrame()
			frame = self._captureManager.frame
			self._captureManager.enterFrame()
			#self._windowManager.show(frame)
			self._captureManager.exitFrame()
			self._windowManager.processEvents()
			
			
	def keyInterrupt(self, keycode):
		"""Handle a keypress.

        space  -> Take a screenshot.
        tab    -> Start/stop recording a screencast.
        x      -> Start/stop drawing debug rectangles around faces.
        escape -> Quit.
        
		"""
		if keycode == 32: # space
			self._captureManager.writeImage('testshot.png')
		elif keycode ==9: # tab
			if not self._captureManager.isWritingVideo:
				self._captureManager.startWritingVideo('testcast.avi')
			else:
				self._captureManager.stopWritingVideo()
		elif keycode == 27: # escape
			self._windowManager.destroyWindow()

		
		 	

if __name__ == '__main__':
	cancer = CervicalCancer()
	cancer.main()

