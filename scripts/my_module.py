import cv2
import cv_bridge
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from cv_bridge.boost.cv_bridge_boost import getCvType
import pickle
def my_function(img): 
	# image=pickle.loads(img)	
	a=getCvType("8UC3")
	img = CvBridge().imgmsg_to_cv2(img,desired_encoding="bgr8")
	a=1
	return a

