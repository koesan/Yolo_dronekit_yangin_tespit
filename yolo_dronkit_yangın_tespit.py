import sys
import time
import cv2
import numpy as np
from dronekit import connect, VehicleMode
from pymavlink import mavutil
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import rospy

# ROS üzerinden alınacak kamera verisi
webcam = "/webcam/image_raw"

# YOLO model dosyaları ve etiket isimleri
config = "yolo/my_yolov4.cfg"
weights = "yolo/my_yolov4_final.weights"
names = "yolo/_darknet.names"

# YOLO modelini OpenCV ile yükleme
model = cv2.dnn.readNetFromDarknet(config, weights)

# Etiket isimlerini yükleme
with open(names, 'r') as f:
    sınıf = f.read().strip().split('\n')

# Dron bağlantıları
connection_string = "127.0.0.1:14550"
print("Drona Bağlanılıyor...")
iha=connect(connection_string,wait_ready=True,timeout=100)
iha.flush()
print("Drona Bağlanıldı.")
time.sleep(2)

# Yangın tespit fonksiyonu
def tespit(img):

  h, w = img.shape[:2]
  blob = cv2.dnn.blobFromImage(img, 1/255.0, (w, h), swapRB=True, crop=False)
  model.setInput(blob)
  outputs = model.forward(model.getUnconnectedOutLayersNames())
  
  for output in outputs:
    for detection in output:
      scores = detection[5:]
      class_id = np.argmax(scores)
      confidence = scores[class_id]

      if confidence > 0.5:
        center_x = int(detection[0] * w)
        center_y = int(detection[1] * h)
        width = int(detection[2] * w)
        height = int(detection[3] * h)

        x = int(center_x - width/2)
        y = int(center_y - height/2)

        cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2)
        etiket = "YANGIN!"
        cv2.putText(img, etiket, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
          
  cv2.imshow('Pencere', img)
  cv2.waitKey(1)

# Dronun hareketini kontrol eden fonksiyon
def hareket(vx, vy, vz):
  msg = iha.message_factory.set_position_target_local_ned_encode( 
      0,
      0, 0,
      mavutil.mavlink.MAV_FRAME_BODY_NED,
      0b0000111111000111, 
      0, 0, 0, 
      vx, vy, vz, 
      0, 0, 0,
      0, 0)
  iha.send_mavlink(msg)
  iha.flush()

# Dronu arm ve kalkış için hazırlayan fonksiyon
def yuksel(hedef_yukseklik):
	while iha.is_armable==False:
		print("Arm ici gerekli sartlar saglanamadi.")
		time.sleep(1)
	print("Iha su anda armedilebilir")
	
	iha.mode=VehicleMode("GUIDED")
	while iha.mode=='GUIDED':
		print('Guided moduna gecis yapiliyor')
		time.sleep(1.5)

	print("Guided moduna gecis yapildi")
	iha.armed=True
	while iha.armed is False:
		print("Arm icin bekleniliyor")
		time.sleep(1)

	print("Ihamiz arm olmustur")
	
	iha.simple_takeoff(hedef_yukseklik)
	while iha.location.global_relative_frame.alt<=hedef_yukseklik*0.94:
		print("Su anki yukseklik{}".format(iha.location.global_relative_frame.alt))
		time.sleep(0.5)
	print("Takeoff gerceklesti")

class image_converter:
  def __init__(self):
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber(webcam,Image,self.callback)

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        
    except CvBridgeError as e:
      print(e)

    hareket(1, 0, 0)
    tespit(cv_image)

def main(args):
    
    # Dronun yüksekliğe kalkış yapması
    yuksel(10)
    time.sleep(2)

    # ROS düğümünü başlatma ve görüntü dönüştürücü nesnesini oluşturma
    ic = image_converter()
    rospy.init_node('image_converter', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("kapatılıyor")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)