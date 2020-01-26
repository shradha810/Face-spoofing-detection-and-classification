import cv2, dlib
import numpy as np
from imutils import face_utils
from keras.models import load_model
from imutils.video import VideoStream
from sklearn.externals import joblib

IMG_SIZE = (34, 26)
blinking=0
ope=0
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r'C:\Users\Navya\Desktop\major\blink-detection\blink-detection\shape_predictor_68_face_landmarks.dat')

model = load_model(r'C:\Users\Navya\Desktop\major\blink-detection\blink-detection\eye_blinkcnn.h5')
model.summary()

def crop_eye(img, eye_points):
  x1, y1 = np.amin(eye_points, axis=0)
  x2, y2 = np.amax(eye_points, axis=0)
  cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

  w = (x2 - x1) * 1.2
  h = w * IMG_SIZE[1] / IMG_SIZE[0]

  margin_x, margin_y = w / 2, h / 2

  min_x, min_y = int(cx - margin_x), int(cy - margin_y)
  max_x, max_y = int(cx + margin_x), int(cy + margin_y)

  eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)

  eye_img = gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

  return eye_img, eye_rect

cap=VideoStream(src=0).start()
count=0
while True:
  ret,img_ori = cap.read()
  
  img_ori = cv2.resize(img_ori, dsize=(0, 0), fx=0.5, fy=0.5)

  img = img_ori.copy()
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  faces = detector(gray)

  for face in faces:
    shapes = predictor(gray, face)
    shapes = face_utils.shape_to_np(shapes)

    eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42])
    eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48])

    eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
    eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
    eye_img_r = cv2.flip(eye_img_r, flipCode=1)

    cv2.imshow('l', eye_img_l)
    cv2.imshow('r', eye_img_r)

    eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.
    eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.

    pred_l = model.predict(eye_input_l)
    pred_r = model.predict(eye_input_r)
    if pred_l <0.1 or pred_r<0.1:
        count=count+1
    # visualize
    state_l = 'O %.1f' if pred_l > 0.5 else '- %.1f'
    state_r = 'O %.1f' if pred_r > 0.5 else '- %.1f'

    state_l = state_l % pred_l
    state_r = state_r % pred_r

    cv2.rectangle(img, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(255,255,255), thickness=2)
    cv2.rectangle(img, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(255,255,255), thickness=2)

    cv2.putText(img, state_l, tuple(eye_rect_l[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(img, state_r, tuple(eye_rect_r[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    if '-' in state_l or '-' in state_r:
       blinking=1
    else:
        ope=1
    if blinking==1 and ope==1:
        cv2.putText(img,'GENUINE',(50,50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)
    
  cv2.imshow('result', img)
  if cv2.waitKey(1) == ord('q'):
    break
if blinking==0:
    print("PHOTO ATTACK")
else:
   print("check for video attack")
   from sklearn.externals import joblib
  

  import scipy.signal as ssg

  def blurriness(image):
      d = 4
      fsize = 2 * d + 1
      kver = np.ones((1, fsize)) / fsize
      khor = kver.T

      Bver = ssg.convolve2d(
          image.astype(
              np.float32), kver.astype(
              np.float32), mode='same')
      Bhor = ssg.convolve2d(
          image.astype(
              np.float32), khor.astype(
              np.float32), mode='same')
      DFver = np.diff(image.astype('int16'), axis=0)
      DFver[np.where(DFver < 0)] = 0
      DFhor = np.diff(image.astype('int16'), axis=1)
      DFhor[np.where(DFhor < 0)] = 0

      DBver = np.abs(np.diff(Bver, axis=0))
      DBhor = np.abs(np.diff(Bhor, axis=1))

      Vver = DFver.astype(float) - DBver.astype(float)
      Vhor = DFhor.astype(float) - DBhor.astype(float)
      Vver[Vver < 0] = 0  
      Vhor[Vhor < 0] = 0  

      SFver = np.sum(DFver)
      SFhor = np.sum(DFhor) 

      SVver = np.sum(Vver)  
      SVhor = np.sum(Vhor)  

      BFver = (SFver - SVver) / SFver
      BFhor = (SFhor - SVhor) / SFhor

      blurF = max(BFver, BFhor)  

      return blurF

  def calColorHist(image, m=100):
      
      numBins = 32
      maxval = 255
      cHist = rgbhist(image, maxval, numBins, 1)

      
      y = sorted(cHist, reverse=True) 
      cHist = y[0:m]                  

      c = np.cumsum(y)                
      numClrs = np.where(c > 0.99)[0][0]    
      return cHist, numClrs




  def rgbhist(image, maxval, nBins, normType=0):
      
      H = np.zeros((nBins, nBins, nBins), dtype=np.uint32)
      decimator = (maxval + 1) / nBins
      numPix = image.shape[1] * image.shape[2]
      im = image.reshape(3, numPix).copy()
      im = im.T

      p = np.floor(im.astype(float) / decimator).astype(np.uint32)
      p2 = np.ascontiguousarray(p).view(
          np.dtype((np.void, p.dtype.itemsize * p.shape[1])))
      unique_p, count = np.unique(p2, return_counts=True)
      unique_p = unique_p.view(p.dtype).reshape(-1, p.shape[1])

      H[unique_p[:, 0], unique_p[:, 1], unique_p[:, 2]] = count

      H = H.ravel() 

      if normType == 1:
          H = H.astype(np.float32) / np.sum(H)  
      return H

  def detect_face(img, faceCascade):
      faces = faceCascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(110, 110)
      )
      return faces


  def calc_hist(img):
      histogram = [0] * 3
      for j in range(3):
          histr = cv2.calcHist([img], [j], None, [256], [0, 256])
          histr *= 255.0 / histr.max()
          histogram[j] = histr
      return np.array(histogram)




      # # Load model
  clf = None

  clf = joblib.load(r"C:\Users\Navya\Desktop\major\replay-attack_ycrcb_luv_extraTreesClassifier.pkl")

  # # Open the camera

  cap = cv2.VideoCapture(0)
    

  width = 320
  height = 240
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)

  # # Initialize face detector
  #cascPath = "C:\Users\Navya\Desktop\major\haarcascade_frontalface_default.xml"
  faceCascade = cv2.CascadeClassifier(r"C:\Users\Navya\Desktop\major\haarcascade_frontalface_default.xml")

  sample_number = 1
  count = 0
  measures = np.zeros(sample_number, dtype=np.float)

  while True:
      ret, img_bgr = cap.read()
      if ret is False:
          print ("Error grabbing frame from camera")
          break
      colorHist, totNumColors = calColorHist(img_bgr)
      totNumColors /= 2000.0
    
    
      blu=blurriness(cv2.cvtColor(img_bgr,cv2.COLOR_RGB2GRAY))
      img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
      
      faces = detect_face(img_gray, faceCascade)
      
      measures[count%sample_number]=0
      
      point = (0,0)
      for i, (x, y, w, h) in enumerate(faces):
      
          roi = img_bgr[y:y+h, x:x+w]
          
          img_ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB)
          img_luv = cv2.cvtColor(roi, cv2.COLOR_BGR2LUV)
          
          ycrcb_hist = calc_hist(img_ycrcb)
          luv_hist = calc_hist(img_luv)
          
          feature_vector = np.append(ycrcb_hist.ravel(), luv_hist.ravel())
          feature_vector = feature_vector.reshape(1, len(feature_vector))
          
          prediction = clf.predict_proba(feature_vector)
          prob = prediction[0][1]
          
          measures[count % sample_number] = prob
          
          cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (255, 0, 0), 2)
          
          point = (x, y-5)
          
          print (measures, np.mean(measures))
          if 0 not in measures:
              text = "true"
              if np.mean(measures) >= 0.7:
                  text = "false"
                  count=count+1
                  font = cv2.FONT_HERSHEY_SIMPLEX
                  cv2.putText(img=img_bgr, text=text, org=point, fontFace=font, fontScale=0.9, color=(0, 0, 255),
                              thickness=2, lineType=cv2.LINE_AA)
              else:
                  font = cv2.FONT_HERSHEY_SIMPLEX
                  cv2.putText(img=img_bgr, text=text, org=point, fontFace=font, fontScale=0.9,
                              color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
          
      
      cv2.imshow('img_rgb', img_bgr)
      
      if cv2.waitKey(1) == ord('q'):
          break
  if count>=10:
      print("GENUINE")
  else:
      print("VIDEO ATTACK")
  cap.release()
  cv2.destroyAllWindows()


    
