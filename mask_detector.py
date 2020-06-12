import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import numpy as np
from mtcnn.mtcnn import MTCNN
detectface = MTCNN()
model = tf.keras.models.load_model('facemask.h5')
rec = cv2.VideoCapture(0)
while True:
    val,frame = rec.read()
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    result = detectface.detect_faces(frame)
    if result != []:
        for person in result:
            box = person['box']
            crop_img = frame[box[1]-30:box[1]+box[3]+30, box[0]-30:box[0]+box[2]+30]
            img = cv2.resize(crop_img, (160,160))
            array = tf.keras.preprocessing.image.img_to_array(img)
            x = np.expand_dims(array, axis=0)
            x=x/255
            pred=model.predict(x)
            if pred[0][0]>0.5:
                frame = cv2.rectangle(frame,(box[0]-10,box[1]-10),(box[0]+box[2]+10,box[1]+box[3]+10),(0,255,0),3)
                frame = cv2.putText(frame, 'Mask Found', (box[0]+20,box[1]+box[3]+20) , cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,0), 1)
            else:
                frame = cv2.rectangle(frame,(box[0]-10,box[1]-10),(box[0]+box[2]+10,box[1]+box[3]+10),(0,0,255),3)
                frame = cv2.putText(frame, 'Mask not Found', (box[0]+20,box[1]+box[3]+20) , cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,255), 1)
        cv2.imshow("video_cam",frame)
    else:
        cv2.imshow("video_cam",frame)       
rec.release()
cv2.destroyAllWindows()

