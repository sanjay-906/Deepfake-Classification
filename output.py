import tensorflow
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sklearn
from PIL import Image
import os
import shutil
from tensorflow.keras.utils import load_img, img_to_array
from skimage.transform import resize
import time
import fnmatch
import moviepy.editor
from os.path import isfile, join


image_frames= 'image_frames'
video_path= 'image_frames/video'
faces_path= 'image_frames/faces'
try:
    shutil.rmtree("image_frames")
except:
    pass

if not os.path.exists(image_frames):
    os.makedirs(image_frames)
    os.makedirs(video_path)
    os.makedirs(faces_path)

src_vid= cv2.VideoCapture("input.mp4")
fps = src_vid.get(cv2.CAP_PROP_FPS)

index= 0
while src_vid.isOpened():
    ret, frame= src_vid.read()
    if not ret:
        break

    name= video_path+ "/"+ str(index)+ ".jpeg"

    cv2.imwrite(name, frame)
    index= index+1
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

src_vid.release()
cv2.destroyAllWindows()


if not os.path.exists(faces_path):
    os.makedirs(faces_path)
files= [img for img in os.listdir(video_path)]
files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
for j,i in enumerate(files):
    img= Image.open(video_path +"/"+ i)
    img= np.array(img)
    img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    face_dt = face_cascade.detectMultiScale(img, 1.1, 10)
    for (x, y, w, h) in face_dt:
        try:
            cx = x+w//2
            cy = y+h//2
            cr  = max(w,h)//2
            r = cr+5*10
            face_dt = img[cy-r : cy+r, cx-r : cx+r]
            face_dt = resize(face_dt, (224, 224))
        except:
            face_dt = img[x:x+w , y:y+h]
            face_dt = resize(face_dt, (224, 224))
            pass
        face_dt= img_to_array(face_dt)
        face_dt= face_dt.reshape((1, face_dt.shape[0], face_dt.shape[1], face_dt.shape[2]))

        prediction = model.predict(face_dt, verbose=0)
        #print(prediction)
        if prediction[0][0]>0.985:
            text= 'Fake:{}'.format(prediction[0][0])
            img = cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
            (w1, h1), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            mask = img.copy()
            cv2.rectangle(mask, (x, y), (x+w1+135, y+h1+20), (0,0,255), -1)
            img = cv2.addWeighted(img, 1- 0.5, mask, 0.5, 0)
            cv2.putText(img, text, (x+5,y+27), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)

        else:
            text= 'Real:{}'.format(prediction[0][0])
            img = cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
            (w1, h1), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            mask = img.copy()
            cv2.rectangle(mask, (x, y), (x+w1+135, y+h1+20), (0,255,0), -1)
            img = cv2.addWeighted(img, 1- 0.5, mask, 0.5, 0)
            cv2.putText(img, text, (x+5,y+27), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)


        name= faces_path+ "/"+ str(j)+ ".jpeg"
        cv2.imwrite(name, img)
        #cv2.imshow(img)


    cv2.waitKey()


import cv2
import os

image_folder = 'image_frames/faces'
video_name = 'video.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".jpeg")]
num_of_files= len(images)
images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'MPEG'), fps, (width,height))

for i in range(num_of_files):
    video.write(cv2.imread(os.path.join(image_folder, '{}.jpeg'.format(i))))

cv2.destroyAllWindows()
video.release()


video_temp= moviepy.editor.VideoFileClip("input.mp4")
audio_temp=video_temp.audio
audio_temp.write_audiofile('org_audio.mp3')

video_temp=moviepy.editor.VideoFileClip("video.avi")
audio_temp=moviepy.editor.AudioFileClip('org_audio.mp3')
final_clip=video_temp.set_audio(audio_temp)

final_clip.write_videofile('final_output.mp4')
