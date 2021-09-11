import numpy
import face_recognition
import cv2
import os

def read_img(path):
	img = cv2.imread(path)
	(h,w) = img.shape[ : 2]
	width = 500
	ratio = width/float(w)
	height = int(h*ratio)
	return cv2.resize(img,(width,height))

video_capture = cv2.VideoCapture(0)
known_faces_encodings = []
known_face_names = []
known_dir = 'known'


for file in os.listdir(known_dir):
	img = read_img(known_dir + "/" + file)
	image_enc = face_recognition.face_encodings(img)[0]
	known_faces_encodings.append(image_enc)
	known_face_names.append(file.split('.')[0])


while True:
	ret, frame = video_capture.read()
	rgb_frame = frame[:,:, ::-1]
	face_location = face_recognition.face_locations(rgb_frame)
	face_encodings = face_recognition.face_encodings(rgb_frame,face_location)
	for (top, right, bottom, left), face_encodings in zip(face_location,face_encodings):
		matches = face_recognition.compare_faces(known_faces_encodings, face_encodings)
		name = "unknown"
		face_distance = face_recognition.face_distance(known_faces_encodings,face_encodings)
		best_match_index = numpy.argmin(face_distance)
		if matches[best_match_index]:
			name = known_face_names[best_match_index]
		
		cv2.rectangle(frame, (left,top),(right,bottom),(255,0,0),2)
		cv2.rectangle(frame, (left, bottom -35),(right,bottom),(255,0,0), cv2.FILLED)
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(frame,name,(left+6,bottom-6),font,1.0,(255,255,255),1)
		cv2.imshow('Abhilashs Classifier',frame)
		if cv2.waitKey(1):
			break

video_capture.release()
cv2.destoryAllWindows()
