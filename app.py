import cv
import math
import argparse
from cv2operator import KeyOperator, OperatorWindow, LineOperator

#Algorithm
def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]

for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn,(x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes

parser=argparse.ArgumentParser()
parser.add_argument('--image')

args=parser.parser_args()

faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uin8.pb"
ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"
genderProto="gender_deploy.protot"
genderModel="gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
agelist=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male', 'Female']

faceNet=cv2.dnn.readNet(faceModel, faceProto)

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel, ageProto)
genderNet=cv2.dnn.readNet(genderNet, genderProto)

video=cv2.VideoCapture(args.image if args.image else 0)
padding=20
while cv2.waitKey(1)>0:
    hasFrame,frame=video.read()
    if not hasFrame:
        cv2.waitKey()
        break

n = int(input("Masukan Nomor Anda : "))
if n == 5:
    print f("Angka Anda adalah" {n})
elif n >=5:
    print f("Angka yang Anda Masukan Melebihi limit")
elif n <=5:
    print f("Angka yang Anda Masukan Terlalu Kecil")
else:
