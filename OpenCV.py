import cv2

cascade_classifier = cv2.CascadeClassifier("C:/Users/dilee/Downloads/haarcascade_eye.xml")
cap = cv2.VideoCapture(0)
                                           
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, 0)
    detections = cascade_classifier.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)
    if(len(detections) > 0):
        (x,y,w,h) = detections[0]
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h), (255,0,0),2)


    cv2.imshow('pythonfile',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
