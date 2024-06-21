import cv2

cap = cv2.VideoCapture('face_video.mp4')
face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye = cv2.CascadeClassifier('haarcascade_eye.xml')

while True:
    success, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # переводим в черно-белый цвет

    faces = face.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        gray_face = gray[y:y + h, x:x + w]
        colored_face = img[y:y + h, x:x + w]

    eyes = eye.detectMultiScale(gray_face, scaleFactor=1.6, minNeighbors=7)

    for (ex, ey, ew, eh) in eyes:
        eyes_on_face = colored_face[ey:ey + eh, 0:colored_face.shape[0]]
        eyes_on_face = cv2.GaussianBlur(eyes_on_face, (73, 23), 51)
        eyes_on_face = cv2.cvtColor(eyes_on_face, cv2.COLOR_BGR2GRAY)
        eyes_on_face = cv2.cvtColor(eyes_on_face, cv2.COLOR_GRAY2BGR)
        colored_face[ey:ey + eh, 0:colored_face.shape[0]] = eyes_on_face

    cv2.imshow('Result', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
