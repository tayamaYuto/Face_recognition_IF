import cv2
from insightface.app import FaceAnalysis

app = FaceAnalysis()
app.prepare(ctx_id=1, det_size=(640, 640))

capture = cv2.VideoCapture(0)

while True:
    ret, flame = capture.read()

    if(ret == False):
        break

    faces = app.get(flame)
    detect = app.draw_on(flame, faces)

    cv2.imshow("flame", detect)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break