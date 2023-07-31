import cv2
from insightface.app import FaceAnalysis

app = FaceAnalysis()
app.prepare(ctx_id=1, det_size=(640, 640))

capture = cv2.VideoCapture(0)
i = 0

while True:
    ret, flame = capture.read()

    if(ret == False):
        break

    faces = app.get(flame)
    detect = app.draw_on(flame, faces)

    if i != 0:
        cv2.imwrite(f'data/output/IFv1/Face_output_v1_{i}.jpg', detect)

    i += 1

    if i == 26:
        break

