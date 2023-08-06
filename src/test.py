import numpy as np
import cv2
from insightface.app import FaceAnalysis

image_file = "./data/input/test_image.jpg"
img = cv2.imread(image_file)
 
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640))

faces = app.get(np.array(img))

rimg = app.draw_on(img, faces)
cv2.imwrite('./data/output/test_output.jpg', rimg)