import cv2
import numpy as np
from insightface.app import FaceAnalysis

def compute_sim(feat1, feat2):
    return np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))

def draw_on(img, faces, name):
    dimg = img.copy()
    for i in range(len(faces)):
        face = faces[i]
        box = face.bbox.astype(np.int_)
        color = (0, 0, 255)
        cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
        if face.kps is not None:
            kps = face.kps.astype(np.int_)
            #print(landmark.shape)
            for l in range(kps.shape[0]):
                color = (0, 0, 255)
                if l == 0 or l == 3:
                    color = (0, 255, 0)
                cv2.circle(dimg, (kps[l][0], kps[l][1]), 1, color, 2)
        cv2.putText(dimg, name, (box[0]-1, box[1]-4),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),1)

    return dimg

app = FaceAnalysis()
app.prepare(ctx_id=1, det_size=(640, 640))

img_path = 'data/input/person2_iphone.JPG'
img = cv2.imread(img_path)

face_k = app.get(img)
embedding = [face_k[0].embedding]

known_face_name = ["Unknown", "kajima"]

capture = cv2.VideoCapture(0)
i = 0

while True:
    ret, flame = capture.read()

    if(ret == False):
        break

    faces = app.get(flame)
    embedding2 = faces[0].embedding

    for embeddings in embedding:
        sim = compute_sim(embeddings, embedding2)
        print(sim)
        if sim >= 0.70:
            best_index = embedding.index(embeddings) + 1
        else:
            best_index = 0


    detect = draw_on(flame, faces, known_face_name[best_index])

    if i != 0:
        cv2.imwrite(f'data/output/IFv2/Face_output_v2_{i}.jpg', detect)

    i += 1

    if i == 26:
        break