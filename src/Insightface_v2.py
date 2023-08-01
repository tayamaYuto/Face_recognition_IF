import cv2
import numpy as np
from insightface.app import FaceAnalysis

def cos_sim(feat1, feat2):
    return np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))

def draw_on(img, faces, name):
    dimg = img.copy()
    for i in range(len(faces)):
        face = faces[i]
        box = face.bbox.astype(int)
        color = (0, 0, 255)
        cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
        if face.kps is not None:
            kps = face.kps.astype(int)
            #print(landmark.shape)
            for l in range(kps.shape[0]):
                color = (0, 0, 255)
                if l == 0 or l == 3:
                    color = (0, 255, 0)
                cv2.circle(dimg, (kps[l][0], kps[l][1]), 1, color, 2)
        cv2.putText(dimg, name, (box[0]-1, box[1]-4),cv2.FONT_HERSHEY_COMPLEX,1.5,(0,255,0),2)

    return dimg

app = FaceAnalysis()
app.prepare(ctx_id=1, det_size=(640, 640))

img_path = 'data/input/person1.JPG'

pre_img = cv2.imread(img_path)
pre_face = app.get(pre_img)
pre_embedding = [pre_face[0].embedding]

known_face_name = ["Unknown", "yuto"]

capture = cv2.VideoCapture(0)

while True:
    ret, flame = capture.read()

    if(ret == False):
        break

    faces = app.get(flame)
    embeddings = faces[0].embedding


    sim = cos_sim(pre_embedding, embeddings)
    print(sim)
    if sim >= 0.70:
        best_name_index = pre_embedding.index(pre_embedding[0]) + 1
    else:
        best_name_index = 0


    detect = draw_on(flame, faces, known_face_name[best_name_index])

    cv2.imshow("Face", detect)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break