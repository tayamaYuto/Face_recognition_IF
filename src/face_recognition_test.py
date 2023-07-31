import face_recognition
import cv2
import numpy as np




# 画像を読み込み、顔の特徴値を取得する

yuto_image = face_recognition.load_image_file("data/input/person1.jpg")
yuto_face_encoding = face_recognition.face_encodings(yuto_image)[0]

kajima_image = face_recognition.load_image_file("data/input/person2.png")
kajima_face_encoding = face_recognition.face_encodings(kajima_image)[0]

#uema_image = face_recognition.load_image_file("images/uema.jpg")
#uema_face_encoding = face_recognition.face_encodings(uema_image)[0]

known_face_encodings = [
    yuto_face_encoding,
    kajima_face_encoding,
    #uema_face_encoding,
]
known_face_names = [
    "yuto",
    "kajima",
    #"Uema",
]

video_capture = cv2.VideoCapture(0)

while True:
    # Webカメラの1フレームを取得、顔を検出し顔の特徴値を取得する
    _, frame = video_capture.read()
    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # 1フレームで検出した顔分ループする
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # 認識したい顔の特徴値と検出した顔の特徴値を比較する
        matches = face_recognition.compare_faces(
            known_face_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(
            known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # 顔の周りに四角を描画する
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # 名前ラベルを描画
        cv2.rectangle(frame, (left, bottom - 35),
                      (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255), 2)

    # 結果を表示する
    cv2.imshow('WebCam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break