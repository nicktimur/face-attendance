import cv2
import os
import glob
from PyQt5.QtWidgets import *



class Detecter():
    def __init__(self, student_no, camera_id, photo_amount, Frame) -> None: 
        self.Frame = Frame
        self.student_no = student_no
        self.camera_id = camera_id
        self.photo_amount = photo_amount
        self.face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.video_capture = cv2.VideoCapture(camera_id)
        self.frame_count = 0
        self.saved_image_count = 0
        self.max = 0
        self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        if self.fps < 10:
            self.fps = 15
        self.output_folder = "new_dataset/" + student_no
        self.title = "Face Detection - 0 Yüz Kaydedildi"
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        else:
            dosyalar = glob.glob(f"{self.output_folder}/*")
            dosya_isimleri = [os.path.basename(dosya) for dosya in dosyalar]
            print(dosya_isimleri)
            for dosya in dosya_isimleri:
                if int(dosya.split("_")[1].split(".")[0]) > self.max:
                    self.max = int(dosya.split("_")[1].split(".")[0])

        self.detect_faces()

    def detect_bounding_box(self, vid):
        gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
        faces = self.face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
        for (x, y, w, h) in faces:
            face_crop = vid[y:y+h, x:x+w]
            if self.frame_count >= self.fps:
                self.save_face(face_crop) # Sadece yüzü kaydeder
                self.frame_count = 0
            cv2.rectangle(vid, (x, y), (x+w, y+h), (0, 255, 0), 4)
        return faces

    def save_face(self, face_image):
        filename = os.path.join(self.output_folder, f"{self.student_no}_{self.max + self.saved_image_count + 1}.jpg")
        cv2.imwrite(filename, face_image)
        self.saved_image_count += 1
        print(f"{self.saved_image_count}. yüz kaydedildi.")

    
    def detect_faces(self):
        while self.saved_image_count < self.photo_amount:

            result, video_frame = self.video_capture.read()
            if result is False:
                break

            faces = self.detect_bounding_box(video_frame)
            new_title = f"Face Detection - {self.saved_image_count} Yuz Kaydedildi"
            cv2.imshow(self.title, video_frame)
            cv2.setWindowTitle(self.title, new_title)

            self.frame_count += 1


            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.video_capture.release()
        cv2.destroyAllWindows()
