import cv2
import numpy as np
import os
import torch
import pickle
from datetime import datetime
from PyQt5.QtWidgets import QMessageBox
from PyQt5 import QtGui
from facenet_pytorch import InceptionResnetV1, MTCNN
from scipy.spatial.distance import cosine
from torchvision import transforms
from PIL import Image

class FacenetRecognizer():

    def __init__(self, camera_id, threshold, database_path, log_callback=None, attendance_callback=None):
        self.camera_id = camera_id
        self.threshold = 1 - (threshold/100)
        self.database_path = database_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.status = "waiting"
        self.log_callback = log_callback
        self.attendance_callback = attendance_callback
        self.attendance_log = {}  # {"student_id": {"count": int, "marked": bool, "timestamp": str}}
        self.detection_threshold = 20  # kaç kez arka arkaya tanınırsa yoklamaya katılır
        self.database = self.create_face_database(self.database_path)
        self.attendance_image_folder = None
        self.unknown_face_count = 0
        self.unknown_save_threshold = 20  # kaç kez arka arkaya bilinmiyor olursa bilinmeyen yüz kaydedilir
        self.unknown_faces_dir = os.path.join(self.database_path, "Tanınmayan Yüzler")
        os.makedirs(self.unknown_faces_dir, exist_ok=True)

    def log(self, message):
        if self.log_callback:
            self.log_callback(message)

    # Cosine Distance hesaplayan fonksiyon
    def cosine_distance(self, vec1, vec2):
        return cosine(vec1, vec2)

    # Cosine Distance'ı Yüzde Benzerlik olarak hesaplayan fonksiyon
    def similarity_percentage(self, cosine_distance_value):
        return (1 - cosine_distance_value) * 100

    # Veritabanını kaydetme
    def save_face_database(self, database, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(database, f)
            print(f"Veritabanı kaydedildi: {filepath}")
            self.log("Veritabanı kaydedildi: " + filepath)

    # Veritabanını yükleme
    def load_face_database(self, filepath):
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        return {}
    

    # Veritabanını oluştur ve her öğrencinin embedding'ini al
    def create_face_database(self, database_path):
        self.status = "running"
        save_path = os.path.join(database_path, "face_database.pkl")
        database = self.load_face_database(save_path)
        
        if len(database) > 0:
            print("Veritabanı yüklendi.")
            self.log("Mevcut veritabanı yüklendi")
            self.status = "waiting"
            return database

        print("Veritabanı oluşturuluyor...")
        self.log("Veritabanı oluşturuluyor... Bu işlem zaman alabilir.")
        mtcnn = MTCNN(keep_all=True, device=self.device)
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        try:
            for student_id in os.listdir(database_path):
                if student_id == "Tanınmayan Yüzler":
                    continue  # Bu klasörü atla
                student_dir = os.path.join(database_path, student_id)
                if os.path.isdir(student_dir):
                    student_embeddings = []  # Her öğrencinin tüm fotoğraflarının embedding'lerini tutacak

                    # Öğrencinin tüm fotoğraflarını kullanıyoruz
                    for image_file in os.listdir(student_dir):
                        image_path = os.path.join(student_dir, image_file)
                        self.log("Görüntü işleniyor: " + image_file)

                        # Yüzü algıla ve embedding oluştur
                        with open(image_path, 'rb') as f:
                            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
                            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                            # Ekle: Griye çevir, histogram eşitleme ve tekrar BGR
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            equalized = cv2.equalizeHist(gray)
                            img = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

                        if img is None:
                            print("❌ Görüntü yüklenemedi:", image_path)
                            self.log("Görüntü yüklenemedi: " + image_path)
                            
                        aligned_faces = mtcnn(img)

                        if aligned_faces is not None:
                            if isinstance(aligned_faces, torch.Tensor) and len(aligned_faces.shape) == 4:
                                # Çoklu yüz algılanmışsa hata kabul et
                                if aligned_faces.shape[0] > 1:
                                    self.log(f"HATALI: {image_file} birden fazla yüz içeriyor, atlanıyor.")
                                    continue  # Bu görüntüyü işleme
                                aligned_face = aligned_faces[0].unsqueeze(0).to(self.device)
                            else:
                                aligned_face = aligned_faces.unsqueeze(0).to(self.device)

                            embedding_tensor = resnet(aligned_face)
                            embedding = embedding_tensor.detach().cpu().numpy().flatten()

                            norm = np.linalg.norm(embedding)
                            if norm != 0:
                                embedding = embedding / norm

                            student_embeddings.append(embedding)
                        else:
                            self.log(f"Uyarı: {image_file} için yüz algılanamadı.")


                    # Öğrencinin tüm fotoğraflarının embedding'lerini kaydet
                    if student_embeddings:
                        database[student_id] = student_embeddings
        except Exception as e:
            print(f"Veritabanı oluşturulurken hata: {e}")
            self.log("Veritabanı oluşturulurken hata: " + str(e))
            return None

        # Veritabanını kaydet
        self.save_face_database(database, save_path)
        self.status = "waiting"
        return database

    # Kameradan yüz tanıma
    def recognize_face_from_camera(self):
        self.status = "running"
        self.log("İşlem birimi olarak " + self.device + " kullanılıyor.")
        if not self.database:
            return  # Veritabanı boşsa çık

        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            self.log("Kamera açılamadı!")
            return None  # veya bir hata durumu işlenebilir
        
        detector = MTCNN(keep_all=True, device=self.device)  # MTCNN yüz algılama modelini başlat
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # MTCNN ile yüz koordinatlarını ve yüz görüntülerinin tensörlerini al
            faces, _ = detector.detect(frame)
            face_tensors = detector(frame)

            if faces is not None and face_tensors is not None:
                for i, face in enumerate(faces):
                    x1, y1, x2, y2 = face

                    if i < len(face_tensors) and face_tensors[i] is not None:
                        # Orijinal frame'den yüz crop'u al
                        height, width = frame.shape[:2]
                        x1 = int(max(0, min(x1, width)))
                        x2 = int(max(0, min(x2, width)))
                        y1 = int(max(0, min(y1, height)))
                        y2 = int(max(0, min(y2, height)))
                        face_crop = frame[y1:y2, x1:x2]

                        # ✅ Sadece yüz crop'u normalize edilir
                        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                        equalized = cv2.equalizeHist(gray)
                        face_crop_gray = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

                        face_pil = Image.fromarray(cv2.cvtColor(face_crop_gray, cv2.COLOR_BGR2RGB))

                        # 3. Tensor ve Resize
                        transform = transforms.Compose([
                            transforms.Resize((160, 160)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5], [0.5])
                        ])

                        face_tensor = transform(face_pil).unsqueeze(0).to(self.device)

                        # Embedding çıkar
                        embedding = resnet(face_tensor).detach().cpu().numpy().flatten()
                        norm = np.linalg.norm(embedding)
                        if norm != 0:
                            embedding = embedding / norm

                        # En yakın kişiyi bul
                        min_distance = float('inf')
                        identity = "Bilinmiyor"

                        for student_id, student_embeddings in self.database.items():
                            for db_embedding in student_embeddings:
                                dist = self.cosine_distance(embedding, db_embedding)
                                if dist < min_distance:
                                    min_distance = dist
                                    identity = student_id

                        if min_distance > self.threshold:
                            identity = "Bilinmiyor"

                        if identity != "Bilinmiyor":
                            info = self.attendance_log.get(identity, {"count": 0, "marked": False, "timestamp": None})

                            if not info["marked"]:
                                info["count"] += 1
                                if info["count"] >= self.detection_threshold:
                                    info["marked"] = True
                                    info["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    self.log(f"Yoklamaya katıldı: {identity} ({info['timestamp']})")
                                    if self.attendance_callback:
                                        self.attendance_callback(identity, info["timestamp"])

                                    # ✅ Yüz görselini kaydet
                                    if self.attendance_image_folder:

                                        if x2 > x1 and y2 > y1:
                                            # Türkçe karakter içeren dosya adı güvenli hale getirilir
                                            img_filename = f"{identity}.jpg"
                                            save_path = os.path.join(self.attendance_image_folder, img_filename)

                                            try:
                                                # Türkçe karakterli yola güvenli kayıt
                                                _, buffer = cv2.imencode(".jpg", face_crop)
                                                with open(save_path, 'wb') as f:
                                                    f.write(buffer)
                                            except Exception as e:
                                                self.log(f"❌ Hata oluştu, kayıt yapılamadı: {save_path}\n{e}")
                                        else:
                                            self.log(f"⚠️ Geçersiz yüz koordinatları: ({x1}, {y1}) - ({x2}, {y2})")

                            else:
                                info["count"] = 0  # zaten işaretli, tekrar saymaya gerek yok

                            self.attendance_log[identity] = info

                        else:
                            self.unknown_face_count += 1
                            if self.unknown_face_count >= self.unknown_save_threshold:
                                self.unknown_face_count = 0  # Sayacı sıfırla
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                filename = f"unknown_{timestamp}.jpg"
                                save_path = os.path.join(self.unknown_faces_dir, filename)
                                try:
                                    _, buffer = cv2.imencode(".jpg", face_crop)
                                    with open(save_path, 'wb') as f:
                                        f.write(buffer)
                                except Exception as e:
                                    self.log(f"❌ Bilinmeyen yüz kaydedilemedi: {e}")

                        # Benzerlik yüzdesi hesapla
                        similarity = self.similarity_percentage(min_distance)

                        # Yüzü kare içine al
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                        # Tahmini yaz ve benzerlik yüzdesini göster
                        cv2.putText(frame, f'{identity} ({similarity:.2f}%)', (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if identity != "Bilinmiyor" else (0, 0, 255), 2)

            # Görüntüyü ekranda göster
            cv2.imshow('Face Recognition', frame)

            # 'q' tuşuna basarak çıkış yap
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.log("Yüz tanıma işlemi durduruldu.")
                self.status = "waiting"
                break

        cap.release()
        cv2.destroyAllWindows()