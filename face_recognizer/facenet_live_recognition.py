import cv2
import sys
import time
from pathlib import Path
from threading import Event
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
from PIL import Image, ImageDraw, ImageFont

class FacenetRecognizer():

    def __init__(self, camera_id, threshold, database_path, log_callback=None, attendance_callback=None, frame_callback=None):
        self.frame_callback = frame_callback
        self._stop_event = Event()
        self.camera_id = camera_id
        self.threshold = 1 - (threshold/100)
        self.database_path = database_path
        if torch.cuda.is_available():
            self.device = "cuda"
        elif sys.platform == "darwin" and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        # MTCNN'nin değişken boyutlu işlemleri için Mac'te CPU kullanılır.
        self.detector_device = "cpu" if self.device == "mps" else self.device
        self.detection_width = 640
        self._resnet = None
        self._visual_tracks = []
        self._overlay_fonts = {}
        self.overlay_hold_seconds = 0.45
        self._face_transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
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

    @staticmethod
    def _box_iou(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        overlap = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        return overlap / max(area_a + area_b - overlap, 1)

    def _update_visual_tracks(self, observations, now):
        # Bu takip yalnızca çizim içindir; eski kutular yoklama sayacına girmez.
        old = [track for track in self._visual_tracks
               if now - track["seen"] <= self.overlay_hold_seconds]
        unused = set(range(len(old)))
        updated = []
        for box, identity, similarity in observations:
            match = max(unused, key=lambda i: self._box_iou(box, old[i]["box"]),
                        default=None)
            if match is not None and self._box_iou(box, old[match]["box"]) >= 0.25:
                previous = old[match]
                unused.remove(match)
                box = 0.65 * np.asarray(box) + 0.35 * previous["box"]
                if identity == previous["identity"]:
                    similarity = 0.65 * similarity + 0.35 * previous["similarity"]
            updated.append(dict(box=np.asarray(box, dtype=float), identity=identity,
                                similarity=similarity, seen=now, missed=0))
        for index in unused:
            track = dict(old[index])
            track["missed"] += 1
            if track["missed"] <= 2:
                updated.append(track)
        self._visual_tracks = updated
        return updated

    def _overlay_font(self, size):
        if size not in self._overlay_fonts:
            folder = Path(__file__).resolve().parent
            candidates = [folder / "fonts/DejaVuSans.ttf",
                          folder.parent / "fonts/DejaVuSans.ttf",
                          Path.cwd() / "fonts/DejaVuSans.ttf",
                          Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
                          Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/arial.ttf",
                          "DejaVuSans.ttf"]
            for candidate in candidates:
                try:
                    font = ImageFont.truetype(str(candidate), size)
                    break
                except OSError:
                    continue
            else:
                raise RuntimeError("Türkçe yazılar için fonts/DejaVuSans.ttf bulunamadı.")
            self._overlay_fonts[size] = font
        return self._overlay_fonts[size]

    def _draw_overlays(self, frame, tracks):
        if not tracks:
            return frame
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        font_size = max(18, round(frame.shape[1] / 60))
        font = self._overlay_font(font_size)
        for track in tracks:
            x1, y1, x2, y2 = (int(value) for value in track["box"])
            color = (0, 255, 0) if track["identity"] != "Bilinmiyor" else (255, 80, 80)
            draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
            text = f'{track["identity"]} ({track["similarity"]:.2f}%)'
            # Unicode TrueType fontu; OpenCV Hershey fontu kullanılmaz.
            draw.text((max(0, x1), max(0, y1 - font_size - 8)), text,
                      font=font, fill=color, stroke_width=1, stroke_fill=(0, 0, 0))
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    def _get_resnet(self):
        if self._resnet is None:
            model = InceptionResnetV1(pretrained="vggface2").eval()
            try:
                self._resnet = model.to(self.device)
            except (RuntimeError, NotImplementedError) as exc:
                if self.device != "mps":
                    raise
                self.log(f"MPS başlatılamadı; yüz tanıma CPU'ya geçti: {exc}")
                self.device = "cpu"
                self._resnet = model.to("cpu")
        return self._resnet

    def _embedding(self, face_tensor):
        model = self._get_resnet()
        with torch.inference_mode():
            try:
                output = model(face_tensor.to(self.device))
                # MPS sonuçlarının CPU'ya aktarımı da hata verebilir.
                return output.cpu().numpy().flatten()
            except (RuntimeError, NotImplementedError) as exc:
                if self.device != "mps":
                    raise
                self.log(f"MPS işlemi desteklenmedi; yüz tanıma CPU'ya geçti: {exc}")
                self.device = "cpu"
                self._resnet = model.to("cpu")
                return self._resnet(face_tensor.to("cpu")).cpu().numpy().flatten()

    def _detect_faces(self, detector, frame):
        # Orijinal kare değişmez; sadece algılayıcının girdisi küçültülür.
        height, width = frame.shape[:2]
        target_width = min(width, self.detection_width)
        target_height = max(1, round(height * target_width / width))
        if target_width < width:
            small = cv2.resize(frame, (target_width, target_height),
                               interpolation=cv2.INTER_AREA)
        else:
            small = frame
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        with torch.inference_mode():
            boxes, _ = detector.detect(rgb)
        if boxes is None:
            return None
        boxes = np.asarray(boxes, dtype=np.float64).copy()
        boxes[:, [0, 2]] *= width / target_width
        boxes[:, [1, 3]] *= height / target_height
        return boxes

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
        mtcnn = MTCNN(keep_all=True, device=self.detector_device)
        self._get_resnet()

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

                            if img is None:
                                self.log("Görüntü yüklenemedi: " + image_path)
                                continue

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
                                aligned_face = aligned_faces[0].unsqueeze(0)
                            else:
                                aligned_face = aligned_faces.unsqueeze(0)

                            embedding = self._embedding(aligned_face)

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
            self.status = "waiting"
            return  # Veritabanı boşsa çık

        if sys.platform == "darwin":
            if self.frame_callback is None:
                self.status = "waiting"
                raise RuntimeError("macOS için frame_callback gereklidir; RecognitionThread kullanın.")
            cap = cv2.VideoCapture(self.camera_id, cv2.CAP_AVFOUNDATION)
        else:
            cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            cap.release()
            self.status = "waiting"
            self.log("Kamera açılamadı!")
            return None  # veya bir hata durumu işlenebilir
        
        try:
            detector = MTCNN(keep_all=True, device=self.detector_device)  # MTCNN yüz algılama modelini başlat
            self._get_resnet()
            self.log(f"Yüz algılama: {self.detector_device}; yüz tanıma: {self.device}; algılama genişliği: {self.detection_width}px")

            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break

                # Her kare yalnızca kutular çizildikten sonra gönderilir.
                # Ham + işlenmiş kareyi sırayla göndermek kutuları titreştirir.
                faces = self._detect_faces(detector, frame)
                observations = []

                if faces is not None:
                    for face in faces:
                        if self._stop_event.is_set():
                            break
                        if not np.isfinite(face).all():
                            continue
                        x1, y1, x2, y2 = face

                        # Orijinal frame'den yüz crop'u al
                        height, width = frame.shape[:2]
                        x1 = int(max(0, min(x1, width)))
                        x2 = int(max(0, min(x2, width)))
                        y1 = int(max(0, min(y1, height)))
                        y2 = int(max(0, min(y2, height)))
                        if x2 <= x1 or y2 <= y1:
                            continue
                        face_crop = frame[y1:y2, x1:x2]

                        # ✅ Sadece yüz crop'u normalize edilir
                        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                        equalized = cv2.equalizeHist(gray)
                        face_crop_gray = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

                        face_pil = Image.fromarray(cv2.cvtColor(face_crop_gray, cv2.COLOR_BGR2RGB))

                        face_tensor = self._face_transform(face_pil).unsqueeze(0)
                        embedding = self._embedding(face_tensor)
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

                        observations.append(((x1, y1, x2, y2), identity, similarity))

                tracks = self._update_visual_tracks(observations, time.monotonic())
                frame = self._draw_overlays(frame, tracks)

                if sys.platform == "darwin":
                    self.frame_callback(frame)
                else:
                    # Windows: mevcut OpenCV penceresi ve Q davranışı korunur.
                    cv2.imshow('Face Recognition', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.log("Yüz tanıma işlemi durduruldu.")
                        break

        finally:
            cap.release()
            self.status = "waiting"
            if sys.platform != "darwin":
                cv2.destroyAllWindows()

    def stop(self):
        self._stop_event.set()
