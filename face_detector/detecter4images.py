import cv2
import os
import glob

# Yüz sınıflandırıcısını yükleme
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_bounding_box(photo, filename):
    gray_image = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    
    for i, (x, y, w, h) in enumerate(faces):
        face = photo[y:y+h, x:x+w]
        save_face(face, filename, i)
        
    return faces

def save_face(face_image, original_filename, index):
    save_dir = "test_faces/"
    os.makedirs(save_dir, exist_ok=True)
    
    base_name = os.path.basename(original_filename)
    name, ext = os.path.splitext(base_name)
    new_filename = os.path.join(save_dir, f"{name}_detected_{index}{ext}")
    
    cv2.imwrite(new_filename, face_image)
    print(f"Yüz kaydedildi: {new_filename}")

def process_images(folder_path):
    image_files = glob.glob(os.path.join(folder_path, "*.jpg"))  # JPG formatındaki resimleri alır
    
    for image_file in image_files:
        photo = cv2.imread(image_file)
        if photo is not None:
            detect_bounding_box(photo, image_file)
        else:
            print(f"Hata: {image_file} yüklenemedi.")

# Kullanım örneği
process_images("test_faces/")  # 'images' klasöründeki tüm .jpg dosyalarını işler