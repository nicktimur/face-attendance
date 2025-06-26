# Face Recognizer

Bu proje, yüz tanıma ve yoklama alma işlemlerini gerçekleştiren bir uygulamadır. Uygulama, PyQt5 tabanlı bir grafik kullanıcı arayüzü (GUI) ile kullanıcı dostu bir deneyim sunar. Yüz tanıma işlemleri için **MTCNN** ve **InceptionResnetV1** modelleri kullanılmıştır.

## Özellikler

- **Yüz Tanıma**: Kamera üzerinden gerçek zamanlı yüz tanıma.
- **Yoklama Sistemi**: Tanınan yüzlere göre yoklama kaydı oluşturma.
- **Veritabanı Yönetimi**: Öğrenci fotoğraflarından yüz embedding'leri oluşturarak veritabanı oluşturma.
- **Kullanıcı Dostu Arayüz**: PyQt5 tabanlı GUI ile kolay kullanım.
- **Loglama**: İşlem sırasında detaylı loglama ve hata bildirimleri.

## Kurulum

1. **Gerekli Kütüphaneleri Yükleyin**:
   Proje için gerekli Python kütüphanelerini yüklemek için aşağıdaki komutu çalıştırın:

   ```bash
   pip install -r requirements.txt
   ```

2. **Veritabanı Hazırlığı**:
   - Öğrenci fotoğraflarını bir klasöre yerleştirin. Her öğrenci için ayrı bir klasör oluşturun ve klasör adını öğrenci numarası olarak belirleyin.
   - Örnek klasör yapısı:
     ```
     new_dataset/
     ├── 12345/
     │   ├── photo1.jpg
     │   ├── photo2.jpg
     ├── 67890/
         ├── photo1.jpg
         ├── photo2.jpg
     ```

3. **Uygulamayı Başlatın**:
   Aşağıdaki komutla uygulamayı çalıştırabilirsiniz:
   ```bash
   python main.py
   ```

## Kullanım

1. **Kamera ve Veritabanı Ayarları**:
   - Kamera ID'sini ve eşik değerini (threshold) GUI üzerinden girin.
   - Öğrenci fotoğraf klasörünü seçin.

2. **Yüz Tanıma ve Yoklama**:
   - "Yoklama Almaya Başla" butonuna tıklayarak yüz tanıma işlemini başlatın.
   - Tanınan yüzlere göre yoklama kaydı oluşturulur ve CSV dosyasına kaydedilir.

3. **Log ve Kayıtlar**:
   - Loglar GUI üzerinde görüntülenir.
   - Yoklama kayıtları `Yoklama Kayıtları/` klasörüne kaydedilir.

## Proje Yapısı
```plaintext
face-recognizer/
├── face_detector/
│   ├── detecter.py
│   ├── detecter4images.py
│   ├── gui.ui
│   ├── main.py
│   ├── UI.py
├── face_recognizer/
│   ├── facenet_live_recognition.py
│   ├── gui.ui
│   ├── main.py
│   ├── recognition_thread.py
│   ├── UI.py
│   ├── dataset/
│   │   ├── 12345/
│   │   │   ├── photo1.jpg
│   │   │   ├── photo2.jpg
│   │   ├── 67890/
│   │       ├── photo1.jpg
│   │       ├── photo2.jpg
├── README.md
├── requirements.txt
```

## Katkıda Bulunma

Bu projeye katkıda bulunmak isterseniz, lütfen bir **pull request** gönderin veya bir **issue** açın. Her türlü geri bildirim ve öneri memnuniyetle karşılanır.

## Lisans

Bu proje MIT Lisansı ile lisanslanmıştır. Daha fazla bilgi için [LICENSE](LICENSE) dosyasına göz atabilirsiniz.
