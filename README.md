# Face Attendance

PyQt5 arayüzüyle kameradan yüz tanıma ve ders yoklaması alma uygulaması. Yüz algılama için MTCNN, embedding üretimi için VGGFace2 ağırlıklarıyla InceptionResnetV1 kullanır. Ayrı `face_detector` aracı, kameradan öğrenci fotoğrafları toplar.

Bu kullanım belgesi 20 Ekim 2026 tarihli macOS portu (`d00383e`) ve MPS/performans güncellemesini (`7a7c0e8`) kapsar.

## Kurulum

64 bit Python 3.10 veya 3.11 ile ayrı bir sanal ortam kullanın. Bu sürümler, aşağıdaki sabit bağımlılıklar için önerilen başlangıç noktasıdır; tüm platformlarda çalıştırma testi yapılmış bir destek matrisi değildir. Apple Silicon üzerinde ARM64 Python kullanın.

### macOS

```bash
git clone https://github.com/nicktimur/face-attendance.git
cd face-attendance
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

Python 3.10 kullanıyorsanız ilk komuttaki `python3.11` yerine `python3.10` yazın. Kamera erişimi istendiğinde izin verin; gerekirse Sistem Ayarları → Gizlilik ve Güvenlik → Kamera bölümünden uygulamayı başlattığınız Terminal veya IDE'nin iznini kontrol edin.

### Windows (PowerShell)

```powershell
git clone https://github.com/nicktimur/face-attendance.git
cd face-attendance
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

`requirements.txt` içindeki `PyQt5-Qt5==5.15.19` sabitlemesinin Windows wheel'i bulunmadığından, Windows için bu tek satırı çıkaran yerel bir liste oluşturun. Qt çalışma zamanını PyQt5 kendi platformuna uygun biçimde çözümlesin:

```powershell
Get-Content requirements.txt | Where-Object { $_ -notmatch '^PyQt5-Qt5==' } | Set-Content requirements.windows.txt
python -m pip install -r requirements.windows.txt
python -m pip check
```

Etkinleştirme engellenirse ortamı etkinleştirmeden `.\.venv\Scripts\python.exe` ile aşağıdaki `python` komutlarını çalıştırabilirsiniz. Windows kamera gizlilik ayarlarında masaüstü uygulamalarına erişim izni verin.

### CUDA veya CPU kurulumu (Windows)

Yukarıdaki bağımlılık kurulumundan sonra aşağıdaki seçeneklerden **birini** uygulayın. CUDA için uyumlu NVIDIA GPU ve sürücüsü gerekir; GPU'nuzun PyTorch 2.2.2 ile uyumluluğunu ayrıca kontrol edin.

```powershell
# NVIDIA CUDA 12.1
python -m pip install --force-reinstall --no-deps torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121

# Alternatif: yalnızca CPU
python -m pip install --force-reinstall --no-deps torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
```

Torch 2.2.2 ve torchvision 0.17.2 eşleşmesini koruyun. macOS için CUDA paketi kurmayın; standart macOS Torch paketiyle MPS kullanılabilirliği çalışma anında kontrol edilir. Kurulum komutlarının kaynağı: [PyTorch önceki sürümler, v2.2.2](https://pytorch.org/get-started/previous-versions/#v222). Qt paketleri: [PyQt5-Qt5 5.15.19 dosyaları](https://pypi.org/project/PyQt5-Qt5/5.15.19/#files).

## İşlem birimi nasıl seçilir?

Cihaz seçimi otomatiktir; GUI veya komut satırında cihaz seçme seçeneği yoktur.

| Koşul / öncelik | Yüz algılama (MTCNN) | Embedding (InceptionResnetV1) |
| --- | --- | --- |
| `torch.cuda.is_available()` doğru | CUDA | CUDA |
| CUDA yok, macOS ve MPS kullanılabilir | CPU | MPS |
| Diğer durumlar | CPU | CPU |

MPS modelinin yüklenmesi veya embedding işlemi `RuntimeError` / `NotImplementedError` üretirse uygulama log yazar ve embedding modelini CPU'ya geçirir. CUDA hataları için aynı otomatik geri dönüş uygulanmaz.

Ortamı doğrulamak için:

```bash
python -c "import sys, platform, torch; print('Python:', sys.version); print('Mimari:', platform.machine()); print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('MPS:', torch.backends.mps.is_available())"
```

Nihai cihazları uygulama logundaki `Yüz algılama: ...; yüz tanıma: ...` satırından kontrol edin. Canlı algılamada kare genişliği en fazla 640 piksele küçültülür; yüz kırpımları orijinal kareden alınır. Model bir kez oluşturulup yeniden kullanılır ve çıkarım `torch.inference_mode()` ile yapılır. Kutuların kısa süreli görsel takibi yoklama sayacını artırmaz. Sabit bir FPS garantisi yoktur.

## Veri hazırlığı ve çalıştırma

**Komutları depo kökünden çalıştırın.** Kökte `main.py` yoktur. PDF üretimi `fonts/DejaVuSans.ttf` yolunu çalışma dizinine göre açar; `face_recognizer` içine girip çalıştırmak font hatasına neden olabilir.

### 1. Öğrenci fotoğraflarını hazırlayın

Kendi fotoğraflarınızı şu yapıda düzenleyin; klasör adları öğrenci kimliği olarak kullanılır. Dataset depoyla birlikte gelmez.

```text
new_dataset/
├── 2345678901/
│   ├── photo1.jpg
│   └── photo2.jpg
└── 2345678902/
    ├── photo1.jpg
    └── photo2.jpg
```

Her fotoğrafta tek yüz bulunmalıdır. Okunamayan, yüz bulunamayan veya birden fazla yüz içeren fotoğraflar loglanıp atlanır. Öğrenci klasörlerine yalnızca görüntü dosyaları koyun.

Kameradan fotoğraf toplamak için isteğe bağlı aracı açın:

```bash
python face_detector/main.py
```

Öğrenci numarası, kamera ID'si ve fotoğraf adedini girin. Bu araç öğrenci numarasını sayısal ve **2000000000'dan büyük** bekler. Fotoğrafları çalışma dizinine göre `new_dataset/<öğrenci_no>/<öğrenci_no>_<sıra>.jpg` olarak kaydeder. Hedef adede ulaşınca veya OpenCV kamera penceresinde `q` basılınca durur. Bu araç Haar cascade kullanır; tanıyıcının macOS Qt önizleme portunu kullanmaz.

### 2. Yoklama uygulamasını açın

```bash
python face_recognizer/main.py
```

1. Öğrencilerin alt klasörlerini içeren **üst veri klasörünü** seçin (örneğin `new_dataset`).
2. Kamera ID'sini girin (boşsa `0`). Benzerlik eşiğini girin (boşsa `60`). Eşik bir doğruluk garantisi değildir; kabul koşulu cosine distance ≤ `1 - eşik/100` şeklindedir. Daha yüksek eşik kabulü zorlaştırır.
3. Ders adı girin veya önceki rapor klasörlerinden bir ders seçin. Ders adı klasör ve dosya adına girdiğinden dosya adı için uygun karakterler kullanın.
4. Yoklamayı başlatın. Seçili veri klasöründe dolu bir `face_database.pkl` varsa yüklenir; yoksa fotoğraflardan embedding veritabanı oluşturulup buraya kaydedilir. İlk model kullanımında VGGFace2 ağırlıkları indirilebileceğinden internet erişimi ve bekleme süresi gerekebilir.
5. Veritabanı oluştuktan sonra eşik önerme düğmesi kullanılabilir. Öneri için hem aynı kişiye ait fotoğraf çiftleri hem farklı kişilere ait embedding'ler gerekir. Yeni eşik sonraki başlatmada okunur.

**Fotoğraf veya öğrenci eklediğinizde:** Dolu önbellek otomatik güncellenmez. Yoklamayı durdurun, seçili veri klasöründeki `face_database.pkl` dosyasını yedekleyip kaldırın ve tekrar başlatın. Geçerli embedding'i olmayan öğrenciler veritabanına ve dolayısıyla PDF öğrenci listesine girmez.

### 3. Yoklamayı durdurun ve raporu alın

| Platform | Kamera görüntüsü | Normal durdurma |
| --- | --- | --- |
| macOS | AVFoundation kamera, ana GUI iş parçacığında Qt önizleme | `Q`, `Esc`, “Kamerayı durdur” veya kamera penceresini kapatma |
| Windows | OpenCV `Face Recognition` penceresi | Kamera penceresi odaktayken küçük `q` |

macOS'ta durdurma isteği devam eden yüz analizinden sonra tamamlanabilir. Ana uygulamayı kapatmadan işlemin bitmesini ve raporun oluşmasını bekleyin; zorla sonlandırma rapor üretimini tamamlamayabilir.

## Gerçek kayıt ve çıktı akışı

- Öğrenci, oturum boyunca **20 tanıma gözlemine** ulaştığında bir kez katıldı işaretlenir; yüz kırpımı rapor klasörüne `<öğrenci_id>.jpg` olarak yazılır. Sayaç kaçırılan karelerde sıfırlanmadığı için 20 ardışık kare şartı yoktur.
- Bilinmeyen yüzler için oturum genelinde ortak sayaç vardır. Her 20 bilinmeyen gözlemde seçili veri klasörünün `Tanınmayan Yüzler/` altına `unknown_YYYYMMDD_HHMMSS.jpg` kaydedilir. Bu sayaç kişi bazlı değildir; aynı saniyedeki aynı adlı kayıt üzerine yazılabilir.
- Tanıma döngüsü döndükten sonra `RecognitionThread` PDF üretir. Liste `face_database.pkl` anahtarlarından gelir; oturumda işaretlenenler `Katıldı`, diğerleri `Katılmadı` görünür. PDF tablosunda katılım durumu vardır; zaman damgaları ayrıca PDF sütunu olarak yazılmaz.
- Bu akış **CSV üretmez** ve `Yoklama Kayıtları/` klasörüne yazmaz. Loglar GUI'de gösterilir; ayrıca log dosyasına aktarım uygulanmamıştır.

Depo kökünden başlatıldığında örnek çıktı:

```text
face-attendance/
├── new_dataset/                           # Seçtiğiniz veri klasörü; adı/yeri değişebilir
│   ├── 2345678901/                        # Kaynak öğrenci fotoğrafları
│   ├── face_database.pkl                 # Oluşturulan embedding önbelleği
│   └── Tanınmayan Yüzler/
│       └── unknown_20261026_101530.jpg
└── Yoklama Raporları/                     # Çalışma dizinine göre
    └── Matematik/
        └── YoklamaRaporu_2026-10-26_10-15/
            ├── 2345678901.jpg             # Katıldı işaretlenince
            └── Matematik_yoklama_2026-10-26_10-15.pdf
```

Rapor dizini oturum oluşturulurken alınan **dakika** hassasiyetli zamanla adlandırılır. Aynı ders için aynı dakikada yeniden başlatma aynı dizini ve dosya adlarını kullanabilir. Yeni oturum açmadan önce önceki çıktıyı saklayın. Kamera açılamaması veya boş veritabanı gibi erken dönüşlerde de rapor üretimi denenebilir; PDF'nin oluşması tek başına başarılı yoklama alındığı anlamına gelmez. Hatalar için logları kontrol edin.

## Depodaki kaynak yapısı

```text
face-attendance/
├── face_detector/
│   ├── detecter.py
│   ├── detecter4images.py
│   ├── gui.ui
│   ├── main.py
│   └── UI.py
├── face_recognizer/
│   ├── facenet_live_recognition.py
│   ├── gui.ui
│   ├── kamera_test.py
│   ├── main.py
│   ├── recognition_thread.py
│   └── UI.py
├── fonts/
│   └── DejaVuSans.ttf
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

`new_dataset`, `face_database.pkl` ve raporlar çalışma sırasında hazırlanan/üretilen veridir. `detecter4images.py`, çalışma dizinindeki `test_faces/*.jpg` dosyalarını işleyen ayrı bir örnek betiktir; aynı klasöre yüz kırpımları yazar.

## Sorun giderme

- **Font bulunamadı / PDF yok:** Depo kökünden çalıştırın ve `fonts/DejaVuSans.ttf` dosyasının bulunduğunu kontrol edin.
- **Kamera açılamadı:** Kamera izinlerini, ID'yi ve kamerayı kullanan diğer uygulamaları kontrol edin.
- **MPS görünmüyor:** Python mimarisini ve yukarıdaki Torch kontrolünü inceleyin. MPS kullanılabilir değilse kod CPU seçer; MPS çalışma hatasında geri dönüş loglanır.
- **Yeni fotoğraflar kullanılmıyor:** Seçili veri klasöründeki embedding önbelleğini yukarıdaki adımlarla yeniden oluşturun.
- **Kalabalık sınıfta PDF taşması:** Mevcut rapor çizimi tek sayfalıdır; otomatik sayfalama yoktur. Büyük listelerde çıktıyı kontrol edin.

## Katkı ve lisans

Hata bildirimleri için issue, geliştirmeler için pull request açabilirsiniz. Proje [MIT Lisansı](LICENSE) ile lisanslanmıştır.
