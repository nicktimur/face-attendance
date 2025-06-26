import os
import sys
from PyQt5 import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QTextCursor
from UI import *
from facenet_live_recognition import FacenetRecognizer
from recognition_thread import RecognitionThread
import pickle
import numpy as np
from scipy.spatial.distance import cosine


app = QApplication(sys.argv)
Frame = QtWidgets.QFrame()
ui = Ui_Frame()
ui.setupUi(Frame)
validator = QtGui.QDoubleValidator()
validator.setRange(0, 99, 0)
ui.camera_id.setValidator(validator)
validator.setRange(0, 999, 0)
ui.threshold.setValidator(validator)
Frame.show()


def cosine_distance(vec1, vec2):
    return cosine(vec1, vec2)

def suggest_threshold_from_database():
    database_path = ui.folderPathLineEdit.text()
    database_file = os.path.join(database_path, "face_database.pkl")

    if not os.path.exists(database_file):
        ui.logOutput.append("Veritabanı bulunamadı. Önce veritabanını oluşturun.")
        return

    with open(database_file, "rb") as f:
        database = pickle.load(f)

    intra_dists = []
    inter_dists = []

    for student_id, embeddings in database.items():
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                intra_dists.append(cosine_distance(embeddings[i], embeddings[j]))

    ids = list(database.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            for emb1 in database[ids[i]]:
                for emb2 in database[ids[j]]:
                    inter_dists.append(cosine_distance(emb1, emb2))

    if not intra_dists or not inter_dists:
        ui.logOutput.append("⚠️ Threshold hesaplamak için yeterli veri yok.")
        return

    avg_intra = np.mean(intra_dists)
    avg_inter = np.mean(inter_dists)
    suggested = (avg_intra + avg_inter) / 2
    suggested_percent = round((1 - suggested) * 100)

    ui.threshold.setText(str(suggested_percent))
    ui.logOutput.append(f"📊 Önerilen threshold: %{suggested_percent}")

def on_lesson_edited():
    # Kullanıcı elle bir şey yazdıysa combobox'ı resetle
    if ui.lessonComboBox.currentIndex() > 0:
        ui.lessonComboBox.setCurrentIndex(0)

def populate_existing_lessons():
    lesson_dir = "Yoklama Raporları"
    if not os.path.exists(lesson_dir):
        return

    lessons = [d for d in os.listdir(lesson_dir) if os.path.isdir(os.path.join(lesson_dir, d))]
    
    ui.lessonComboBox.clear()
    ui.lessonComboBox.addItem("📂 Mevcut derslerden seç")  # Placeholder
    ui.lessonComboBox.model().item(0).setEnabled(False)    # Seçilemez hale getir

    ui.lessonComboBox.addItems(lessons)

    def on_lesson_selected(index):
        if index > 0:
            selected = ui.lessonComboBox.itemText(index)
            ui.lesson_name.setText(selected)

    ui.lessonComboBox.currentIndexChanged.connect(on_lesson_selected)

populate_existing_lessons()

def Start_Recognizing():
    global recog_thread  # Burada global değişkeni tanımlıyoruz
    try:
        # Eğer recog_thread daha önce tanımlanmış ve aktifse, hata mesajı ver
        if 'recog_thread' in globals() and recog_thread.is_running():
            QMessageBox.about(Frame, "Hata", "<html><head/><body><p align=\"center\"><span style=\" color:#eaeaea;\">Bir tanıma işlemi zaten devam ediyor. Lütfen önce o işlemi tamamlayın.</span></p></body></html>")
            return

        camera_id = int(ui.camera_id.text()) if ui.camera_id.text() else 0
        accuracy_threshold = int(ui.threshold.text()) if ui.threshold.text() else 60
        database_path = ui.folderPathLineEdit.text()
        lesson_name = ui.lesson_name.text()
        
        if not os.path.exists(database_path):
            raise ValueError("Veri klasörü mevcut değil.")
        if lesson_name == "":
            raise ValueError("Ders adı boş olamaz.")

        # Yeni bir iş parçacığı başlat
        recog_thread = RecognitionThread(database_path, lesson_name, camera_id, accuracy_threshold)
        recog_thread.log_signal.connect(ui.logOutput.append)
        recog_thread.start()

    except ValueError:
        QMessageBox.about(Frame, "Hata", "<html><head/><body><p align=\"center\"><span style=\" color:#eaeaea;\">Lütfen tüm alanları doldurunuz ve veri klasörünü seçiniz.</span></p></body></html>")

ui.logOutput.append("İlk çalıştırmada veritabanı oluşturma işlemlerinden kaynaklı olarak bir süre donma olabilir.")
ui.logOutput.append("")
ui.logOutput.moveCursor(QtGui.QTextCursor.End)
ui.suggestThresholdButton.clicked.connect(suggest_threshold_from_database)
ui.lesson_name.textEdited.connect(on_lesson_edited)

ui.start_detect.clicked.connect(Start_Recognizing)
ui.selectFolderButton.clicked.connect(lambda: ui.folderPathLineEdit.setText(QFileDialog.getExistingDirectory(Frame, "Klasör Seç")))
sys.exit(app.exec_())

