from PyQt5.QtCore import QThread, pyqtSignal
from facenet_live_recognition import FacenetRecognizer
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import pickle
import os

class RecognitionThread(QThread):
    log_signal = pyqtSignal(str)
    attendance_signal = pyqtSignal(str, str)
    finished_signal = pyqtSignal()

    def __init__(self, database_path, lesson_name, camera_id=0, accuracy_threshold=60):
        super().__init__()
        self.camera_id = camera_id
        self.accuracy_threshold = accuracy_threshold
        self.database_path = database_path
        self.lesson_name = lesson_name
        self.attendance_records = []
        self.known_students = set()
        now = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.pdf_output_path = os.path.join("Yoklama Raporları", lesson_name, f"YoklamaRaporu_{now}")
        self.pdf_name = f"{lesson_name}_yoklama_{now}.pdf"
        self._recognizer = None
        self._is_running = True

    def run(self):
        self._recognizer = FacenetRecognizer(
            self.camera_id,
            self.accuracy_threshold,
            self.database_path,
            log_callback=self.send_log,
            attendance_callback=self.send_attendance
        )
        self._recognizer.attendance_image_folder = self.pdf_output_path
        os.makedirs(self._recognizer.attendance_image_folder, exist_ok=True)
        self._recognizer.recognize_face_from_camera()
        self.known_students = self.get_students_from_database()
        self.generate_attendance_pdf()
        self.finished_signal.emit()

    def send_log(self, msg):
        self.log_signal.emit(msg)

    def is_running(self):
        return self._recognizer.status == "running"

    def send_attendance(self, student_id, timestamp):
        self.attendance_signal.emit(student_id, timestamp)
        self.attendance_records.append({"id": student_id, "timestamp": timestamp})
        self.known_students.add(student_id)

    def get_students_from_database(self):
        database_path = os.path.join(self.database_path, "face_database.pkl")
        if os.path.exists(database_path):
            with open(database_path, 'rb') as f:
                database = pickle.load(f)
                return set(database.keys())
        return set()

    def generate_attendance_pdf(self):
        folder_path = self.pdf_output_path
        pdfmetrics.registerFont(TTFont("DejaVuSans", "fonts/DejaVuSans.ttf"))
        os.makedirs(folder_path, exist_ok=True)

        c = canvas.Canvas(os.path.join(folder_path, self.pdf_name), pagesize=A4)
        c.setFont("DejaVuSans", 14)
        c.drawString(50, 800, f"Ders: {self.lesson_name}")
        c.drawString(300, 800, f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

        data = [["No", "Öğrenci ID", "Katılım Durumu"]]
        for i, student_id in enumerate(sorted(self.known_students), 1):
            status = "Katıldı" if any(r["id"] == student_id for r in self.attendance_records) else "Katılmadı"
            data.append([str(i), student_id, status])

        table = Table(data, colWidths=[50, 200, 150])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'DejaVuSans'),
            ('FONTNAME', (0, 1), (-1, -1), 'DejaVuSans'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))

        table.wrapOn(c, 50, 600)
        table.drawOn(c, 50, 750 - 20 * len(data))  # Konumu dinamik olarak ayarlar
        c.save()

    def stop(self):
        self._is_running = False
        if self._recognizer:
            self._recognizer.stop()
