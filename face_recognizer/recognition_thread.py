from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer, QEvent
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QPushButton
from threading import Lock
import sys
import cv2
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

class MacCameraWindow(QWidget):
    """Ana pencereye bağlı kamera penceresi; Q uygulama genelinde yakalanır."""

    def __init__(self, on_close, parent=None):
        super().__init__(parent, Qt.Tool | Qt.WindowStaysOnTopHint)
        self._on_close = on_close
        self._closing = False
        self._listening = False
        self.setWindowTitle("Face Recognition — Q / Esc ile durdur")
        self.resize(960, 720)
        self.setMinimumSize(320, 240)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(1, 1)
        self.image_label.setStyleSheet("background: black;")
        self.stop_button = QPushButton("Kamerayı durdur (Q / Esc)")
        self.stop_button.clicked.connect(self.close)
        layout = QVBoxLayout(self)
        layout.addWidget(self.image_label, 1)
        layout.addWidget(self.stop_button)

    def start_listening(self):
        if not self._listening:
            QApplication.instance().installEventFilter(self)
            self._listening = True

    def stop_listening(self):
        if self._listening:
            QApplication.instance().removeEventFilter(self)
            self._listening = False

    def eventFilter(self, obj, event):
        if event.type() in (QEvent.ShortcutOverride, QEvent.KeyPress):
            modifiers = event.modifiers()
            if (event.key() in (Qt.Key_Q, Qt.Key_Escape)
                    and not modifiers & (Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier)):
                event.accept()
                if event.type() == QEvent.KeyPress:
                    self.close()
                return True
        return super().eventFilter(obj, event)

    def closeEvent(self, event):
        if not self._closing:
            self._closing = True
            self.stop_listening()
            self._on_close()
        event.accept()


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
        self._frame_lock = Lock()
        self._latest_frame = None
        self._camera_window = None
        self._preview_timer = None

        if sys.platform == "darwin":
            app = QApplication.instance()
            if app is None or QThread.currentThread() != app.thread():
                raise RuntimeError("RecognitionThread ana GUI iş parçacığında oluşturulmalıdır.")
            parent_window = app.activeWindow()
            if parent_window is None:
                parent_window = next((
                    widget for widget in app.topLevelWidgets()
                    if widget.isVisible() and widget.windowType() == Qt.Window
                ), None)
            self._camera_window = MacCameraWindow(self._stop_preview, parent_window)
            self._preview_timer = QTimer(self)
            self._preview_timer.setInterval(33)
            self._preview_timer.timeout.connect(self._show_latest_frame)
            self.started.connect(self._start_preview)
            self.finished.connect(self._close_preview)

    def run(self):
        try:
            kwargs = {}
            if sys.platform == "darwin":
                kwargs["frame_callback"] = self._receive_frame
            self._recognizer = FacenetRecognizer(
                self.camera_id,
                self.accuracy_threshold,
                self.database_path,
                log_callback=self.send_log,
                attendance_callback=self.send_attendance,
                **kwargs
            )
            if self.isInterruptionRequested():
                return
            self._recognizer.attendance_image_folder = self.pdf_output_path
            os.makedirs(self._recognizer.attendance_image_folder, exist_ok=True)
            self._recognizer.recognize_face_from_camera()
            self.known_students = self.get_students_from_database()
            self.generate_attendance_pdf()
        except Exception as exc:
            self.send_log(f"Yüz tanıma hatası: {exc}")
        finally:
            self._is_running = False
            self.finished_signal.emit()

    def _receive_frame(self, frame):
        # Tek kare saklanır; GUI yavaşlarsa sinyal kuyruğu birikmez.
        with self._frame_lock:
            self._latest_frame = frame.copy()

    def _show_latest_frame(self):
        if not self._is_running or self.isInterruptionRequested():
            return
        with self._frame_lock:
            frame = self._latest_frame
            self._latest_frame = None
        if frame is None:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = rgb.shape
        image = QImage(
            rgb.data, width, height, rgb.strides[0], QImage.Format_RGB888
        ).copy()
        self._camera_window.image_label.setPixmap(QPixmap.fromImage(image).scaled(
            self._camera_window.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        if not self._camera_window.isVisible():
            self._camera_window.show()
            self._camera_window.raise_()
            self._camera_window.activateWindow()
            self._camera_window.setFocus(Qt.OtherFocusReason)

    def _start_preview(self):
        if not self.isInterruptionRequested():
            self._camera_window.start_listening()
            self._preview_timer.start()

    def _stop_preview(self):
        # GUI hemen kapanır; worker devam eden yüz analizinden sonra çıkar.
        self._preview_timer.stop()
        self._camera_window.hide()
        self.stop()

    def _close_preview(self):
        self._preview_timer.stop()
        self._camera_window.stop_listening()
        self._camera_window.hide()
        with self._frame_lock:
            self._latest_frame = None

    def send_log(self, msg):
        self.log_signal.emit(msg)

    def is_running(self):
        return self.isRunning()

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
        self.requestInterruption()
        if self._recognizer:
            self._recognizer.stop()
