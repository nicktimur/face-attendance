import os
import sys
from PyQt5 import *
from PyQt5.QtWidgets import *
from UI import *
from detecter import *


app = QApplication(sys.argv)
Frame = QtWidgets.QFrame()
ui = Ui_Frame()
ui.setupUi(Frame)
validator = QtGui.QDoubleValidator()
validator.setRange(0, 29999999999, 0)
ui.student_number.setValidator(validator)
validator.setRange(0, 99, 0)
ui.camera_id.setValidator(validator)
ui.photo_amount.setValidator(validator)
Frame.show()


def Start_Detection():
    try:
        student_number = ui.student_number.text()
        camera_id = int(ui.camera_id.text())
        needed_photo_amount = int(ui.photo_amount.text())
        if(int(student_number) > 2000000000):
            dt = Detecter(student_number, camera_id, needed_photo_amount, Frame)
        else:
            QMessageBox.about(Frame, "Hata", "<html><head/><body><p align=\"center\"><span style=\" color:#eaeaea;\"></br><br></br> Lütfen geçerli bir öğrenci numarası giriniz </body></html>")
    except ValueError:
       QMessageBox.about(Frame, "Hata", "<html><head/><body><p align=\"center\"><span style=\" color:#eaeaea;\"></br><br></br> Lütfen tüm alanları doldurunuz </body></html>")



ui.start_detect.clicked.connect(Start_Detection)

ui.student_number.text()


sys.exit(app.exec_())