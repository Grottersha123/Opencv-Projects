from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog,QMessageBox
from PyQt5 import QtWidgets
from PyQt5 import uic
from PyQt5.QtGui import QPixmap,QImage
import sys
import os
import numpy as n
import Simple_function as Simp
import configparser
form_class = uic.loadUiType("Mode2.ui")[0]  # Load the UI
form_login = uic.loadUiType("Login_Sap.ui")[0]
form_class2 = uic.loadUiType("Mode22.ui")[0]
class MyWin(QMainWindow,form_login):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.setupUi(self)
        # t1 = self.lineEdit_5
        # print('lol')
        # self.lineEdit_5.setText('lok')
        # print(t1.text(),'lol')
    def save_results(self):
         if self.lineEdit_6.text() == '' or self.lineEdit_5.text() == '' or self.lineEdit_7.text()== '' or self.lineEdit_8.text()== '' :
             msg = QMessageBox()
             msg.setIcon(QMessageBox.Information)
             msg.setText('Не все поля заполнены')
             msg.setWindowTitle("Оповещение")
             retval = msg.exec_()
         else:
             config = configparser.ConfigParser()
             config['DEFAULT'] = {'TOKEN': self.lineEdit_5.text(),
                      'MESSAGE_DEVICE': self.lineEdit_6.text(),
                     'NAME_ACCOUNT': self.lineEdit_7.text(),
                     'MESSAGE':self.lineEdit_8.text()}
             with open('{0}.ini'.format(self.lineEdit_7.text()), 'w') as configfile:
                 config.write(configfile)
             msg = QMessageBox()
             msg.setIcon(QMessageBox.Information)
             msg.setText('Данные сохранены выберете файл {0} и загрузите его'.format(self.lineEdit_7.text()))
             msg.setWindowTitle("Оповещение")
             retval = msg.exec_()
                 # print('Ok')




# class LoginWindowClass(QtWidgets.QDialog,form_login):
#      def __init__(self, parent=None):
#         QtWidgets.QDialog.__init__(self, parent)
#         self.setupUi(self)
#      def save_results(self):
#          t1 = self.lineEdit_5
#          t2 = self.lineEdit_6
#          t3 = self.lineEdit_7
#          t4 = self.lineEdit_8
#          print(t1,'lol')
#          if t1.text() == '' or t2.text() == '' or t3.text() == '' or t4.text() == '':
#              msg = QMessageBox()
#              msg.setIcon(QMessageBox.Information)
#              msg.setText('Не все поля заполнены')
#              msg.setWindowTitle("Оповещение")
#          else:
#              pass






class MyWindowClass(QMainWindow, form_class):
    path = os.getcwd()
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.setupUi(self)
        self.TOKEN = None
        self.MESSAGE_DEVICE= None
        self.NAME_ACCOUNT = None
        self.MESSAGE = None
        self.img_name = None
        self.flag = False
        self.window = None
        self.res = None


    def mybutton_clicked(self):
        options = QFileDialog.Options()
        fileName = QFileDialog.getOpenFileName(self, 'Open file', self.path,"Image files (*.jpg *.png *.bmp *.gif)")[0]
        if fileName:
            print(fileName)
            img_name = os.path.split(fileName)
            p = QPixmap(fileName).scaled(600,400)
            l = self.label
            l.setPixmap(p)
            try:
                res = Simp.Prepare_Img(fileName)
                l2 = self.label_2
                self.img_name = img_name[1]
                img2 = QPixmap('{0}\{1}'.format(r'.\CARS_OUTPUT',img_name[1]))
                print('{0}\{1}'.format(r'.\CARS_OUTPUT',img_name))
                l2.setPixmap(img2)
                if res is not None:
                    self.lineEdit.setText(res)
                    self.flag = True
                else:
                    self.lineEdit.setText('Не распознается')
                    self.flag = False
            except:
                self.lineEdit.setText('Не распознается')
                self.flag = False
    def sent_data(self):
        # print(self.lineEdit.text())
        res = self.lineEdit.text()


        if self.TOKEN is None:
            self.Show_message('Войдите в систему Сап')
        else:
            if self.flag == False:
                self.Show_message('Не отправлено')
            else:
                S = Simp.SAP(self.TOKEN,self.MESSAGE_DEVICE,self.NAME_ACCOUNT,self.MESSAGE)
                t = S.send_data_urllib(self.img_name,res)
                if t == 202:
                    self.Show_message("Данные отправлены на сервер")

                # self.lineEdit.setText('Отправлено')
                else:
                    self.Show_message("Ошибка, данные не отправлены")

    def Open_conf(self):
        fileName = QFileDialog.getOpenFileName(self, 'Open file', self.path,"Image files (*.ini)")[0]
        if fileName:
            print(fileName)
            self.read_config(fileName)
            self.Show_message('Данные загружены')
            print(self.TOKEN,self.MESSAGE_DEVICE,self.NAME_ACCOUNT,self.MESSAGE)
    def open_login(self):
        if self.window is None:
            self.window = MyWin(self)
        self.window.show()

        # pass
    def read_config(self,path):
        config = configparser.ConfigParser()
        config.read(path)
        self.TOKEN = config['DEFAULT']['TOKEN']
        self.MESSAGE_DEVICE= config['DEFAULT']['MESSAGE_DEVICE']
        self.NAME_ACCOUNT = config['DEFAULT']['NAME_ACCOUNT']
        self.MESSAGE = config['DEFAULT']['MESSAGE']
    def Show_message(self,text):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(text)
        msg.setWindowTitle("Оповещение")

        retval = msg.exec_()
    def video_real(self):
        Simp.Video()
    def open_video(self):
        fileName  = QFileDialog.getOpenFileName(self, 'Open file', r'D:\Github_project\OPENCV_Examples\CAR_NUMBER_RECOGNITION\CARSSSS_BLEYTTTTT',"Image files (*.mkv *.wmv *.avi *.mp4)")[0]

        if fileName:
            Simp.Video_file(fileName)


if __name__ == '__main__':

    app = QApplication(sys.argv)
    myWindow = MyWindowClass(None)
    myWindow.show()
    app.exec_()
