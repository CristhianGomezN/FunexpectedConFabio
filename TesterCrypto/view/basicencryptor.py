# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'simpleencryptor.ui'
#
# Created: Wed Nov 15 00:20:41 2017
#      by: pyside-uic 0.2.15 running on PySide 1.2.4
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui
from modules.util import Util

class Ui_MainWindow(object):
    def __init__(self, app):
        self.app = app
        
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(939, 494)
        MainWindow.setAcceptDrops(False)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        self.decrypt = QtGui.QPushButton(self.centralwidget)
        self.decrypt.setGeometry(QtCore.QRect(260, 300, 85, 27))
        self.decrypt.setObjectName("decrypt")
        self.decrypt.clicked.connect(lambda: self.decryptClicked())
        
        self.encrypt = QtGui.QPushButton(self.centralwidget)
        self.encrypt.setGeometry(QtCore.QRect(130, 300, 85, 27))
        self.encrypt.setObjectName("encrypt")
        self.encrypt.clicked.connect(lambda: self.encryptClicked())
        
        self.textInput = QtGui.QTextEdit(self.centralwidget)
        self.textInput.setGeometry(QtCore.QRect(30, 200, 421, 75))
        self.textInput.setObjectName("textInput")
        
        self.textOutput = QtGui.QTextEdit(self.centralwidget)
        self.textOutput.setGeometry(QtCore.QRect(30, 360, 421, 75))
        self.textOutput.setObjectName("textOutput")
        
        self.titleWidget = QtGui.QLabel(self.centralwidget)
        self.titleWidget.setGeometry(QtCore.QRect(300, 20, 321, 41))
        font = QtGui.QFont()
        font.setFamily("Verdana")
        font.setPointSize(16)
        font.setWeight(75)
        font.setItalic(True)
        font.setStrikeOut(False)
        font.setBold(True)
        self.titleWidget.setFont(font)
        self.titleWidget.setAlignment(QtCore.Qt.AlignCenter)
        self.titleWidget.setIndent(0)
        self.titleWidget.setObjectName("titleWidget")
        
        self.buttonGroupInput = QtGui.QGroupBox(self.centralwidget)
        self.buttonGroupInput.setGeometry(QtCore.QRect(30, 140, 411, 51))
        self.buttonGroupInput.setObjectName("buttonGroupInput")
        
        self.radioButtonText = QtGui.QRadioButton(self.buttonGroupInput)
        self.radioButtonText.setGeometry(QtCore.QRect(260, 30, 100, 22))
        self.radioButtonText.setObjectName("radioButtonText")
        self.radioButtonText.toggled.connect(self.buttonTextChecked)
        
        self.radioButtonBinary = QtGui.QRadioButton(self.buttonGroupInput)
        self.radioButtonBinary.setGeometry(QtCore.QRect(30, 30, 121, 22))
        self.radioButtonBinary.setChecked(True)
        self.radioButtonBinary.setObjectName("radioButtonBinary")
        self.radioButtonBinary.toggled.connect(self.buttonBinaryChecked)
        self.binary = True 
        
        self.automatons = QtGui.QLabel(self.centralwidget)
        self.automatons.setGeometry(QtCore.QRect(30, 90, 131, 21))
        self.automatons.setObjectName("automatons")

        self.numberAutomatons = QtGui.QLabel(self.centralwidget)
        self.numberAutomatons.setGeometry(QtCore.QRect(180, 90, 91, 21))
        self.numberAutomatons.setObjectName("numberAutomatons")
        self.changedNumberAutomatons(self.app.automatons)
        
        self.textOutputPublicKey = QtGui.QTextEdit(self.centralwidget)
        self.textOutputPublicKey.setGeometry(QtCore.QRect(490, 310, 421, 131))
        self.textOutputPublicKey.setObjectName("textOutputPrivateKey")
        
        self.privateKey = QtGui.QLabel(self.centralwidget)
        self.privateKey.setGeometry(QtCore.QRect(490, 110, 131, 21))
        self.privateKey.setObjectName("privateKey")
        
        self.publicKey = QtGui.QLabel(self.centralwidget)
        self.publicKey.setGeometry(QtCore.QRect(490, 280, 131, 21))
        self.publicKey.setObjectName("publicKey")
        
        self.textOutputPrivateKey = QtGui.QTextEdit(self.centralwidget)
        self.textOutputPrivateKey.setGeometry(QtCore.QRect(490, 140, 421, 131))
        self.textOutputPrivateKey.setObjectName("textOutputPublicKey")
        
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 939, 25))
        self.menubar.setObjectName("menubar")
        
        self.menuOptions = QtGui.QMenu(self.menubar)
        self.menuOptions.setObjectName("menuOptions")
        
        self.menuProtocolConfig = QtGui.QMenu(self.menuOptions)
        self.menuProtocolConfig.setObjectName("menuProtocolConfig")
        
        MainWindow.setMenuBar(self.menubar)
        
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setSizeGripEnabled(True)
        self.statusbar.setObjectName("statusbar")        
        MainWindow.setStatusBar(self.statusbar)
        
        self.actionExit = QtGui.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        
        self.actionUseGuan = QtGui.QAction(MainWindow)
        self.actionUseGuan.setCheckable(True)
        self.actionUseGuan.setObjectName("actionUseGuan")
        
        self.actionSetNumberAutomatons = QtGui.QAction(MainWindow)
        self.actionSetNumberAutomatons.setObjectName("actionSetNumberAutomatons")
        
        self.menuProtocolConfig.addAction(self.actionUseGuan)
        self.menuProtocolConfig.addAction(self.actionSetNumberAutomatons)
        
        self.menuOptions.addAction(self.menuProtocolConfig.menuAction())
        self.menuOptions.addAction(self.actionExit)
        
        self.menubar.addAction(self.menuOptions.menuAction())
        self.menubar.triggered[QtGui.QAction].connect(self.processtrigger)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
    def processtrigger(self, q):
        if(q == self.actionUseGuan and self.actionUseGuan.isChecked()):
            self.app.setRule('guan')
        
        if (q == self.actionSetNumberAutomatons):
            i, okPressed = QtGui.QInputDialog.getInt(self.centralwidget, 
                                                     "Numero de automatas",
                                                     'Seleccione la cantidad de automatas para la encripcion: ', 
                                                     self.app.automatons, 0, 100, 1)
            if okPressed:
                self.app.setAutomatons(int(i)) 
                
        if(q == self.actionExit):
            self.app.close()
                
    def changedNumberAutomatons(self, i):
        self.numberAutomatons.setText(str(i))
        
    def buttonBinaryChecked(self, enabled):
        if enabled:
            self.binary = True        
        
    def buttonTextChecked(self, enabled):
        if(enabled):
            self.binary = False
        
    def encryptGuanClicked(self):
        plain = self.textInput.toPlainText()
        enc = self.app.encrypt(plain, len(plain), 2, 'guan')
        self.textOutput.setPlainText(enc)

    def encryptClicked(self):
        plain = self.textInput.toPlainText()        

        if(self.binary == False):
            plain = Util.stringToBits(plain)
        else:
            plain = Util.binaryStrToBits(plain)
                
        enc = self.app.encrypt(plain, len(plain))
                
        if(self.binary == False):
            enc = Util.bitsToString(enc)
        else:
            enc = Util.bitsToBinaryStr(enc)
        
        self.textOutput.setPlainText(enc)
        
        self.textOutputPrivateKey.setPlainText(self.app.getPrivateKey())
        self.textOutputPublicKey.setPlainText(self.app.getPublicKey())
        
    def decryptClicked(self):
        plain = self.textInput.toPlainText()
        
        if(self.binary == False):
            plain = Util.stringToBits(plain)
        else:
            plain = Util.binaryStrToBits(plain)
        
        dec = self.app.decrypt(plain)
        
        if(self.binary == False):
            dec = Util.bitsToString(dec)
        else:
            dec = Util.bitsToBinaryStr(dec)        
        
        self.textOutput.setPlainText(dec)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QtGui.QApplication.translate("MainWindow", "Encriptor/Decriptor", None, QtGui.QApplication.UnicodeUTF8))
        self.decrypt.setText(QtGui.QApplication.translate("MainWindow", "Decriptar", None, QtGui.QApplication.UnicodeUTF8))
        self.encrypt.setText(QtGui.QApplication.translate("MainWindow", "Encriptar", None, QtGui.QApplication.UnicodeUTF8))
        self.titleWidget.setText(QtGui.QApplication.translate("MainWindow", "Encriptor/Decriptor", None, QtGui.QApplication.UnicodeUTF8))
        self.buttonGroupInput.setTitle(QtGui.QApplication.translate("MainWindow", "Entrada", None, QtGui.QApplication.UnicodeUTF8))
        self.radioButtonText.setText(QtGui.QApplication.translate("MainWindow", "Texto", None, QtGui.QApplication.UnicodeUTF8))
        self.radioButtonBinary.setText(QtGui.QApplication.translate("MainWindow", "Cadena binaria", None, QtGui.QApplication.UnicodeUTF8))
        self.automatons.setText(QtGui.QApplication.translate("MainWindow", "Número de autómatas:", None, QtGui.QApplication.UnicodeUTF8))
        #self.numberAutomatons.setText(QtGui.QApplication.translate("MainWindow", "2", None, QtGui.QApplication.UnicodeUTF8))
        self.privateKey.setText(QtGui.QApplication.translate("MainWindow", "Llave Privada:", None, QtGui.QApplication.UnicodeUTF8))
        self.publicKey.setText(QtGui.QApplication.translate("MainWindow", "Llave Pública:", None, QtGui.QApplication.UnicodeUTF8))
        self.menuOptions.setTitle(QtGui.QApplication.translate("MainWindow", "Opciones", None, QtGui.QApplication.UnicodeUTF8))
        self.menuProtocolConfig.setTitle(QtGui.QApplication.translate("MainWindow", "Configurar protocolo", None, QtGui.QApplication.UnicodeUTF8))
        self.actionExit.setText(QtGui.QApplication.translate("MainWindow", "Salir", None, QtGui.QApplication.UnicodeUTF8))
        self.actionUseGuan.setText(QtGui.QApplication.translate("MainWindow", "Regla de Guan", None, QtGui.QApplication.UnicodeUTF8))
        self.actionSetNumberAutomatons.setText(QtGui.QApplication.translate("MainWindow", "Cantidad autómatas", None, QtGui.QApplication.UnicodeUTF8))

if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))