# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 09:46:33 2017

@author: fabiotovar

sapnu puas
"""

import sys
from PySide import QtGui
from view.basicencryptor import Ui_MainWindow
from modules.crypto import encryption

class App:
    def __init__(self):
        self.app = QtGui.QApplication.instance()
        if not self.app:
            self.app = QtGui.QApplication('app.py')

        self.window = QtGui.QMainWindow()
        self.mainWindow = Ui_MainWindow(self)
        
        self.size = 5
        self.automatons = 2
        self.rule = 'None'
        self.encryptor = encryption(self.size, self.automatons)
        
        self.mainWindow.setupUi(self.window)
        
    def close(self):
        self.app.quit()

    def run(self):
        self.window.show()
        sys.exit(self.app.exec_())
        
    def setRule(self, rule):
        self.rule = rule
        
        if rule == 'guan':
            self.setAutomatons(2)
            self.size = 5
        
        self.encryptor = encryption(self.size, self.automatons, self.rule)
        
    def setAutomatons(self, automatons):
        if(automatons != self.automatons):
            print 'YAS'
            self.automatons = int(automatons)
            print 'YAS'
            self.encryptor = encryption(self.size, self.automatons, self.rule)
            print 'YAS'
            self.mainWindow.changedNumberAutomatons(self.automatons)
            print 'YAS'
        
    def encrypt(self, text, size):
        if(size != self.size):
            self.size = size
            self.encryptor = encryption(self.size, self.automatons, self.rule)
            
        return self.encryptor.encrypt(text)
    
    def decrypt(self, text):
        return self.encryptor.decrypt(text)
        
    def getPrivateKey(self):
        s = ''
        j = 1
        for i in self.encryptor.automatons:
            s += 'Automata ' + str(j) + ':\n' + str(i)
            s += 'blocks len: ' + str(i.blocklen) + '\n' 
            s += str(i.seq) + "\n\n"
            j += 1
    
        return s
        
    def getPublicKey(self):
        return str(self.encryptor.composite)
        
if __name__ == "__main__":
   app = App()
   app.run()