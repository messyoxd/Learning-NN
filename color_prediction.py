from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtGui import QPainter, QColor, QBrush, QFont, QPen
from PyQt5.QtCore import Qt
import sys
import random
from NeuralNetwork import *

class Example(QWidget):

    def __init__(self, widgth, height):
        super().__init__()
        self.widgth = widgth
        self.height = height
        self.text1 = "black"
        self.text2 = "white"
        self.initUI()
        self.r = 0
        self.g = 0
        self.b = 0

        self.nn = NeuralNetwork(3,3,2)

        for i in range(0, 100000):
            r = random.randint(0,255)
            g = random.randint(0,255)
            b = random.randint(0,255)
            targets = self.predictColor(r,g,b)
            inputs = [r/255, g/255, b/255]
            self.nn.train(inputs, targets)

        choice = self.nn.feedfoward(inputs)
        if choice[0] > choice[1]:
            self.which = "black"
        else:
            self.which = "white"

    def initUI(self):

        self.setGeometry(300, 300, self.widgth, self.height)
        self.setWindowTitle('Colours')
        self.show()


    def paintEvent(self, e):

        qp = QPainter()
        qp.begin(self)
        self.drawRectangles(qp)
        self.drawText(e, qp)
        self.drawLines(qp)
        self.predict(qp)
        qp.end()

    def predictColor(self, r,g,b):
        if(r+g+b > 300):
            return [1,0]
        else:
            return [0,1]

    def predict(self, qp):
        inputs = [self.r/255, self.g/255, self.b/255]
        choice = self.nn.feedfoward(inputs)
        if choice[0] > choice[1]:
            self.which = "black"
        else:
            self.which = "white"

        print(choice[0],choice[1])
        if(self.which == "black"):
            col = QColor(0, 0, 0)
            col.setNamedColor('#d4d4d4')
            qp.setPen(col)

            qp.setBrush(QColor(0, 0, 0))
            qp.drawRect(60, 100, 40, 40)
        else:
            col = QColor(0, 0, 0)
            col.setNamedColor('#d4d4d4')
            qp.setPen(col)

            qp.setBrush(QColor(255, 255, 255))
            qp.drawRect(self.widgth - 60, 100, 40, 40)

    def drawRectangles(self, qp):

        col = QColor(0, 0, 0)
        col.setNamedColor('#d4d4d4')
        qp.setPen(col)

        self.r = random.randint(0,255)
        self.g = random.randint(0,255)
        self.b = random.randint(0,255)

        qp.setBrush(QColor(self.r, self.g, self.b))
        qp.drawRect(0, 0, self.widgth, self.height)

    def drawText(self, event, qp):
        qp.setPen(QColor(0,0,0))
        qp.setFont(QFont('Decorative', 50))
        qp.drawText(event.rect(),Qt.AlignLeft, self.text1)
        qp.setPen(QColor(255,255,255))
        qp.setFont(QFont('Decorative', 50))
        qp.drawText(event.rect(),Qt.AlignRight, self.text2)

    def drawLines(self, qp):
        pen = QPen(Qt.black, 4, Qt.SolidLine)
        qp.setPen(pen)
        qp.drawLine(300, 0, 300, 400)

    def mousePressEvent(self, e):
        # if e.x() > self.widgth/2:
        #     targets = [0,1]
        # else:
        #     targets = [1,0]
        #
        # inputs = [self.r/255, self.g/255, self.b/255]
        #
        # self.nn.train(inputs,targets)

        self.update()



if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = Example(600,400)
    sys.exit(app.exec_())
