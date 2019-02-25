from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtGui import QPainter, QColor, QBrush, QFont, QPen
from PyQt5.QtCore import Qt
from NeuralNetwork import *
import struct
from PIL import Image


class Doodle(QWidget):
    def __init__(self, widgth, height):
        super().__init__()
        self.widgth = widgth
        self.height = height

    def initUI(self):
        self.setWindowTitle("aaaa")
        self.setGeometry(0,0, self.width, self.height)
        
        # Create widget
        label = QLabel(self)
        pixmap = QPixmap('image.jpeg')
        label.setPixmap(pixmap)
        self.resize(pixmap.width(),pixmap.height())

def tanh(x):
	return math.tanh(x)

def dtanh(y):
	return 1 - (y * y)

class MNIST:

    def __init__(self):
        # print("s")
        self.nn = NeuralNetwork([64,16,16,10],0.1,tanh,dtanh)
        self.train_labels = []
        self.train_images = []

    def initializate(self):
        with open("train-labels.idx1-ubyte", 'rb') as f:
            # reads the magic number
            struct.unpack('>I',f.read(4))[0]
            # reads the number of labels
            aux = struct.unpack('>I',f.read(4))[0]
            # saves labels
            for i in range(0, aux/4):
                self.train_labels.append(struct.unpack('>B',f.read(1))[0])

        with open('train-images.idx3-ubyte', 'rb') as f:
            # reads the magic number
            struct.unpack('>I',f.read(4))[0]
            # reads the number of labels
            aux = struct.unpack('>I',f.read(4))[0]
            print(aux)
            # number of rows
            rows = struct.unpack('>I',f.read(4))[0]
            print(rows)
            # number of cols
            cols = struct.unpack('>I',f.read(4))[0]
            print(cols)
            for i in range(0, aux/4):
                self.train_images.append([])
                for k in range(0,rows):
                    self.train_images[i].append([])
                    for n in range(0, cols):
                        self.train_images[i].append(struct.unpack('>B',f.read(1))[0])
                    print(len(self.train_images[i][0]))
                
        a = np.array(self.train_images[0]).reshape(rows,cols)
        img = Image.fromarray(a, 'RGB')
        img.save('my.png')
        

if __name__ == '__main__':
    a = MNIST()
    a.initializate()