import sys

import cv2

from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QSizePolicy, QGraphicsView, QGraphicsScene, QRubberBand
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import QtCore, Qt, QtGui
from PyQt5.Qt import QSize, QRect, QRectF, QPoint, QPointF

def printrect(r):
    return "Rect(({}, {}), {}x{})".format(r.x(), r.y(), r.width(), r.height())

class ImageWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        #self.setScaledContents(True)
        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.rubberband = None

        self.setStyleSheet("background-color: red;")

    def hasHeightForWidth(self):
        return self.pixmap() is not None

    def heightForWidth(self, w):
        if self.pixmap():
            return int(w * (self.pixmap().height() / self.pixmap().width()))

    def sizeHint(self):
        pix = self.pixmap()
        size = QSize() if pix is None else pix.size()
        self.updateGeometry()
        return size

    def map_to_image(self, widget_point):
        pix = self.pixmap()
        if pix:
            w_off = 0.5 * (self.width() - pix.width())
            h_off = 0.5 * (self.height() - pix.height())
            pixmap_point = QPointF(widget_point.x() - w_off, widget_point.y() - h_off)
            image_point = QPointF(pixmap_point.x() / pix.width(), pixmap_point.y() / pix.height())
            return image_point
        else:
            return QPointF()

    def map_from_image(self, image_point):
        pix = self.pixmap()
        if pix:
            w_off = 0.5 * (self.width() - pix.width())
            h_off = 0.5 * (self.height() - pix.height())
            pixmap_point = QPointF(image_point.x() * pix.width(), image_point.y() * pix.height())
            widget_point = QPointF(pixmap_point.x() + w_off, pixmap_point.y() + h_off)
            return widget_point.toPoint()
        else:
            return QPoint()

    def mousePressEvent(self, event):
        origin = event.pos()
        origin_global = event.globalPos()

        print('Widget pos:', origin, 'Image point:', self.map_to_image(origin), 'Backwards:', self.map_from_image(self.map_to_image(origin)))

        if self.pixmap() is not None:
            if self.pixmap().rect().contains(origin):
                self.origin = self.map_to_image(origin)
                if self.rubberband is None:
                    self.rubberband = QRubberBand(QRubberBand.Rectangle, self)
                else:
                    self.rubberband.hide()
                self.rubberband.setGeometry(QRect(self.map_from_image(self.origin), QSize()))
                self.rubberband.show()
            else:
                print('Pressed outside image')

    def mouseMoveEvent(self, event):
        if self.rubberband is not None:
            self.final = self.map_to_image(event.pos())
            self.update_rubberband()

    def update_rubberband(self):
        self.rubberband.setGeometry(QRect(self.map_from_image(self.origin), self.map_from_image(self.final)).normalized())

    def mouseReleaseEvent(self, event):
        if self.rubberband is not None:
            print(self.origin, self.final)
            self.update_rubberband()

    def resizeEvent(self, event):
        super().resizeEvent(event)

        if self.pixmap() and self.rubberband:
            self.update_rubberband()

class ImageWithKeyPoints(QWidget):
    def __init__(self, name):
        super().__init__()

        self.name = name
        self.initUI()

        self.image_path = None
        self.image = None # OpenCV / NumPy image array

    def initUI(self):
        self.setMinimumSize(300, 300)

        self.image_widget = ImageWidget()

        self.load_button = QPushButton("Load {}".format(self.name))
        self.load_button.clicked.connect(self.open_file)

        vbox = QVBoxLayout()
        vbox.addWidget(self.image_widget, 10)
        vbox.addWidget(self.load_button)
        self.setLayout(vbox)

    def open_file(self):
        path, *_ = QFileDialog.getOpenFileName()
        if path is None:
            return
        else:
            print(path, type(path))
            self.set_image(path)

    def set_image(self, image_path):
        if image_path is None:
            return

        self.image_path = image_path
        self.image = cv2.imread(self.image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.update_image()

    def resizeEvent(self, event):
        self.update_image()
        return super().resizeEvent(event)

    def update_image(self):
        if self.image is None:
            return

        height, width, channels = self.image.shape
        assert channels == 3
        bytes_per_line = channels * width
        qt_image = QImage(self.image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap(qt_image)
        self.image_widget.setPixmap(pix.scaled(self.image_widget.size(), QtCore.Qt.KeepAspectRatio | QtCore.Qt.SmoothTransformation))


class MatchComparerWindow(QMainWindow):
    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path
        self.initUI()

    def initUI(self):
        self.setGeometry(400, 400, 600, 300)
        self.setWindowTitle('Feature matches')

        self.left = ImageWithKeyPoints("left image")
        self.right = ImageWithKeyPoints("right image")

        hbox = QHBoxLayout()
        hbox.addWidget(self.left, 1)
        #hbox.addStretch(1)
        hbox.addWidget(self.right, 1)

        # Proxy Widget
        widget = QWidget()
        widget.setLayout(hbox)
        self.setCentralWidget(widget)

        self.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mwin = MatchComparerWindow(sys.argv[1])
    sys.exit(app.exec_())