from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, \
    QSizePolicy, QGraphicsView, QGraphicsScene, QRubberBand, QDialog, QDialogButtonBox, QDialogButtonBox, QErrorMessage, QProgressDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import QtCore, Qt, QtGui
from PyQt5.Qt import QSize, QRect, QRectF, QPoint, QPointF, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWebKitWidgets import QWebView

import numpy as np

class ImageWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_array = None
        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

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

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_image()

    def set_array(self, array):
        self.image_array = array
        self.update_image()

    def update_image(self):
        if self.image_array is None:
            self.clear()
            return

        height, width, channels = self.image_array.shape
        assert channels == 3
        bytes_per_line = channels * width
        qt_image = QImage(self.image_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap(qt_image)
        self.setPixmap(pix.scaled(self.size(), QtCore.Qt.KeepAspectRatio | QtCore.Qt.SmoothTransformation))

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

class ImageWithROI(ImageWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rubberband = None
        self.origin = None
        self.final = None

    def set_array(self, array, preserve_roi=False):
        super().set_array(array)
        if self.rubberband and not preserve_roi:
            self.rubberband.hide()
            self.rubberband = None

    def mousePressEvent(self, event):
        origin = event.pos()

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
            self.update_rubberband()

    def resizeEvent(self, event):
        super().resizeEvent(event)

        if self.pixmap() and self.rubberband:
            self.update_rubberband()

    def get_rubberband_rect(self):
        if self.origin and self.final:
            return QRectF(self.origin, self.final).normalized()
        else:
            return None

    def get_image_and_roi(self):
        roi_norm = self.get_rubberband_rect() or QRectF(0, 0, 1, 1)

        if self.image_array is not None:
            h, w, *_ = self.image_array.shape
            rx = int(np.round(roi_norm.x() * w))
            ry = int(np.round(roi_norm.y() * h))
            rw = int(np.round(roi_norm.width() * w))
            rh = int(np.round(roi_norm.height() * h))
            roi = (rx, ry, rw, rh)
            return self.image_array, roi
        else:
            raise ValueError("No image set")