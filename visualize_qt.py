import sys
import os
import collections
import threading

import cv2
import numpy as np

from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, \
    QSizePolicy, QGraphicsView, QGraphicsScene, QRubberBand, QDialog, QDialogButtonBox, QDialogButtonBox, QErrorMessage, QProgressDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import QtCore, Qt, QtGui
from PyQt5.Qt import QSize, QRect, QRectF, QPoint, QPointF, QThread, pyqtSignal, pyqtSlot

from vsim_common import load_SIFT_file, sift_file_for_image, inside_roi

ImageData = collections.namedtuple('ImageData', ['path', 'image', 'keypoints', 'descriptors', 'roi'])

def filter_roi(imd):
    rx, ry, rw, rh = [int(v) for v in imd.roi]
    im = imd.image[ry:ry+rh, rx:rx+rw]
    valid = [i for i, kp in enumerate(imd.keypoints) if inside_roi(kp, imd.roi)]
    # We are modifying the keypoints, so they must be copied first
    def new_roi_keypoint(kp):
        x, y = kp.pt
        return cv2.KeyPoint(x - rx, y - ry, kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
    kps = [new_roi_keypoint(kp) for i, kp in enumerate(imd.keypoints) if i in valid]
    des = imd.descriptors[valid]
    return kps, des, im

class ImageWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_array = None
        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        #self.setStyleSheet("background-color: red;")

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

class ImageWidgetError(Exception):
    pass

class ImageWithROI(ImageWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rubberband = None
        self.origin = None
        self.final = None

    def mousePressEvent(self, event):
        origin = event.pos()
        origin_global = event.globalPos()

        #print('Widget pos:', origin, 'Image point:', self.map_to_image(origin), 'Backwards:', self.map_from_image(self.map_to_image(origin)))

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
            raise ImageWidgetError


class MatcherWorker(QThread):
    success_signal = pyqtSignal(np.ndarray)
    fail_signal = pyqtSignal(str)
    match_progress = pyqtSignal(int, str)

    def __init__(self, imdata1, imdata2, parent=None):
        super().__init__(parent)
        self.imd1 = imdata1
        self.imd2 = imdata2
        self.log_message('Keypoints: {} and {}'.format(len(self.imd1.keypoints), len(self.imd2.keypoints)))

    def log_message(self, text):
        tid = threading.current_thread().getName()
        print('Thread[{}]: {}'.format(tid, text))

    def run(self):
        self.log_message("Starting")
        self.log_message('Keypoints: {} and {}'.format(len(self.imd1.keypoints), len(self.imd2.keypoints)))
        self.log_message("Image 1 ROI: {}".format(self.imd1.roi))
        self.log_message("Image 2 ROI: {}".format(self.imd2.roi))
        self.match_progress.emit(0, "Filtering Image 1")
        kps1, des1, im1_roi = filter_roi(self.imd1)
        self.log_message("Image 1 had {:d} keypoints within ROI".format(len(kps1)))
        self.log_message("Filtered 1")
        self.match_progress.emit(20, "Filtering Image 2")
        kps2, des2, im2_roi = filter_roi(self.imd2)
        self.log_message("Image 2 had {:d} keypoints within ROI".format(len(kps2)))
        self.log_message("Filtered 2")

        self.match_progress.emit(40, "Matching features")
        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(des1, des2, k=2)
        self.log_message("Got {:d} matches".format(len(matches)))
        if not matches:
            self.fail_signal.emit("Found no matches")

        def good_match(m1, m2, max_distance=np.inf):
            assert m1.distance <= m2.distance, "Not ordered, distances: {} and {}".format(m1.distance, m2.distance)
            return (m1.distance < 0.75 * m2.distance) and m1.distance < max_distance

        ratio_matches = [m1 for m1, m2 in matches if good_match(m1, m2)]
        self.log_message('Ratio test reduced to {:d} matches'.format(len(ratio_matches)))

        if ratio_matches:
            self.log_message("Drawing")
            self.match_progress.emit(75, "Drawing result")
            flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS | cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
            cmp_image = cv2.drawMatches(im1_roi, kps1, im2_roi, kps2, ratio_matches, np.array([]), flags=flags)
            self.log_message("Done drawing")
            self.success_signal.emit(cmp_image)
        else:
            self.log_message("Nothing to draw")
            self.fail_signal.emit("No matches to draw")
        self.match_progress.emit(100, "Done")
        self.log_message("Exiting")


class ImageSelectDialog(QDialog):
    def __init__(self, parent = None):
        super().__init__(parent)

        self.setMinimumSize(QSize(400, 400))
        vbox = QVBoxLayout()
        load_button = QPushButton("Load Image")
        load_button.clicked.connect(self.load_image)
        self.image_widget = ImageWithROI()
        self.image_widget.setText("Open an image file")
        vbox.addWidget(load_button)
        vbox.addWidget(self.image_widget, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, QtCore.Qt.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        vbox.addWidget(buttons)

        self.setLayout(vbox)

    def accept(self):
        return super().accept()

    def load_image(self):
        path, *args = QFileDialog.getOpenFileName(filter='Images (*.jpg)')
        print('Path', path, 'args', args)
        if not path:
            return

        image = cv2.imread(path)
        if image is None:
            QErrorMessage(self).showMessage("Failed to load image file")
            return
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        sift_file = sift_file_for_image(path)
        if not os.path.exists(sift_file):
            QErrorMessage(self).showMessage("The selected file did not have an accompanying sift descriptor file")
            return

        des, kps = load_SIFT_file(sift_file)
        self.sift_descriptors = des
        self.sift_keypoints = kps
        self.image_path = path
        self.image_widget.set_array(image)

    def get_image_data(self):
        image, roi = self.image_widget.get_image_and_roi()
        return ImageData(self.image_path, image, self.sift_keypoints, self.sift_descriptors, roi)

class MatchComparerWindow(QMainWindow):
    def __init__(self, image_path):
        super().__init__()
        self.left_image_dialog = ImageSelectDialog(self)
        self.right_image_dialog = ImageSelectDialog(self)
        self.initUI()

    def initUI(self):
        self.setGeometry(400, 400, 600, 300)
        self.setWindowTitle('Feature matches')

        vbox = QVBoxLayout()
        hbox = QHBoxLayout()
        left_button = QPushButton("Setup Image 1")
        left_button.clicked.connect(lambda: self.run_dialog(self.left_image_dialog))
        right_button = QPushButton("Setup Image 2")
        right_button.clicked.connect(lambda: self.run_dialog(self.right_image_dialog))
        hbox.addWidget(left_button)
        hbox.addWidget(right_button)
        self.image = ImageWithROI()
        self.image.setText("Set the two images to use")
        vbox.addLayout(hbox)
        vbox.addWidget(self.image, 1)

        # Proxy Widget
        widget = QWidget()
        widget.setLayout(vbox)
        self.setCentralWidget(widget)

        self.show()

    def run_dialog(self, dialog):
        retval = dialog.exec_()
        if retval:
            self.update_matches()

    def update_matches(self):
        try:
            imdata1 = self.left_image_dialog.get_image_data()
            imdata2 = self.right_image_dialog.get_image_data()
        except ImageWidgetError:
            return

        worker = MatcherWorker(imdata1, imdata2, parent=self)
        worker.success_signal.connect(self.image.set_array)
        worker.fail_signal.connect(self.image.setText)

        if True:
            progress = QProgressDialog(parent=self)
            def progress_update(value, text):
                progress.setValue(value)
                progress.setLabelText(text)
            worker.match_progress.connect(progress_update)
            progress.setMinimum(0)
            progress.setMaximum(100)
            progress.setModal(True)
            progress.setWindowTitle("Matching features")

            def cancel_action():
                print('Disconnecting signals')
                worker.success_signal.disconnect()
                worker.fail_signal.disconnect()
                worker.match_progress.disconnect()
                print('Signals disconnected')
                #progress.deleteLater()
                #worker.deleteLater()

            progress.canceled.connect(cancel_action)

        worker.start()
        print('Thread started', worker)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mwin = MatchComparerWindow(sys.argv[1])
    sys.exit(app.exec_())