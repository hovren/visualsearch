import sys
import os

import numpy as np
import PyQt5.QtWidgets as w
from PyQt5.QtCore import QFileInfo, QUrl, QSize

from vsearch import AnnDatabase, DatabaseError
from vsearch.gui import ImageWidget, ImageWithROI, LeafletWidget


NKPG_LAT, NKPG_LNG = 58.58923, 16.18035


class MainWindow(w.QMainWindow):
    def __init__(self):
        self.database = None

        super().__init__()
        self.setup_ui()
        self.show()

    def setup_ui(self):
        self.setWindowTitle('Visual search tool')

        # Widgets
        self.map_view = LeafletWidget(NKPG_LAT, NKPG_LNG)

        self.query_image = ImageWidget()
        self.query_image.setMinimumSize(QSize(256, 256))
        self.preview_image = ImageWidget()
        self.preview_image.setMinimumSize(QSize(256, 256))

        w.QButtonGroup
        load_database_button = w.QPushButton("Load database")
        load_database_button.clicked.connect(lambda *args: self.load_database('../test/test_db.h5'))

        # Layout
        vbox1 = w.QVBoxLayout()
        hbox = w.QHBoxLayout()
        vbox2 = w.QVBoxLayout()

        vbox2.addWidget(self.query_image)
        vbox2.addWidget(self.preview_image)

        hbox.addWidget(self.map_view)
        hbox.addLayout(vbox2)

        vbox1.addWidget(load_database_button)
        vbox1.addLayout(hbox)

        dummy = w.QWidget()
        dummy.setLayout(vbox1)

        self.setCentralWidget(dummy)

    def load_database(self, path):
        self.database = AnnDatabase.from_file(path)
        self.map_view.getCenter()

        sw_lat, sw_lng, ne_lat, ne_lng = self.map_view.getBounds()
        for i in range(3):
            lat = np.random.uniform(sw_lat, ne_lat)
            lng = np.random.uniform(sw_lng, ne_lng)
            print('Adding marker at', lat, lng)
            mid = self.map_view.add_marker(lat, lng)
            print('Marker ID:', mid)

        #bounds = [float(x) for x in frame.evaluateJavaScript('mymap.getBounds().toBBoxString();').split(",")]
        #sw_lng, sw_lat, ne_lng, ne_lat = bounds

        for key in self.database.image_vectors:
            pass


if __name__ == "__main__":
    app = w.QApplication(sys.argv)
    mwin = MainWindow()
    sys.exit(app.exec_())