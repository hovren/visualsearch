import sys
import os

import numpy as np
import cv2
import PyQt5.QtWidgets as w
from PyQt5.QtCore import QFileInfo, QUrl, QSize

from vsearch import AnnDatabase, DatabaseError
from vsearch.gui import ImageWidget, ImageWithROI, LeafletWidget


NKPG_LAT, NKPG_LNG = 58.58923, 16.18035


class MainWindow(w.QMainWindow):
    def __init__(self):
        self.database = None
        self.marker_id_mapping = {}

        super().__init__()
        self.setup_ui()
        self.show()

    def marker_clicked(self, marker_id):
        key, marker = self.marker_id_mapping[marker_id]
        print('Marker with ID {} and key {}'.format(marker.id, key))
        database_dir = '/home/hannes/Datasets/narrative2'
        path = os.path.join(database_dir, key + '.jpg')
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.preview_image.set_array(img)

        for (key, other) in self.marker_id_mapping.values():
            if not other is marker:
                other.setOpacity(0.5)

    def setup_ui(self):
        self.setWindowTitle('Visual search tool')

        # Widgets
        self.map_view = LeafletWidget(NKPG_LAT, NKPG_LNG)
        self.map_view.onMarkerClicked.connect(self.marker_clicked)

        self.query_image = ImageWidget()
        self.query_image.setMinimumSize(QSize(256, 256))
        self.preview_image = ImageWidget()
        self.preview_image.setMinimumSize(QSize(256, 256))

        w.QButtonGroup
        load_database_button = w.QPushButton("Load database")
        load_database_button.clicked.connect(lambda *args: self.load_database('../test/test_db.h5'))

        clear_markers_button = w.QPushButton("Clear markers")
        clear_markers_button.clicked.connect(lambda *args: self.map_view.clear_markers())

        # Layout
        vbox1 = w.QVBoxLayout()
        hbox = w.QHBoxLayout()
        button_hbox = w.QHBoxLayout()
        vbox2 = w.QVBoxLayout()

        vbox2.addWidget(self.query_image)
        vbox2.addWidget(self.preview_image)

        hbox.addWidget(self.map_view)
        hbox.addLayout(vbox2)

        button_hbox.addWidget(load_database_button)

        vbox1.addLayout(button_hbox)
        vbox1.addLayout(hbox)

        dummy = w.QWidget()
        dummy.setLayout(vbox1)

        self.setCentralWidget(dummy)

    def load_database(self, path):
        self.database = AnnDatabase.from_file(path)
        self.map_view.getCenter()

        sw_lat, sw_lng, ne_lat, ne_lng = self.map_view.getBounds()

        from vsearch.gui.leaflet import LeafletMarker

        latlngs = []
        keys = []
        for key in self.database.image_vectors:
            lat = np.random.uniform(sw_lat, ne_lat)
            lng = np.random.uniform(sw_lng, ne_lng)
            latlngs.append([lat, lng])
            keys.append(key)

        markers = LeafletMarker.add_to_map(self.map_view, latlngs)

        for key, m in zip(keys, markers):
            self.marker_id_mapping[m.id] = (key, m)

if __name__ == "__main__":
    app = w.QApplication(sys.argv)
    mwin = MainWindow()
    sys.exit(app.exec_())