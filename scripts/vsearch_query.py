import sys
import os
import collections

import numpy as np
import cv2
import PyQt5.QtWidgets as w
from PyQt5.QtCore import QFileInfo, QUrl, QSize

from vsearch import AnnDatabase, DatabaseError
from vsearch.database import LatLng, DatabaseEntry, DatabaseWithLocation
from vsearch.gui import ImageWidget, ImageWithROI, LeafletWidget, LeafletMarker

NORRKOPING = LatLng(58.58923, 16.18035)

QUERY_TAB = 0
DATABASE_TAB = 1


class MainWindow(w.QMainWindow):
    def __init__(self):
        self.database = None
        self.marker_id_mapping = {}

        super().__init__()
        self.setup_ui()
        self.show()

    def setup_ui(self):
        self.setWindowTitle('Visual search tool')

        # Widgets
        self.map_view = LeafletWidget(NORRKOPING.lat, NORRKOPING.lng)
        self.map_view.onMarkerClicked.connect(self.marker_clicked)
        self.map_view.onClick.connect(self.map_clicked)

        self.query_image = ImageWidget()
        self.query_image.setMinimumSize(QSize(256, 256))
        self.preview_image = ImageWidget()
        self.preview_image.setMinimumSize(QSize(256, 256))

        self.tab_widget = tab_widget = w.QTabWidget()
        tab_widget.currentChanged.connect(self.tab_changed)

        load_database_button = w.QPushButton("Load database")
        load_database_button.clicked.connect(lambda *args: self.load_database('../test/test_db.h5'))

        clear_markers_button = w.QPushButton("Clear markers")
        clear_markers_button.clicked.connect(lambda *args: self.map_view.clear_markers())

        # Layout
        vbox1 = w.QVBoxLayout()
        hbox = w.QHBoxLayout()
        button_hbox = w.QHBoxLayout()
        vbox2 = w.QVBoxLayout()
        tab2_layout = w.QVBoxLayout()

        self.query_page = QueryPage()
        tab_widget.addTab(self.query_page, "Query")
        self.database_page = DatabasePage()
        tab_widget.addTab(self.database_page, "Database")

        hbox.addWidget(self.map_view)
        hbox.addWidget(tab_widget)

        button_hbox.addWidget(load_database_button)

        vbox1.addLayout(button_hbox)
        vbox1.addLayout(hbox)

        dummy = w.QWidget()
        dummy.setLayout(vbox1)

        self.setCentralWidget(dummy)

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

    def map_clicked(self, lat, lng):
        if self.tab_widget.currentIndex() == DATABASE_TAB:
            pass

    def tab_changed(self, tabnr):
        if tabnr == 0: # Query
            self.enter_query_tab()
        elif tabnr == 1: # Database
            self.enter_database_tab()

    def remove_all_markers(self):
        if self.marker_id_mapping:
            print('Removing {:d} markers'.format(len(self.marker_id_mapping)))
            for key, m in self.marker_id_mapping.values():
                m.remove(update=False)
            self.marker_id_mapping.clear()
            self.map_view.update()

    def enter_query_tab(self):
        print('Query tab')
        self.remove_all_markers()

    def enter_database_tab(self):
        print('Database tab')
        if not self.image_locations:
            return

        self.remove_all_markers()
        keys, latlngs = zip(*list(self.image_locations.items()))
        markers = LeafletMarker.add_to_map(self.map_view, latlngs)

        for key, m in zip(keys, markers):
            self.marker_id_mapping[m.id] = (key, m)


    def load_database(self, path):
        visual_database = AnnDatabase.from_file(path)
        self.database = DatabaseWithLocation(visual_database)

        sw_lat, sw_lng, ne_lat, ne_lng = self.map_view.getBounds()

        for key in self.database:
            lat = np.random.uniform(sw_lat, ne_lat)
            lng = np.random.uniform(sw_lng, ne_lng)
            self.image_locations[key] = LatLng(lat, lng)
            item = w.QListWidgetItem(key)
            self.database_page.image_list.addItem(item)

        self.tab_widget.setCurrentIndex(DATABASE_TAB)


class DatabasePage(w.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setup_ui()

    def setup_ui(self):
        self.image = ImageWidget()
        self.image.setMinimumSize(QSize(256, 256))
        self.image_list = w.QListWidget()
        self.image_list.currentItemChanged.connect(self.on_item_changed)
        clear_button = w.QPushButton("Clear selection")
        clear_button.clicked.connect(lambda: self.image_list.setCurrentItem(None))

        vbox = w.QVBoxLayout()
        vbox.addWidget(self.image_list)
        vbox.addWidget(clear_button)
        vbox.addWidget(self.image)
        self.setLayout(vbox)

    def on_item_changed(self, current, prev):
        if current is None:
            self.image.set_array(None)
            self.populate()
        else:
            print(self.parent().database)


class QueryPage(w.QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        self.query_image = ImageWidget()
        self.query_image.setMinimumSize(QSize(256, 256))
        self.preview_image = ImageWidget()
        self.preview_image.setMinimumSize(QSize(256, 256))

        vbox = w.QVBoxLayout()
        vbox.addWidget(self.query_image)
        vbox.addWidget(self.preview_image)
        self.setLayout(vbox)


if __name__ == "__main__":
    app = w.QApplication(sys.argv)
    mwin = MainWindow()
    sys.exit(app.exec_())