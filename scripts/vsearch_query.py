import sys
import os
import collections

import numpy as np
import cv2
import PyQt5.QtWidgets as w
from PyQt5.QtCore import QFileInfo, QUrl, QSize
from PyQt5.QtCore import Qt

from vsearch import AnnDatabase, DatabaseError
from vsearch.database import LatLng, DatabaseEntry, DatabaseWithLocation
from vsearch.gui import ImageWidget, ImageWithROI, LeafletWidget, LeafletMarker

NORRKOPING = LatLng(58.58923, 16.18035)

QUERY_TAB = 0
DATABASE_TAB = 1


class DatabaseWithLocationAndMarkers(DatabaseWithLocation):
    def __init__(self, visualdb, map_widget):
        super().__init__(visualdb)
        self.map_widget = map_widget
        self._marker_id_to_key = {}
        self._key_to_marker_id = {}

    def remove_all_markers(self):
        self.map_widget.remove_all_markers()
        self._marker_id_to_key.clear()
        self._key_to_marker_id.clear()

    def add_marker_for_key(self, key):
        if key not in self._key_to_marker_id:
            print('Adding marker')
            marker, *_ = LeafletMarker.add_to_map(self.map_widget, [self[key].latlng])
            self._marker_id_to_key[marker.id] = key
            self._key_to_marker_id[key] = marker.id
            return marker
        else:
            print('Marker already existed for key', key)

    def add_all_markers(self):
        keys, latlngs = zip(*[(key, e.latlng) for key, e in self.items()])
        markers = LeafletMarker.add_to_map(self.map_widget, latlngs)
        d1 = {m.id: key for m, key in zip(markers, keys)}
        d2 = {key: m.id for m, key in zip(markers, keys)}
        self._marker_id_to_key.update(d1)
        self._key_to_marker_id.update(d2)

    def key_for_marker(self, marker_id):
        return self._marker_id_to_key[marker_id]

    def marker_for_key(self, key):
        mid = self._key_to_marker_id[key]
        return self.map_widget.markers[mid]

class MainWindow(w.QMainWindow):
    def __init__(self):
        self.database = None

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
        self.database_page = DatabasePage(self.database)
        tab_widget.addTab(self.query_page, "Query")
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
        print('clicked marker #{:d}'.format(marker_id))
        if self.tab_widget.currentIndex() == DATABASE_TAB:
            self.database_page.select_by_marker(marker_id)
        else:
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
            self.database_page.on_enter()

    def enter_query_tab(self):
        print('Query tab')
        self.map_view.remove_all_markers()
        self.database_page.image_list.setCurrentItem(None)


    def load_database(self, path):
        visual_database = AnnDatabase.from_file(path)
        #self.database = DatabaseWithLocation(visual_database)
        self.database = DatabaseWithLocationAndMarkers(visual_database, self.map_view)

        sw_lat, sw_lng, ne_lat, ne_lng = self.map_view.getBounds()

        for key, entry in self.database.items():
            lat = np.random.uniform(sw_lat, ne_lat)
            lng = np.random.uniform(sw_lng, ne_lng)
            new_entry = DatabaseEntry(key, entry.bow, LatLng(lat, lng))
            self.database[key] = new_entry
            item = w.QListWidgetItem(key)
            self.database_page.image_list.addItem(item)

        self.database_page.database = self.database
        self.tab_widget.setCurrentIndex(DATABASE_TAB)


class DatabasePage(w.QWidget):
    def __init__(self, database, parent=None):
        super().__init__(parent=parent)
        self.database = database
        self.setup_ui()

    def setup_ui(self):
        self.image = ImageWidget()
        self.image.setMinimumSize(QSize(256, 256))
        self.image_list = w.QListWidget()
        self.image_list.currentItemChanged.connect(self.on_item_changed)

        clear_button = w.QPushButton("Clear selection")
        clear_button.clicked.connect(lambda: self.image_list.setCurrentItem(None))

        save_location_button = w.QPushButton("Save location")
        save_location_button.clicked.connect(self.save_location_clicked)

        button_box = w.QHBoxLayout()
        button_box.addWidget(clear_button)
        button_box.addWidget(save_location_button)


        vbox = w.QVBoxLayout()
        vbox.addWidget(self.image_list)
        #vbox.addWidget(clear_button)
        vbox.addLayout(button_box)
        vbox.addWidget(self.image)
        self.setLayout(vbox)

    def on_enter(self):
        if not self.database:
            return
        self.database.remove_all_markers()
        self.database.add_all_markers()

    def save_location_clicked(self):
        item = self.image_list.currentItem()
        if item:
            key = item.text()
            marker = self.database.marker_for_key(key)
            latlng = LatLng(*marker.latlng)
            old = self.database[key]
            new = DatabaseEntry(old.key, old.bow, latlng)
            self.database[key] = new


    def select_by_marker(self, marker_id):
        key = self.database.key_for_marker(marker_id)
        print('{:d} -> {:s}'.format(marker_id, key))
        return self.select_by_key(key)

    def select_by_key(self, key):
        items = self.image_list.findItems(key, Qt.MatchExactly)
        if len(items) > 1:
            raise ValueError("List contained multiple entries with same key")
        elif len(items) < 1:
            raise ValueError("No such key in item list")
        else:
            item = items[0]
            self.image_list.setCurrentItem(item)

    def on_item_changed(self, current, prev):
        self.database.remove_all_markers()

        if current is None:
            self.image.set_array(None)
            self.on_enter()
        elif self.database is not None:
            key = current.text()
            marker = self.database.add_marker_for_key(key)
            print(marker)
            entry = self.database[key]
            path = os.path.join('/home/hannes/Datasets/narrative2/', entry.key + '.jpg')
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.image.set_array(image)

            marker.setDraggable(True)



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