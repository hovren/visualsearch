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


class GuiWrappedDatabase(DatabaseWithLocation):
    def __init__(self, visualdb, map_widget, image_root, geofile_path):
        super().__init__(visualdb)
        self.image_root = image_root
        self.geofile_path = geofile_path
        self.map_widget = map_widget
        self._marker_id_to_key = {}
        self._key_to_marker_id = {}

        self.load_from_geofile()

    def load_from_geofile(self):
        if not (self.geofile_path and os.path.exists(self.geofile_path)):
            return

        with open(self.geofile_path, 'r') as f:
            for line in f:
                try:
                    key, lat, lng = line.split(",")
                except ValueError:
                    pass
                key = key.strip()
                latlng = LatLng(float(lat), float(lng))
                old = self[key]
                new = DatabaseEntry(old.key, old.bow, latlng)

    def remove_all_markers(self):
        self.map_widget.remove_all_markers()
        self._marker_id_to_key.clear()
        self._key_to_marker_id.clear()

    def add_marker_for_key(self, key):
        if key not in self._key_to_marker_id:
            latlng = self[key].latlng
            if latlng:
                marker, *_ = LeafletMarker.add_to_map(self.map_widget, [latlng])
                self._marker_id_to_key[marker.id] = key
                self._key_to_marker_id[key] = marker.id
                return marker
        raise KeyError("Failed to add marker for key '{}'".format(key))

    def add_all_markers(self):
        pairs = [(key, e.latlng) for key, e in self.items() if e.latlng is not None]
        if pairs:
            keys, latlngs = zip(*pairs)
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

    def update_location(self, key, latlng):
        old = self[key]
        new = DatabaseEntry(old.key, old.bow, latlng)
        self[key] = new


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

        def load_db():
            dialog = LoadDatabaseDialog(self)
            if dialog.exec_():
                self.load_database(dialog.db_path, dialog.image_root, dialog.geofile)


        load_database_button.clicked.connect(load_db)

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

    def map_clicked(self, lat, lng):
        if self.tab_widget.currentIndex() == DATABASE_TAB:
            self.database_page.on_map_click(lat, lng)

    def tab_changed(self, tabnr):
        if tabnr == 0: # Query
            self.enter_query_tab()
        elif tabnr == 1: # Database
            self.database_page.on_enter()

    def enter_query_tab(self):
        print('Query tab')
        self.map_view.remove_all_markers()
        self.database_page.image_list.setCurrentItem(None)


    def load_database(self, path, image_root, geofile_path):
        visual_database = AnnDatabase.from_file(path)
        self.database = GuiWrappedDatabase(visual_database, self.map_view, image_root, geofile_path)

        if False:
            sw_lat, sw_lng, ne_lat, ne_lng = self.map_view.getBounds()
            for key, entry in self.database.items():
                lat = np.random.uniform(sw_lat, ne_lat)
                lng = np.random.uniform(sw_lng, ne_lng)
                self.database.update_location(key, LatLng(lat, lng))
                item = w.QListWidgetItem(key)
                self.database_page.image_list.addItem(item)

        self.database_page.load_database(self.database)
        self.tab_widget.setCurrentIndex(DATABASE_TAB)


class DatabasePage(w.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.database = None
        self.setup_ui()

    def load_database(self, database):
        self.database = database
        for key in database:
            self.image_list.addItem(w.QListWidgetItem(key))

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
            self.database.update_location(key, latlng)

    def on_map_click(self, lat, lng):
        # Update only if an item is selected and doesn't already have a marker on the map
        item = self.image_list.currentItem()
        if item:
            key = item.text()
            try:
                marker = self.database.marker_for_key(key)
            except KeyError:
                self.database.update_location(key, LatLng(lat, lng))
                marker = self.database.add_marker_for_key(key)
                marker.setDraggable(True)

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
            entry = self.database[key]
            path = os.path.join(self.database.image_root, entry.key + '.jpg')
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.image.set_array(image)

            try:
                marker = self.database.add_marker_for_key(key)
                marker.setDraggable(True)
            except KeyError:
                pass



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


class LineFileChooser(w.QWidget):
    def __init__(self, directory=False, parent=None, **kwargs):
        super().__init__(parent=parent)
        self.kwargs = kwargs
        self.line = w.QLineEdit()
        self.line.setEnabled(False)
        self.is_directory = directory
        self.button = w.QPushButton("Select")
        self.button.clicked.connect(self.load_file)
        hbox = w.QHBoxLayout()
        hbox.addWidget(self.line)
        hbox.addWidget(self.button)
        self.setLayout(hbox)

    def load_file(self):
        if self.is_directory:
            path = w.QFileDialog.getExistingDirectory(**self.kwargs)
        else:
            path, _ = w.QFileDialog.getOpenFileName(**self.kwargs)

        self.line.setText(path)


class LoadDatabaseDialog(w.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setModal(True)
        self.setWindowTitle("Load database")
        self._db_path = LineFileChooser(filter='*.h5')
        width = 400
        self._db_path.setMinimumWidth(width)
        self._image_root = LineFileChooser(directory=True)
        self._db_path.setMinimumWidth(width)
        self._geofile = LineFileChooser(filter='*.csv')
        self._geofile.setMinimumWidth(width)

        bb = w.QDialogButtonBox(w.QDialogButtonBox.Ok | w.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)

        form = w.QFormLayout()
        form.addRow("Visual database", self._db_path)
        form.addRow("Image root", self._image_root)
        form.addRow("Location file", self._geofile)
        form.addWidget(bb)

        self.setLayout(form)

    @property
    def db_path(self):
        return self._db_path.line.text()

    @property
    def image_root(self):
        return self._image_root.line.text()

    @property
    def geofile(self):
        return None


if __name__ == "__main__":
    app = w.QApplication(sys.argv)
    mwin = MainWindow()
    sys.exit(app.exec_())