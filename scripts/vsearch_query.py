import sys
import os
import collections

import numpy as np
import cv2
import PyQt5.QtWidgets as w
from PyQt5.QtCore import QFileInfo, QUrl, QSize, QThread, pyqtSignal
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon

from vsearch import AnnDatabase, DatabaseError
from vsearch.database import LatLng, DatabaseEntry, DatabaseWithLocation
from vsearch.gui import ImageWidget, ImageWithROI, LeafletWidget, LeafletMarker
from vsearch.utils import load_descriptors_and_keypoints, sift_file_for_image, filter_roi, calculate_sift

NORRKOPING = LatLng(58.58923, 16.18035)

QUERY_TAB = 1
DATABASE_TAB = 0


class GuiWrappedDatabase(DatabaseWithLocation):
    def __init__(self):
        super().__init__(None)
        self.image_root = None
        self.geofile_path = None
        self.map_widget = None
        self._marker_id_to_key = {}
        self._key_to_marker_id = {}


    def load_data(self, visualdb, image_root, geofile):
        self.geofile_path = geofile
        self.image_root = image_root
        self.visualdb = visualdb

        self.load_geofile()

    def load_geofile(self):
        print('Loading geo locations from', self.geofile_path)
        if not (self.geofile_path and os.path.exists(self.geofile_path)):
            print('Error in geofile path')
            return

        with open(self.geofile_path, 'r') as f:
            for line in f:
                try:
                    key, lat, lng = line.split(",")
                except ValueError:
                    pass
                key = key.strip()
                latlng = LatLng(float(lat), float(lng))
                print(key, latlng)
                self.update_location(key, latlng)

    def save_geofile(self):
        if not self.geofile_path:
            raise ValueError("No geofile path set")

        with open(self.geofile_path, 'w') as f:
            for entry in self.values():
                if entry.latlng:
                    f.write('{:s}, {:f}, {:f}\n'.format(entry.key, entry.latlng.lat, entry.latlng.lng))

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
        super().__init__()
        self.database = GuiWrappedDatabase()
        self.setup_ui()
        self.database.map_widget = self.map_view
        self.show()

    def setup_ui(self):
        self.setWindowTitle('Visual search tool')

        # Widgets
        self.map_view = LeafletWidget(NORRKOPING.lat, NORRKOPING.lng)
        self.map_view.onMarkerClicked.connect(self.marker_clicked)
        self.map_view.onMarkerMoved.connect(self.marker_moved)
        self.map_view.onClick.connect(self.map_clicked)

        self.query_image = ImageWidget()
        self.query_image.setMinimumSize(QSize(256, 256))
        self.preview_image = ImageWidget()
        self.preview_image.setMinimumSize(QSize(256, 256))

        self.tab_widget = tab_widget = w.QTabWidget()
        tab_widget.currentChanged.connect(self.tab_changed)

        # Layout
        vbox1 = w.QVBoxLayout()

        self.query_page = QueryPage(self.database)
        self.database_page = DatabasePage(self.database)
        tab_widget.addTab(self.database_page, "Database")
        tab_widget.addTab(self.query_page, "Query")

        splitter = w.QSplitter(self)
        splitter.addWidget(self.map_view)
        splitter.addWidget(tab_widget)

        vbox1.addWidget(splitter)

        dummy = w.QWidget()
        dummy.setLayout(vbox1)

        self.setCentralWidget(dummy)

    def marker_clicked(self, marker_id):
        print('clicked marker #{:d}'.format(marker_id))
        if self.tab_widget.currentIndex() == DATABASE_TAB:
            self.database_page.select_by_marker(marker_id)
        elif self.tab_widget.currentIndex() == QUERY_TAB:
            self.query_page.select_by_marker(marker_id)

    def marker_moved(self, marker_id):
        print('Moved marker', marker_id)
        if self.tab_widget.currentIndex() == DATABASE_TAB:
            m = self.map_view.markers[marker_id]
            self.database_page.on_marker_moved(marker_id, LatLng(*m.latlng))

    def map_clicked(self, lat, lng):
        if self.tab_widget.currentIndex() == DATABASE_TAB:
            self.database_page.on_map_click(lat, lng)
        elif self.tab_widget.currentIndex() == QUERY_TAB:
            self.query_page.on_map_click(lat, lng)

    def tab_changed(self, tabnr):
        if tabnr == QUERY_TAB: # Query
            self.enter_query_tab()
        elif tabnr == DATABASE_TAB: # Database
            self.database_page.on_enter()

    def enter_query_tab(self):
        print('Query tab')
        self.map_view.remove_all_markers()
        self.database_page.image_list.setCurrentItem(None)


class DatabasePage(w.QWidget):
    def __init__(self, database, parent=None):
        super().__init__(parent=parent)
        self.database = database
        self._icon = QIcon.fromTheme("applications-internet")
        self.setup_ui()

    def setup_ui(self):
        self.image = ImageWidget()
        self.image.setMinimumSize(QSize(256, 256))
        self.image_list = w.QListWidget()
        self.image_list.setIconSize(QSize(16, 16))
        self.image_list.currentItemChanged.connect(self.on_item_changed)

        self.load_database_button = w.QPushButton("Load database")
        self.load_database_button.clicked.connect(self.on_load_database)

        save_geo_button = w.QPushButton("Save Locations")
        save_geo_button.clicked.connect(self.on_save_locations)

        clear_button = w.QPushButton("Clear selection")
        clear_button.clicked.connect(lambda: self.image_list.setCurrentItem(None))

        clear_location_button = w.QPushButton("Clear location")
        clear_location_button.clicked.connect(self.clear_location_clicked)

        use_as_query_button = w.QPushButton("Use as Query")

        button_box = w.QHBoxLayout()
        button_box.addWidget(clear_button)
        button_box.addWidget(clear_location_button)
        button_box.addWidget(use_as_query_button)

        vbox = w.QVBoxLayout()
        vbox.addWidget(self.load_database_button)
        vbox.addWidget(self.image_list)
        vbox.addLayout(button_box)
        vbox.addWidget(self.image)
        vbox.addWidget(save_geo_button)
        self.setLayout(vbox)

    def on_enter(self):
        if not self.database:
            return
        self.database.remove_all_markers()
        self.database.add_all_markers()

    def on_marker_moved(self, marker_id, latlng):
        key = self.database.key_for_marker(marker_id)
        self.database.update_location(key, latlng)

    def on_load_database(self):
        dialog = LoadDatabaseDialog(self)
        if dialog.exec_():
            self.load_database(dialog.db_path, dialog.image_root, geofile_path=None)

    def load_database(self, path, image_root, geofile_path=None):
        visual_database = AnnDatabase.from_file(path)

        for key in visual_database.image_vectors:
            path = os.path.join(image_root, key)
            if not os.path.exists(path):
                msg = "Database file/key '{}' not found in image directory '{}'. Please try again.".format(key, image_root)
                w.QErrorMessage(self).showMessage(msg)
                return

        if not geofile_path:
            geofile_path = os.path.join(image_root, 'geo.csv')

        self.database.load_data(visual_database, image_root, geofile_path)

        for key, entry in self.database.items():
            item = w.QListWidgetItem(key)
            if entry.latlng:
                item.setIcon(self._icon)
            self.image_list.addItem(item)

        self.image_list.sortItems()
        self.load_database_button.setDisabled(True)
        self.on_enter() # Draw markers on map

    def on_save_locations(self):
        self.database.save_geofile()

    def clear_location_clicked(self):
        item = self.image_list.currentItem()
        if item:
            key = item.text()
            try:
                marker = self.database.marker_for_key(key)
                self.database.update_location(key, None)
                marker.remove()
            except KeyError:
                pass # No marker

    def on_map_click(self, lat, lng):
        # Update only if an item is selected and doesn't already have a marker on the map
        item = self.image_list.currentItem()
        if item:
            key = item.text()
            try:
                # If there is a marker then clear selection (unselect current)
                marker = self.database.marker_for_key(key)
                self.image_list.setCurrentItem(None)
            except KeyError: # No marker on map
                self.database.update_location(key, LatLng(lat, lng))
                marker = self.database.add_marker_for_key(key)
                item.setIcon(self._icon)
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
            path = os.path.join(self.database.image_root, entry.key)
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.image.set_array(image)

            try:
                marker = self.database.add_marker_for_key(key)
                marker.setDraggable(True)
            except KeyError:
                pass



class QueryPage(w.QWidget):
    def __init__(self, database):
        super().__init__()
        self.database = database
        self.matches = []
        self.setup_ui()

    def setup_ui(self):
        self.query_image = ImageWidget()
        self.query_image.setMinimumSize(QSize(256, 256))
        self.query_dialog = QueryImageDialog(self)
        self.preview_image = ImageWidget()
        self.preview_image.setMinimumSize(QSize(256, 256))
        self.result_list = w.QListWidget()
        self.result_list.setMinimumHeight(150)
        self.result_list.currentItemChanged.connect(self.on_item_changed)
        edit_query_button = w.QPushButton("Query")
        edit_query_button.clicked.connect(self.on_set_query)
        clear_button = w.QPushButton("Clear selection")
        clear_button.clicked.connect(lambda: self.result_list.setCurrentItem(None))

        vbox = w.QVBoxLayout()
        vbox.addWidget(edit_query_button)
        vbox.addWidget(self.query_image)
        vbox.addWidget(w.QLabel("Result list"))
        vbox.addWidget(self.result_list)
        vbox.addWidget(clear_button)

        vbox.addWidget(self.preview_image)
        self.setLayout(vbox)

    def on_item_changed(self, current, prev):
        self.database.remove_all_markers()

        if current is None:
            self.preview_image.set_array(None)
            self.add_markers_for_matches()
        else:
            key, *_ = current.text().split(" ")
            entry = self.database[key]
            path = os.path.join(self.database.image_root, entry.key)
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.preview_image.set_array(image)
            try:
                marker = self.database.add_marker_for_key(key)
            except KeyError:
                pass

    def on_set_query(self):
        if self.query_dialog.exec_():
            self.result_list.clear()

            image_path = self.query_dialog.image_path
            image, roi = self.query_dialog.image.get_image_and_roi()
            x, y, width, height = roi
            patch = np.copy(image[y:y+height, x:x+width])
            self.query_image.set_array(patch)

            self._search_thread = SearchThread(self.database, image, roi, image_path)

            progress = w.QProgressDialog("Loading SIFT features", "Abort", 0, 100, parent=self)
            progress.setModal(True)

            def progress_cb(percentage, status):
                print("[{:d}%] {}".format(percentage, status))
                progress.setValue(percentage)
                progress.setLabelText(status)
            self._search_thread.progress_update.connect(progress_cb)

            self._search_thread.finished.connect(self.on_search_done)

            progress.show()
            self._search_thread.start()

    def add_markers_for_matches(self):
        for entry, score in self.matches:
            self.database.add_marker_for_key(entry.key)

    def on_search_done(self, matches):
        self.matches = matches
        for entry, score in matches:
            text = "{} ({:.4f})".format(entry.key, score)
            self.result_list.addItem(text)
        self.database.remove_all_markers()
        self.add_markers_for_matches()

    def on_map_click(self, lat, lng):
        self.result_list.setCurrentItem(None)

    def select_by_marker(self, marker_id):
        key = self.database.key_for_marker(marker_id)
        print('{:d} -> {:s}'.format(marker_id, key))
        return self.select_by_key(key)

    def select_by_key(self, key):
        items = self.result_list.findItems(key, Qt.MatchStartsWith)
        if len(items) > 1:
            raise ValueError("List contained multiple entries with same key")
        elif len(items) < 1:
            raise ValueError("No such key in item list")
        else:
            item = items[0]
            self.result_list.setCurrentItem(item)



class QueryImageDialog(w.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.setModal(True)
        self.setWindowTitle("Select Query Image")

        self.image_path = None
        self.image = ImageWithROI(self)
        self.image.setMinimumSize(QSize(512, 512))
        open_button = w.QPushButton("Select file")
        open_button.clicked.connect(self.on_open)

        instructions = w.QLabel("Open an image and use the mouse to select a region of interest.")

        bb = w.QDialogButtonBox(w.QDialogButtonBox.Ok | w.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)

        vbox = w.QVBoxLayout()
        vbox.addWidget(instructions)
        vbox.addWidget(open_button)
        vbox.addWidget(self.image)
        vbox.addWidget(bb)

        self.setLayout(vbox)

    def on_open(self):
        path, *_ = w.QFileDialog.getOpenFileName()
        if not path:
            return

        self.image_path = path
        try:
            image = cv2.imread(self.image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.image.set_array(image)
        except (OSError, cv2.error):
            w.QErrorMessage(self).showMessage("Failed to open image file")

    def accept(self):
        try:
            image, roi = self.image.get_image_and_roi()
        except ValueError:
            w.QErrorMessage(self).showMessage("No image selected!")
            return False

        if self.image.get_rubberband_rect() is None:
            w.QErrorMessage(self).showMessage("No region of interest set!")
            return False

        print('All OK')
        return super().accept()


class SearchThread(QThread):
    finished = pyqtSignal(list)
    progress_update = pyqtSignal(int, str) # Percentage, text

    def __init__(self, database, image, roi, image_path, parent=None):
        super().__init__(parent)
        self.database = database
        self.image = image
        self.image_path = image_path
        self.roi = roi

    def run(self):
        sift_file = sift_file_for_image(self.image_path)
        if os.path.exists(sift_file):
            print('Loading SIFT features from', sift_file)
            self.progress_update.emit(10, "Loading SIFT features")
            descriptors, keypoints = load_descriptors_and_keypoints(sift_file)
            descriptors, keypoints = filter_roi(descriptors, keypoints, self.roi)
        else:
            print('Calculating SIFT features in patch')
            self.progress_update.emit(10, "Calculating SIFT features")
            descriptors, keypoints = calculate_sift(self.image, roi=self.roi)

        print('num keypoints:', len(keypoints))
        print('descriptor shape:', descriptors.shape)
        print('Searching...')

        self.progress_update.emit(30, "Searching database")

        matches = self.database.query(descriptors)
        max_matches = 10

        self.progress_update.emit(100, "Finished")
        self.finished.emit(matches[:max_matches])


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
        #self._geofile = LineFileChooser(filter='*.csv')
        #self._geofile.setMinimumWidth(width)

        bb = w.QDialogButtonBox(w.QDialogButtonBox.Ok | w.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)

        form = w.QFormLayout()
        form.addRow("Visual database", self._db_path)
        form.addRow("Image root", self._image_root)
        #form.addRow("Location file", self._geofile)
        form.addWidget(bb)

        self.setLayout(form)

    @property
    def db_path(self):
        return self._db_path.line.text()

    @property
    def image_root(self):
        return self._image_root.line.text()

    #@property
    #def geofile(self):
    #    return None


if __name__ == "__main__":
    app = w.QApplication(sys.argv)
    mwin = MainWindow()

    def on_load():
        mwin.database_page.load_database('/home/hannes/Projects/VS-imsearch/db_sift_2k.h5', '/home/hannes/Datasets/narrative2')

    from PyQt5.QtCore import QTimer
    timer = QTimer(mwin)
    timer.setSingleShot(True)
    timer.timeout.connect(on_load)
    #timer.start(2000)

    sys.exit(app.exec_())