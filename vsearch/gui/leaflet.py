import os

from PyQt5.QtWebKitWidgets import QWebView
from PyQt5.QtCore import QFileInfo, QUrl, pyqtSignal, pyqtSlot

RESOURCE_DIR = os.path.dirname(__file__)
MAP_HTML = os.path.join(RESOURCE_DIR, 'map.html')
MAP_JS = os.path.join(RESOURCE_DIR, 'map.js')

class LeafletMarker:
    def __init__(self, lat, lng):
        self.lat = lat
        self.lng = lng

class LeafletWidget(QWebView):

    onMove = pyqtSignal()
    onClick = pyqtSignal()
    onMarkerClicked = pyqtSignal(int)

    next_marker_id = 0

    def __init__(self, lat, lng, parent=None):
        print('Creating LeafletWidget')
        super().__init__(parent=parent)
        page = self.page()
        self.frame = page.mainFrame()

        self.frame.addToJavaScriptWindowObject("QtWidget", self)
        url = QUrl.fromLocalFile(QFileInfo(MAP_HTML).absoluteFilePath())
        self.load(url)
        self.loadFinished.connect(lambda: self.on_load_finished(lat, lng))

    def on_load_finished(self, lat, lng):
        print('Leaflet loading')
        frame = self.page().mainFrame()

        with open(MAP_JS, 'r') as f:
            frame.evaluateJavaScript(f.read())
        print('Loaded javascript')
        self.setView(lat, lng)

    def map_command(self, map_command_str):
        js_str = 'mymap.{};'.format(map_command_str)
        return self.run_js(js_str)

    def run_js(self, js_str):
        return self.frame.evaluateJavaScript(js_str)

    def setView(self, lat, lng, zoom=17):
        print('setView:', lat, lng, zoom)
        self.map_command('setView([{:f}, {:f}], {:d})'.format(lat, lng, zoom))

    def getCenter(self):
        res = self.map_command('getCenter()')
        return res['lat'], res['lng']

    def getBounds(self):
        res = self.map_command('getBounds()')
        return res['_southWest']['lat'], res['_southWest']['lng'], res['_northEast']['lat'], res['_northEast']['lng']

    def add_marker(self, lat, lng):
        return self.add_markers_bulk([[lat, lng]])[0]

    def add_markers_bulk(self, latlngs):
        import json
        js = "add_markers_bulk({});".format(json.dumps(latlngs))
        return self.run_js(js)

    def clear_marker(self, marker_id):
        js = """marker_dict[{:d}].remove();""".format(marker_id)
        #js = """mymap.removeLayer(marker_dict[{:d}]);""".format(marker_id)
        self.run_js(js)
        self.update()