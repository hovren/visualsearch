import os

from PyQt5.QtWebKitWidgets import QWebView
from PyQt5.QtCore import QFileInfo, QUrl, pyqtSignal, pyqtSlot

RESOURCE_DIR = os.path.dirname(__file__)
MAP_HTML = os.path.join(RESOURCE_DIR, 'map.html')
MAP_JS = os.path.join(RESOURCE_DIR, 'map.js')

class LeafletMarker:
    def __init__(self, id, widget):
        self.id = id
        self.map_widget = widget

    def __repr__(self):
        return '<LeafletMarker id={:d} widget={}>'.format(self.id, self.map_widget)

    def get_property(self, property):
        js_fmt = "marker_dict[{:d}].{:s};"
        js = js_fmt.format(self.id, property)
        return self.map_widget.run_js(js)

    @property
    def latlng(self):
        res = self.get_property('getLatLng()')
        return res['lat'], res['lng']

    def object_call(self, js_method, return_value=True):
        js = 'marker_dict[{:d}].'.format(self.id) + js_method
        return self.map_widget.run_js(js, return_value=return_value)

    def remove(self, update=True):
        js = "marker_dict[{id:d}].remove(); delete marker_dict[{id:d}];".format(id=self.id)
        self.map_widget.run_js(js)
        self.id = None

        if update:
            self.map_widget.update()

    def setDraggable(self, draggable):
        if draggable:
            self.object_call('dragging.enable();'.format(draggable), return_value=False)
        else:
            self.object_call('dragging.disable();'.format(draggable), return_value=False)

    def setOpacity(self, opacity):
        self.object_call("setOpacity({:f});".format(opacity), return_value=False)

    @classmethod
    def create_js(cls, lat, lng):
        js_fmt = """L.marker([{lat}, {lng}])"""
        return js_fmt.format(lat=lat, lng=lng)

    @classmethod
    def add_to_map(cls, map_widget, latlng_list):
        print(len(latlng_list))
        statements = ["var ids = [];"]
        for lat, lng in latlng_list:
            statements.append("var marker = " + cls.create_js(lat, lng) + ";")
            statements.append("marker.addTo(mymap);")
            statements.append("marker.on('click', onMarkerClicked);")
            statements.append("var id = L.stamp(marker);")
            statements.append("marker_dict[id] = marker;")
            statements.append("ids.push(id);")
        statements.append("ids")

        js = "\n".join(statements)
        marker_ids = map_widget.run_js(js)

        markers = [cls(int(mid), map_widget) for mid in marker_ids]
        map_widget.markers.update({m.id: m for m in markers})

        return markers

class LeafletWidget(QWebView):

    onMove = pyqtSignal()
    onClick = pyqtSignal(float, float)
    onMarkerClicked = pyqtSignal(int)

    next_marker_id = 0

    def __init__(self, lat, lng, parent=None):
        print('Creating LeafletWidget')
        super().__init__(parent=parent)
        page = self.page()
        self.markers = {}
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

    def map_command(self, map_command_str, return_value=True):
        js_str = 'mymap.{};'.format(map_command_str)
        return self.run_js(js_str, return_value=return_value)

    def run_js(self, js_str, return_value=True):
        if return_value:
            return self.frame.evaluateJavaScript(js_str)
        else:
            return self.frame.evaluateJavaScript(js_str + "; null;")

    def setView(self, lat, lng, zoom=17):
        print('setView:', lat, lng, zoom)
        self.map_command('setView([{:f}, {:f}], {:d})'.format(lat, lng, zoom), return_value=False)

    def getCenter(self):
        res = self.map_command('getCenter()')
        return res['lat'], res['lng']

    def getBounds(self):
        res = self.map_command('getBounds()')
        return res['_southWest']['lat'], res['_southWest']['lng'], res['_northEast']['lat'], res['_northEast']['lng']

    def remove_all_markers(self):
        if self.markers:
            print('Removing {:d} markers'.format(len(self.markers)))
            for m in self.markers.values():
                m.remove(update=False)
            self.markers.clear()
            self.update()
        else:
            print('No markers to remove')