var mymap = L.map('map');

L.tileLayer('http://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 20,
    attribution: 'Map data &copy; <a href="http://openstreetmap.org">OpenStreetMap</a> contributors, ' +
        '<a href="http://creativecommons.org/licenses/by-sa/2.0/">CC-BY-SA</a>, ' +
        'Imagery © <a href="http://mapbox.com">Mapbox</a>',
    id: 'mapbox.streets'
}).addTo(mymap);

if(typeof QtWidget != 'undefined') {
    var onMarkerClicked = function(e) { QtWidget.on_marker_clicked(this.options.marker_id) };

    var onMapMove = function() { QtWidget.onMove() };
    mymap.on('move', onMapMove);
}
