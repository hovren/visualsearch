var mymap = L.map('map');

L.tileLayer('http://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 20,
    attribution: 'Map data &copy; <a href="http://openstreetmap.org">OpenStreetMap</a> contributors, ' +
        '<a href="http://creativecommons.org/licenses/by-sa/2.0/">CC-BY-SA</a>, ' +
        'Imagery © <a href="http://mapbox.com">Mapbox</a>',
    id: 'mapbox.streets'
}).addTo(mymap);

var marker_dict = {};

if(typeof QtWidget != 'undefined') {
    //var onMarkerClicked = function(e) { QtWidget.onMarkerClicked(this.options.marker_id) };
    var onMarkerClicked = function(e) { QtWidget.onMarkerClicked(L.stamp(this)) };
    var onMarkerMoveEnded = function(e) { QtWidget.onMarkerMoved(L.stamp(this)) };
    var onMapMove = function() { QtWidget.onMove() };
    var onMapClick = function(e) { QtWidget.onClick(e.latlng.lat, e.latlng.lng) };

    mymap.on('move', onMapMove);
    mymap.on('click', onMapClick);
}
