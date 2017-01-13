var mymap = L.map('map');

L.tileLayer('http://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 20,
    attribution: 'Map data &copy; <a href="http://openstreetmap.org">OpenStreetMap</a> contributors, ' +
        '<a href="http://creativecommons.org/licenses/by-sa/2.0/">CC-BY-SA</a>, ' +
        'Imagery © <a href="http://mapbox.com">Mapbox</a>',
    id: 'mapbox.streets'
}).addTo(mymap);

var marker_dict = {};

var add_markers_bulk = function(latlngs) {
    //alert('Add bulk!')
    var ids = [];
    for (i=0; i < latlngs.length; ++i) {
        marker = L.marker(latlngs[i]);
        marker.addTo(mymap);
        marker.on('click', onMarkerClicked);

        var id = L.stamp(marker);
        marker_dict[id] = marker;

        ids.push(id);
    }

    //alert(ids);
    return ids;
}

if(typeof QtWidget != 'undefined') {
    //var onMarkerClicked = function(e) { QtWidget.onMarkerClicked(this.options.marker_id) };
    var onMarkerClicked = function(e) { QtWidget.onMarkerClicked(L.stamp(this)) };
    var onMapMove = function() { QtWidget.onMove() };
    var onMapClick = function(e) { QtWidget.onClick(e.latlng.lat, e.latlng.lng) };

    mymap.on('move', onMapMove);
    mymap.on('click', onMapClick);
}
