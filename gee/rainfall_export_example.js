// Minimal public-input example. Replace the AOI asset before running.
var aoi = ee.FeatureCollection('users/replace_with_your_aoi').geometry();
var start = '2018-01-01';
var end = '2026-09-10';

var rainfall = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
  .filterDate(start, end)
  .select('precipitation');

var table = rainfall.map(function(image) {
  var value = image.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: aoi,
    scale: 5566,
    bestEffort: true,
    maxPixels: 1e8
  }).get('precipitation');
  return ee.Feature(null, {
    date: image.date().format('YYYY-MM-dd'),
    precipitation_mm: value
  });
});

Export.table.toDrive({
  collection: table,
  description: 'daily_rainfall_example',
  fileFormat: 'CSV'
});

