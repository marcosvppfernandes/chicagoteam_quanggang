





#################### FOLIUM ####################
################################################

# !pip install folium
# import folium
# from folium import plugins


# chimap = folium.Map(location=[41.878876, -87.635918],
#                     zoom_start = 12,
#                     control_scale=True,
#                    tiles = "OpenStreetMap")

# folium.raster_layers.TileLayer('Open Street Map').add_to(chimap)
# folium.raster_layers.TileLayer('Stamen Toner').add_to(chimap)
# folium.raster_layers.TileLayer('Stamen Watercolor').add_to(chimap)
# folium.raster_layers.TileLayer('CartoDB Positron').add_to(chimap)
# folium.raster_layers.TileLayer('CartoDB Dark_Matter').add_to(chimap)
# folium.raster_layers.TileLayer('Stamen Terrain').add_to(chimap)


# # folium.LayerControl().add_to(chimap)

# minimap = plugins.MiniMap(toggle_display=True)
# chimap.add_child(minimap)

# plugins.Fullscreen(position='topright').add_to(chimap)

# draw = plugins.Draw(export=True)
# draw.add_to(chimap)
# display(chimap)


# zipmap = folium.Map(location=[41.878876, -87.635918],
#                      zoom_start = 10)
# zipcodes = "https://data.cityofchicago.org/api/geospatial/gdcf-axmw?method=export&format=GeoJSON"
# folium.GeoJson(zipcodes, name="Chicago Zipcodes").add_to(zipmap)
# dataheat['LIGHTING_CONDITION'].value_counts()

# dataheat = data.dropna(subset = ['LATITUDE'])

# dataheatdaylight = dataheat[dataheat['LIGHTING_CONDITION'] == 'DAYLIGHT']
# dataheatdarknesslight = dataheat[dataheat['LIGHTING_CONDITION'] == 'DARKNESS, LIGHTED ROAD']
# dataheatdarkness = dataheat[dataheat['LIGHTING_CONDITION'] == 'DARKNESS']
# dataheatunknown = dataheat[dataheat['LIGHTING_CONDITION'] == 'UNKNOWN']
# dataheatdusk = dataheat[dataheat['LIGHTING_CONDITION'] == 'DUSK']
# dataheatdawn = dataheat[dataheat['LIGHTING_CONDITION'] == 'DAWN']


# folium.plugins.HeatMap(list(zip(dataheat['LATITUDE'], dataheat['LONGITUDE'])), radius=2, blur=3).add_to(chimap)
# folium.LayerControl().add_to(chimap)
# display(chimap)


# chimapdaylight = folium.Map(location=[41.878876, -87.635918],
#                     zoom_start = 12,
#                     control_scale=True,
#                    tiles = "OpenStreetMap")

# folium.plugins.HeatMap(list(zip(dataheatdaylight['LATITUDE'], dataheatdaylight['LONGITUDE'])), radius=2, blur=3).add_to(chimapdaylight)
# folium.LayerControl().add_to(chimapdaylight)
# plugins.Fullscreen(position='topright').add_to(chimapdaylight)

# display(chimapdaylight)


# chimapdarknesslight = folium.Map(location=[41.878876, -87.635918],
#                     zoom_start = 12,
#                     control_scale=True,
#                    tiles = "OpenStreetMap")

# folium.plugins.HeatMap(list(zip(dataheatdarknesslight['LATITUDE'], dataheatdarknesslight['LONGITUDE'])), radius=2, blur=3).add_to(chimapdarknesslight)
# folium.LayerControl().add_to(chimapdarknesslight)
# plugins.Fullscreen(position='topright').add_to(chimapdarknesslight)

# display(chimapdarknesslight)


# chimapdarkness = folium.Map(location=[41.878876, -87.635918],
#                     zoom_start = 12,
#                     control_scale=True,
#                    tiles = "OpenStreetMap")

# folium.plugins.HeatMap(list(zip(dataheatdarkness['LATITUDE'], dataheatdarkness['LONGITUDE'])), radius=2, blur=3).add_to(chimapdarkness)
# folium.LayerControl().add_to(chimapdarkness)
# plugins.Fullscreen(position='topright').add_to(chimapdarkness)

# display(chimapdarkness)


# datainj = data[data['INJURIES_TOTAL'] > 0]
# datainjuries = datainj.dropna(subset = ['LATITUDE'])
# datainjuries['LATITUDE'].isna().sum()


# chiinjuries = folium.Map(location=[41.878876, -87.635918],
#                     zoom_start = 12,
#                     control_scale=True,
#                    tiles = "OpenStreetMap")

# folium.plugins.HeatMap(list(zip(datainjuries['LATITUDE'], datainjuries['LONGITUDE'])), radius=2, blur=3).add_to(chiinjuries)
# folium.LayerControl().add_to(chiinjuries)
# plugins.Fullscreen(position='topright').add_to(chiinjuries)

# display(chiinjuries)


# datainjy = data[data['INJURIES_TOTAL'] == 0]
# datainjuriesy = datainjy.dropna(subset = ['LATITUDE'])
# datainjuriesy['LATITUDE'].isna().sum()


# chiinjuriesy = folium.Map(location=[41.878876, -87.635918],
#                     zoom_start = 12,
#                     control_scale=True,
#                    tiles = "OpenStreetMap")

# folium.plugins.HeatMap(list(zip(datainjuriesy['LATITUDE'], datainjuriesy['LONGITUDE'])), radius=2, blur=3).add_to(chiinjuriesy)
# folium.LayerControl().add_to(chiinjuriesy)
# plugins.Fullscreen(position='topright').add_to(chiinjuriesy)

# display(chiinjuriesy)


