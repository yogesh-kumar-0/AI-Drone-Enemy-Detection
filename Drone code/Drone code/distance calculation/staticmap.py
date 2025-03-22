import googlemaps

# Set up the Google Maps client
gmaps = googlemaps.Client(key='AIzaSyABIVDl1dq84KbcL1Fa4Ce0j6YX-FZCSbA')

# Set the center, zoom level, and size of the map
center = (37.7749, -122.4194)  # San Francisco
zoom = 12
size = (600, 400)

# Generate the static map image
static_map = gmaps.static_map(center=center, zoom=zoom, size=size)

# Convert the generator object to a bytes-like object
with open('static_map.png', 'wb') as f:
    for chunk in static_map:
        f.write(chunk)