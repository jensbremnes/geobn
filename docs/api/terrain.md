# Terrain

`geobn.terrain` computes terrain derivatives from a source of elevations in metres. Each
function returns a [`DerivedSource`][geobn.DerivedSource], which is passed to
[`set_input()`][geobn.GeoBayesianNetwork.set_input] like any other source:

```python
dem = geobn.WCSSource(
    url="https://hoydedata.no/arcgis/services/las_dtm_somlos/ImageServer/WCSServer",
    layer="las_dtm",
    version="1.0.0",
    valid_range=(-500.0, 9000.0),
    cache_dir="cache/",
)

bn.set_input("slope_angle", geobn.terrain.slope(dem))
bn.set_input("aspect",      geobn.terrain.aspect(dem))
bn.set_input("roughness",   geobn.terrain.roughness(dem, size=5))
bn.set_input("tpi",         geobn.terrain.tpi(dem, size=9))
```

The elevations are aligned to the reference grid before the derivative is taken, so the
result describes the terrain at the grid's resolution. A slope computed on a 200 m grid is
gentler than one computed on the 10 m DTM behind it and then aggregated.

## Slope and aspect

Slope and aspect come from the elevation gradient. The distance between neighbouring pixel
centres is measured along the WGS84 ellipsoid, so the result is correct on a grid in
degrees, where a pixel is narrower east–west than north–south away from the equator, and
on a projected grid, whatever its scale factor. The distances and directions are measured
exactly on a lattice of up to 65 × 65 pixels and interpolated linearly in between.

Aspect is the compass bearing the terrain faces, the direction of steepest descent, in
degrees clockwise from true north. On a grid whose columns do not point east, such as a
UTM zone away from its central meridian, a rotated grid or a polar grid, the gradient is
turned into true east and north components before the bearing is taken.

A pixel next to a NaN pixel uses a one-sided difference instead of the central one. The
edge of the data, such as a coastline in a land DTM, therefore does not appear as a cliff.
A pixel with no valid neighbour along a row or a column is NaN.

Level ground has no aspect. `aspect()` returns NaN there by default. Pass `flat=` to give
such pixels a value of their own, for instance `-1.0`, and a state in the discretization.

## Roughness and TPI

Roughness and the topographic position index work on a square window of `size` × `size`
pixels, and ignore NaN pixels within it. Roughness is the largest elevation difference in
the window. TPI is the pixel's elevation minus the mean of the other pixels in the window:
positive on ridges and hilltops, negative in valleys and hollows. Both follow the
definitions used by GDAL's `gdaldem`.

## Function reference

::: geobn.terrain.slope
    options:
      show_root_heading: true

::: geobn.terrain.aspect
    options:
      show_root_heading: true

::: geobn.terrain.roughness
    options:
      show_root_heading: true

::: geobn.terrain.tpi
    options:
      show_root_heading: true
