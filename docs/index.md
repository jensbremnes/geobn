# geobn

Pixel-wise Bayesian network inference over geospatial data.

geobn connects a Bayesian network (BN) to geographic data sources and runs inference
at every pixel of a raster grid. Each pixel is an independent set of evidence. The
result is a posterior distribution over the states of each query node, plus a Shannon
entropy map.

```python
import geobn

bn = geobn.load("avalanche_risk.bif")

bn.set_input("slope_angle", geobn.WCSSource(
    url="https://hoydedata.no/arcgis/services/las_dtm_somlos/ImageServer/WCSServer",
    layer="las_dtm", version="1.0.0", valid_range=(-500, 9000),
))  # Kartverket DTM via WCS
bn.set_input("recent_snow", geobn.ConstantSource(30.0))    # 30 cm snowfall
bn.set_input("temperature", geobn.ConstantSource(-5.0))    # −5 °C

bn.set_discretization("slope_angle", [0, 25, 40, 90], ["gentle", "steep", "extreme"])
bn.set_discretization("recent_snow", [0, 10, 25, 150], ["light", "moderate", "heavy"])
bn.set_discretization("temperature", [-40, -8, -2, 15], ["cold", "moderate", "warming"])

result = bn.infer(query=["avalanche_risk"])
result.show_map()          # interactive Leaflet map
result.to_geotiff("out/")  # multi-band GeoTIFF per query node
```

## Features

- Any Python data source can feed a BN node: rasters, WCS services, point APIs or
  constants.
- Reprojection and resampling onto the common grid use only numpy and pyproj.
- NoData pixels are left out of inference and stay NaN in the outputs.
- Pixels are grouped by evidence combination. When there are many combinations, they
  are all solved in one joint query, so networks with many evidence nodes stay fast.
- For repeated runs you can freeze static inputs such as terrain, or precompute a
  lookup table so that inference becomes array indexing.

## Next steps

- [Installation](installation.md)
- [How it works](concepts.md): the pipeline and the main concepts
- [Lyngen Alps example](examples/lyngen_alps.md): avalanche risk over Norwegian terrain
- [API Reference](api/network.md)
