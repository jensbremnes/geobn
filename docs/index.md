# geobn

**Pixel-wise Bayesian network inference over geospatial data.**

geobn lets you connect a Bayesian network (BN) to geographic data sources and run
posterior-probability inference at every pixel of a raster grid. Each pixel is
treated as an independent evidence vector; the BN returns a full probability
distribution over the query node states, plus Shannon entropy.

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

## Why geobn?

- **No special GIS knowledge required** — wire any Python data source to a BN node.
- **Pure-Python reprojection** — numpy + pyproj only; no rasterio needed for grid alignment.
- **NaN-aware** — NoData pixels are excluded from inference and stay NaN in outputs.
- **Efficient batching** — unique evidence combinations are grouped; one pgmpy query
  per unique combo, not per pixel.
- **Real-time ready** — freeze static inputs (terrain), precompute lookup tables for
  sub-millisecond per-pixel inference in streaming scenarios.

## Next steps

- [Installation](installation.md) — install the right extras for your use case
- [How it works](concepts.md) — pipeline diagram and core concepts
- [Lyngen Alps example](examples/lyngen_alps.md) — avalanche risk over real Norwegian terrain
- [API Reference](api/network.md) — full class and method documentation
