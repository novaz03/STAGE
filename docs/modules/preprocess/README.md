# `evacmob.preprocess`

Preprocessing utilities extracted from the DRIVES and SES notebooks that prepare mobility data for modelling.

- `normalize_columns` – standardise numeric features to zero mean/unit variance.
- `make_points_gdf` – build a GeoDataFrame of trip start/end points with cleaned timestamps.
- `stitch_trips_to_lines_with_gaps` – reproduce the trip-segmentation logic, returning both segments and inter-trip links.
- `nearest_pois_for_links` – find the closest POIs to segment links using a vectorised haversine.
- `iterative_impute` – wrap scikit-learn’s `IterativeImputer` for column-wise gap filling.
