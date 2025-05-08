from pathlib import Path

import cartopy.io.shapereader as shpreader
import duckdb
import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from src.util import create_config, make_cell_geom

BASE = Path("/Users/kyledorman/data/planet_coverage/points_30km/")  # <-- update this
config_file = BASE / "dove" / "config.yaml"  # <-- update this
config = create_config(config_file)

# ------------------------------------------------------------------
#  Parquet file patterns (Dove + SkySat)
# ------------------------------------------------------------------
all_files_pattern_dove = str(BASE / "*/results/*/*/*/*/data.parquet")
all_files_pattern_skysat = str(BASE / "skysat/results/*/*/*/data.parquet")

all_dove = list(BASE.glob("*/results/*/*/*/*/data.parquet"))
all_skysat = list(BASE.glob("skysat/results/*/*/*/data.parquet"))

# Combined list used later when we search individual files
all_parquets = all_dove + all_skysat

# 1. Use Cartopy's Natural Earth admin_1_states_provinces shapefile
shp_path = shpreader.natural_earth(resolution="10m", category="cultural", name="admin_1_states_provinces")
states = gpd.read_file(shp_path)

# 2. Filter to California
ca = states[(states["admin"] == "United States of America") & (states["name"] == "California")]

# 3. Project to CA Albers for buffering by 5 km
ca_proj = ca.to_crs("EPSG:3310").buffer(5e4)

# 6. Convert back to WGS84
ca_wgs = ca_proj.to_crs("EPSG:4326")

orig_geo_df = gpd.read_file(config.grid_path)
gdf = orig_geo_df.to_crs(epsg=4326)

# Build a single CA polygon and filter points by intersection
ca_shape = ca_wgs.union_all(method="coverage")
mask_ca = gdf.geometry.intersects(ca_shape)
gdf = gdf[mask_ca].copy()

# gdf["lon_bin"] = (np.floor(gdf.geometry.x / degree_size) * degree_size).astype(float)
# gdf["lat_bin"] = (np.floor(gdf.geometry.y / degree_size) * degree_size).astype(float)
# gdf["grid_index"] = gdf.index

# ----------------------------------------------------------------------------
# 1) group by cell and collect both points AND their original indices
# ----------------------------------------------------------------------------
grouped = (
    gdf.groupby(["lon_bin", "lat_bin"])
    .agg(
        {
            "geometry": list,  # list of Point geometries
            "grid_index": list,  # list of original indices
        }
    )
    .reset_index()
)
grouped["cell_geom"] = grouped["geometry"].apply(make_cell_geom)  # type: ignore
cells = gpd.GeoDataFrame(grouped, geometry="cell_geom", crs="EPSG:4326")

# --- Connect to DuckDB ---
con = duckdb.connect()

# --- Find first grid with a point in California ---
search_idx = gdf.index[0]
for file in all_parquets:
    result = con.execute(f"SELECT COUNT(*) FROM read_parquet('{file}') WHERE grid_idx = {search_idx}").fetchone()
    if result is None:
        continue
    result = result[0]

    if result > 0:
        print(f"Found grid_idx {search_idx} in: {file}")
        single_file = file
        break

# Register a view for all files
con.execute(
    f"""
    CREATE OR REPLACE VIEW samples_all AS
    SELECT * FROM read_parquet('{all_files_pattern_dove}')
"""
)

# Register a view for a single file for faster iteration
con.execute(
    f"""
    CREATE OR REPLACE VIEW samples_one AS
    SELECT * FROM '{single_file}'
"""
)

# --- Schema Inspection ---
print("Schema of samples_one:")
print(con.execute("DESCRIBE samples_one").fetchdf())

# --- Preview Data ---
df_preview = con.execute("SELECT * FROM samples_one LIMIT 10").fetchdf()
print(df_preview.head())

# --- Count Rows ---
print("Total rows in sample file:")
print(con.execute("SELECT COUNT(*) FROM samples_one").fetchone()[0])  # type: ignore


# # --- NULL Check ---
# print("Checking for NULL values:")
# df_nulls = con.execute(
#     """
#     SELECT
#         SUM(CASE WHEN has_8_channel IS NULL THEN 1 ELSE 0 END) AS null_has_8_channel,
#         SUM(CASE WHEN id IS NULL THEN 1 ELSE 0 END) AS null_id,
#         SUM(CASE WHEN acquired IS NULL THEN 1 ELSE 0 END) AS null_acquired,
#         SUM(CASE WHEN clear_percent IS NULL THEN 1 ELSE 0 END) AS null_clear_percent,
#         SUM(CASE WHEN item_type IS NULL THEN 1 ELSE 0 END) AS null_item_type,
#         SUM(CASE WHEN quality_category IS NULL THEN 1 ELSE 0 END) AS null_quality_category,
#         SUM(CASE WHEN satellite_azimuth IS NULL THEN 1 ELSE 0 END) AS null_satellite_azimuth,
#         SUM(CASE WHEN sun_azimuth IS NULL THEN 1 ELSE 0 END) AS null_sun_azimuth,
#         SUM(CASE WHEN sun_elevation IS NULL THEN 1 ELSE 0 END) AS null_sun_elevation,
#         SUM(CASE WHEN view_angle IS NULL THEN 1 ELSE 0 END) AS null_view_angle,
#         SUM(CASE WHEN instrument IS NULL THEN 1 ELSE 0 END) AS null_instrument,
#         SUM(CASE WHEN grid_idx IS NULL THEN 1 ELSE 0 END) AS null_grid_idx
#     FROM samples_all
# """
# ).fetchdf()
# print(df_nulls)

# print("item_type")
# print(con.execute("SELECT DISTINCT item_type from samples_all").fetchdf())

# print("quality_category")
# print(con.execute("SELECT DISTINCT quality_category from samples_all").fetchdf())

# print("instrument")
# print(con.execute("SELECT DISTINCT instrument from samples_all").fetchdf())

print("Pct test")
df_fraction_test = con.execute(
    """
    SELECT grid_idx,
           SUM(quality_category = 'test')::DOUBLE  / COUNT(*) AS frac_test
    FROM samples_all
    GROUP BY grid_idx
"""
).fetchdf()

geo_frac = gdf.join(df_fraction_test.set_index("grid_idx"), how="left").fillna({"frac_test": 1.0})

# Set up the figure
fig, ax = plt.subplots(figsize=(7, 14))

# Optional: Plot world basemap
# world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
# world.plot(ax=ax, color='lightgray', edgecolor='white')

# Plot California as a light gray fill with a dark border
ca_wgs.plot(ax=ax, facecolor="lightgray", edgecolor="black", linewidth=0.5)


# Plot points with frac_8_channel shading
geo_frac.plot(
    ax=ax,
    column="frac_test",
    cmap="plasma",
    markersize=1,
    legend=True,
    vmin=0.0,  # Force lower bound
    vmax=1.0,  # Force upper bound
    legend_kwds={"label": "Fraction of Test Observations", "shrink": 0.7, "orientation": "vertical"},
)

ax.set_title("Fraction of Test Observations per Grid Point", fontsize=14)
ax.set_axis_off()
plt.tight_layout()
plt.savefig("plots/fraction_test.png")

# -------------------------------------------------------------------
#  Count of NON‑test observations per grid point and plot
# -------------------------------------------------------------------
print("Count non‑test")
df_count_nontest = con.execute(
    """
    SELECT
        grid_idx,
        SUM(quality_category <> 'test') AS n_non_test
    FROM samples_all
    GROUP BY grid_idx
"""
).fetchdf()

geo_count = gdf.join(df_count_nontest.set_index("grid_idx"), how="left").fillna({"n_non_test": 0})

# Set up the figure
fig2, ax2 = plt.subplots(figsize=(7, 14))

# Plot California as a light gray fill with a dark border
ca_wgs.plot(ax=ax2, facecolor="lightgray", edgecolor="black", linewidth=0.5)

geo_count.plot(
    ax=ax2,
    column="n_non_test",
    cmap="viridis",
    markersize=1,
    legend=True,
    vmin=0.0,
    legend_kwds={"label": "Count of Non‑test Observations", "shrink": 0.7, "orientation": "vertical"},
)

ax2.set_title("Count of Non‑test Observations per Grid Point", fontsize=14)
ax2.set_axis_off()
plt.tight_layout()
plt.savefig("plots/count_nontest.png")

# -------------------------------------------------------------------
#  First month when ≥50 % of non‑test scenes have 8‑channel
# -------------------------------------------------------------------
print("Compute first‑month >50 % 8‑channel")

df_first_month = con.execute(
    """
WITH monthly AS (
    SELECT
        grid_idx,
        date_trunc('month', acquired) AS month,
        COALESCE(
            SUM(CASE WHEN has_8_channel AND quality_category <> 'test' THEN 1 ELSE 0 END)::DOUBLE
            /
            NULLIF(SUM(CASE WHEN quality_category <> 'test' THEN 1 ELSE 0 END), 0),
            0
        ) AS pct_8
    FROM samples_all
    GROUP BY grid_idx, month
),
over50 AS (
    SELECT grid_idx, month
    FROM monthly
    WHERE pct_8 > 0.5
)
SELECT
    grid_idx,
    MIN(month) AS first_month_over50
FROM over50
GROUP BY grid_idx
"""
).fetchdf()

#
# Join with point geometry (CA only)
geo_first = gdf.join(df_first_month.set_index("grid_idx"), how="inner")
# # Ensure only CA points
# geo_first = geo_first[geo_first.geometry.intersects(ca_shape)]

# Convert to datetime and then to Matplotlib date numbers
geo_first["first_month_over50"] = pd.to_datetime(geo_first["first_month_over50"])
geo_first["month_num"] = mdates.date2num(geo_first["first_month_over50"])

# Plot
fig3, ax3 = plt.subplots(figsize=(7, 14))

points = geo_first.dropna(subset=["month_num"])

# Plot California as a light gray fill with a dark border
ca_wgs.plot(ax=ax3, facecolor="lightgray", edgecolor="black", linewidth=0.5)

# plot without automatic legend
points.plot(
    ax=ax3,
    column="month_num",
    cmap="plasma",
    markersize=1,
    vmin=points["month_num"].min(),
    vmax=points["month_num"].max(),
)

# build a scalar mappable to add custom colorbar
norm = plt.Normalize(vmin=points["month_num"].min(), vmax=points["month_num"].max())
sm = plt.cm.ScalarMappable(cmap="plasma", norm=norm)
sm.set_array([])

cbar = fig3.colorbar(sm, ax=ax3, shrink=0.7, orientation="vertical")
cbar.set_label("Date first ≥50 % 8‑channel")


# Format the colorbar with Year-Month labels
cbar.ax.yaxis_date()
cbar.ax.yaxis.set_major_locator(mdates.MonthLocator(interval=6))
cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))


ax3.set_title("First Month ≥50 % 8‑channel per Grid Point", fontsize=14)
ax3.set_axis_off()
plt.tight_layout()
plt.savefig("plots/first_month_over50.png")
