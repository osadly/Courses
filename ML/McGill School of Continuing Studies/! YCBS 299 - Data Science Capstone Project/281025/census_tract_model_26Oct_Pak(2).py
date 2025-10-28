import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_score, recall_score, f1_score,
    accuracy_score, balanced_accuracy_score
)
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

pwd
from google.colab import drive
drive.mount('/content/drive')
get_ipython().run_line_magic('cd', '"/content/drive/MyDrive/Capstone Project- McGill/Data/"')
print(f"Working directory: {os.getcwd()}")
get_ipython().system('ls')

shp_path = 'census/census_ct_cleaned.shp'
census_df = gpd.read_file(shp_path)
print(census_df.head())
print(census_df.columns)
census_df = census_df.rename(columns = {'Population': 'Population_density', 'Populatio2': 'Population_2021'})
census_df
census_df.plot()

fires = pd.read_csv('sim_combined_df.csv', engine='python')
print(f"Total fire incidents: {len(fires):,}")
fires.head()
fires.rename(columns={'LATITUDE': 'latitude', 'LONGITUDE': 'longitude'}, inplace=True)
fires_geo = gpd.GeoDataFrame(
    fires,
    geometry=gpd.points_from_xy(fires['longitude'], fires['latitude']),
    crs=4326
)
fires_geo = fires_geo.to_crs(epsg=32188)
fires_geo['CREATION_DATE_FIXED'] = pd.to_datetime(fires_geo['CREATION_DATE_FIXED'])
fires_geo['year_month'] = fires_geo['CREATION_DATE_FIXED'].dt.to_period('M').astype(str)

print(f"Date range: {fires_geo['CREATION_DATE_FIXED'].min()} to {fires_geo['CREATION_DATE_FIXED'].max()}")
print(f"Year-month range: {fires_geo['year_month'].min()} to {fires_geo['year_month'].max()}")
fires_geo.plot()

print(census_df.crs)
print(fires_geo.crs)
census_df = census_df.to_crs(epsg=32188)
print(census_df.crs)
print("Joining fires to census...")

census_fire_join2 = gpd.sjoin(
    fires_geo,
    census_df,
    how="left",
    predicate="within"
)
print(f"Fires successfully joined: {census_fire_join2['geometry'].notna().sum():,}")
print(f"Fires without cell assignment: {census_fire_join2['geometry'].isna().sum():,}")
print("Counting fires per cell and month...")

fire_counts2 = (
    census_fire_join2.groupby(['geometry', 'year_month'])
    .size()
    .reset_index(name='fire_count')
)
print(f"Total census-month combinations with fires: {len(fire_counts2):,}")
fire_counts2.tail()
census_fire_join2.geometry
fire_counts2.head()

print("Joining fires to census...")
census_fire_join = gpd.sjoin(
    census_df,
    fires_geo,
    how="left",
    predicate="contains"
)
print(f"Fires successfully joined: {census_fire_join['geometry'].notna().sum():,}")
print(f"Fires without cell assignment: {census_fire_join['geometry'].isna().sum():,}")
print("Counting fires per cell and month...")
fire_counts = (
    census_fire_join.groupby(['geometry', 'year_month'])
    .size()
    .reset_index(name='fire_count')
)
print(f"Total census-month combinations with fires: {len(fire_counts):,}")
fire_counts.iloc[100:200]
fire_counts.shape
census_fire_join.shape
unique_geometries = fire_counts['geometry'].nunique()
unique_year_months = fire_counts['year_month'].nunique()
print(f"Number of unique geometries: {unique_geometries:,}")
print(f"Number of unique year-months: {unique_year_months:,}")
unique_geometries = census_fire_join['geometry'].nunique()
unique_year_months = census_fire_join['year_month'].nunique()
print(f"Number of unique geometries: {unique_geometries:,}")
print(f"Number of unique year-months: {unique_year_months:,}")

buildings = gpd.read_file('montreal_dataset_v1.geojson', engine='pyogrio')
print(f"Initial buildings: {len(buildings):,}")
print(f"Original CRS: {buildings.crs}")
buildings = buildings[buildings['ANNEE_CONS'] != 9999]
print(f"After filtering invalid construction years: {len(buildings):,}")
buildings = buildings.to_crs(epsg=32188)
print(f"Converted to CRS: {buildings.crs}")
print(census_fire_join.crs)
print(buildings.crs)

print("Converting buildings to points...")
buildings_points = buildings.copy()
buildings_points['geometry'] = buildings.representative_point()
print("Joining buildings to grid...")
bld_with_census = gpd.sjoin(
    buildings_points,
    census_df['geometry'].drop_duplicates().reset_index(),
    how="left",
    predicate="within"
)
print(f"Buildings successfully joined: {bld_with_census['geometry'].notna().sum():,}")
print(f"Buildings without geometry assignment: {bld_with_census['geometry'].isna().sum():,}")

print("Aggregating building features by cell...")
bld_agg = (
    bld_with_census.groupby('geometry')
    .agg(
        num_buildings=('geometry', 'count'),
        avg_floors=('ETAGE_HORS', 'mean'),
        median_construction_year=('ANNEE_CONS', 'median')
    )
    .reset_index()
)
print(f"Geom with buildings: {len(bld_agg):,}")
print(f"\nBuilding statistics:")
print(bld_agg.describe())
bld_agg.head()

bld_with_census.geometry
bld_with_census.head()
bld_with_census.geometry

buildings = gpd.read_file('montreal_dataset_v1.geojson', engine='pyogrio')
print(f"Initial buildings: {len(buildings):,}")
print(f"Original CRS: {buildings.crs}")
buildings = buildings[buildings['ANNEE_CONS'] != 9999]
print(f"After filtering invalid construction years: {len(buildings):,}")
buildings = buildings.to_crs(epsg=32188)
print(f"Converted to CRS: {buildings.crs}")
print(census_fire_join.crs)
print(buildings.crs)
print("Converting buildings to points...")
buildings_points = buildings.copy()
buildings_points['geometry'] = buildings.representative_point()
print("Joining buildings to grid...")
bld_with_census = gpd.sjoin(
    census_df['geometry'].drop_duplicates().reset_index(),
    buildings_points,
    how="left",
    predicate="contains"
)
print(f"Buildings successfully joined: {bld_with_census['geometry'].notna().sum():,}")
print(f"Buildings without geometry assignment: {bld_with_census['geometry'].isna().sum():,}")
print("Aggregating building features by cell...")
bld_agg = (
    bld_with_census.groupby('geometry')
    .agg(
        num_buildings=('geometry', 'count'),
        avg_floors=('ETAGE_HORS', 'mean'),
        median_construction_year=('ANNEE_CONS', 'median')
    )
    .reset_index()
)
print(f"Geom with buildings: {len(bld_agg):,}")
print(f"\nBuilding statistics:")
print(bld_agg.describe())
bld_with_census.info()
bld_agg.info()
bld_with_census.head()
bld_agg.head()

print(census_df.columns)
print(fires_geo.columns)
print(buildings.columns)
unique_geometries = bld_agg['geometry'].nunique()
print(f"Number of unique geometries: {unique_geometries:,}")
bld_with_census.describe()
print("Filtering time range...")
fire_counts = fire_counts[fire_counts['year_month'] >= '2005-09']
print(f"After filtering: {len(fire_counts):,} cell-month observations")
print("Creating Panel Dataset...")

pivot_fire = fire_counts.pivot(
    index='geometry',
    columns='year_month',
    values='fire_count'
)
pivot_fire = pivot_fire.fillna(0)
print(f"Cells: {len(pivot_fire)}")
print(f"Months: {len(pivot_fire.columns)}")
print(f"Total observations: {len(pivot_fire) * len(pivot_fire.columns):,}")
final_fire = pivot_fire.stack().reset_index(name="number_of_fires")
print(f"\nPanel Dataset created: {len(final_fire):,} rows")
pivot_fire
final_fire
unique_geometries = final_fire['geometry'].nunique()
unique_year_months = final_fire['year_month'].nunique()
print(f"Number of unique geometries: {unique_geometries:,}")
print(f"Number of unique year-months: {unique_year_months:,}")
bld_agg

print("Merging building features...")
fire_bld = final_fire.merge(bld_agg, on='geometry', how='left')
final_df = fire_bld.merge(census_df, on='geometry', how='left')
final_df['num_buildings'] = final_df['num_buildings'].fillna(0)
final_df['avg_floors'] = final_df['avg_floors'].fillna(final_df['avg_floors'].median())
final_df['median_construction_year'] = final_df['median_construction_year'].fillna(
    final_df['median_construction_year'].median()
)
final_df['year_month'] = pd.PeriodIndex(final_df['year_month'], freq='M')

final_df['Target_Variable'] = (final_df['number_of_fires'] > final_df['number_of_fires'].median()).astype(int)
print(f"\nFinal dataframe shape: {final_df.shape}")
print(f"\nColumns: {list(final_df.columns)}")
print(f"\nTarget variable distribution:")
print(final_df['Target_Variable'].value_counts(normalize=True))
print(f"\nFirst few rows:")
print(final_df.head(10))
final_df.shape
median_fires = final_df['number_of_fires'].median()
print(f"The median of 'number_of_fires' in the final_df is: {median_fires}")
final_df[final_df['CTUID']=='4620001.00'].shape
final_df.isna().sum()
plt.figure(figsize=(8, 6))
sns.countplot(x='Target_Variable', data=final_df, palette='viridis')
plt.title('Distribution of Target_Variable (0: number_of_fires<=2, 1: number_of_fires>2)', fontsize=14, fontweight='bold')
plt.xlabel('Target Variable', fontsize=12)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xticks(ticks=[0, 1], labels=['number_of_fires<=2 (Target_Variable=0)', 'number_of_fires>2 (Target_Variable=1)'])
plt.grid(axis='y', alpha=0.3)
plt.show()

print("Creating temporal lag features...")
final_df = final_df.sort_values(['geometry', 'year_month'])
for lag in [1, 2, 3, 6, 12]:
    final_df[f'fires_lag_{lag}m'] = (
        final_df
        .groupby('geometry')['number_of_fires']
        .shift(lag)
        .fillna(0)
    )
    print(f"Created fires_lag_{lag}m")
for lag in [1, 2, 3]:
    final_df[f'target_lag_{lag}m'] = (
        final_df
        .groupby('geometry')['Target_Variable']
        .shift(lag)
        .fillna(0)
    )
    print(f"Created target_lag_{lag}m")
print(f"\nTemporal lag features added successfully")
print(f"New shape: {final_df.shape}")

print("Creating seasonality features...")
final_df['month'] = final_df['year_month'].dt.month
final_df['year'] = final_df['year_month'].dt.year
final_df['month_sin'] = np.sin(2 * np.pi * final_df['month'] / 12)
final_df['month_cos'] = np.cos(2 * np.pi * final_df['month'] / 12)
print("Seasonality features created:")
print("  - month (1-12)")
print("  - year")
print("  - month_sin (circular encoding)")
print("  - month_cos (circular encoding)")
print("\nSample seasonality values:")
example_months = final_df[['month', 'month_sin', 'month_cos']].drop_duplicates().sort_values('month')
print(example_months.head(12))
print("Building spatial adjacency structure...")
neighbor_map = {}
census_geometries = census_df[['geometry', 'CTUID']].drop_duplicates().set_index('CTUID')
for ctuid, geometry in census_geometries['geometry'].items():
    # Find neighbors that touch the current census tract
    neighbors = census_geometries[census_geometries.geometry.touches(geometry)].index.tolist()
    neighbor_map[ctuid] = neighbors
print(f"Neighbor map created for {len(neighbor_map):,} cells")
neighbor_counts = {cell: len(neighbors) for cell, neighbors in neighbor_map.items()}
print("\nNeighbor count distribution:")
print(f"  Cells with 0 neighbors: {sum(1 for c in neighbor_counts.values() if c == 0)}")
print(f"  Cells with 1 neighbor: {sum(1 for c in neighbor_counts.values() if c == 1)}")
print(f"  Cells with 2 neighbors: {sum(1 for c in neighbor_counts.values() if c == 2)}")
print(f"  Cells with 3 neighbors: {sum(1 for c in neighbor_counts.values() if c == 3)}")
print(f"  Cells with 4 neighbors: {sum(1 for c in neighbor_counts.values() if c == 4)}")
print(f"  Cells with 5 neighbors: {sum(1 for c in neighbor_counts.values() if c == 5)}")
print(f"  Cells with 6 neighbors: {sum(1 for c in neighbor_counts.values() if c == 6)}")
print(f"  Cells with 7 neighbors: {sum(1 for c in neighbor_counts.values() if c == 7)}")
print(f"  Cells with 8 neighbors: {sum(1 for c in neighbor_counts.values() if c == 8)}")
print(f"  Cells with 9 neighbors: {sum(1 for c in neighbor_counts.values() if c == 9)}")
print(f"  Cells with 10 neighbors: {sum(1 for c in neighbor_counts.values() if c == 10)}")
print(f"  Cells with 11 neighbors: {sum(1 for c in neighbor_counts.values() if c == 11)}")
print(f"  Cells with 12 neighbors: {sum(1 for c in neighbor_counts.values() if c == 12)}")
print(f"  Cells with 13 neighbors: {sum(1 for c in neighbor_counts.values() if c == 13)}")
print(f"  Cells with 14 neighbors: {sum(1 for c in neighbor_counts.values() if c == 14)}")
print(f"  Cells with 15 neighbors: {sum(1 for c in neighbor_counts.values() if c == 15)}")
print(f"  Cells with 16 neighbors: {sum(1 for c in neighbor_counts.values() if c == 16)}")
print("\nComputing spatial lag features with temporal lag (this may take a few minutes)...")
def get_neighbor_fires_lagged(row):
    """
    Sum of fire counts in neighboring cells from PREVIOUS month.
    This avoids data leakage by not using concurrent fire information.
    """
    cell_ctuid = row['CTUID']
    current_period = row['year_month']
    neighbors = neighbor_map.get(cell_ctuid, [])
    if len(neighbors) == 0:
        return 0
    # Get previous month (temporal lag)
    current_date = pd.to_datetime(str(current_period))
    previous_period = (current_date - pd.DateOffset(months=1)).to_period('M')
    neighbor_data = final_df[
        (final_df['CTUID'].isin(neighbors)) &
        (final_df['year_month'] == previous_period)
    ]
    return neighbor_data['number_of_fires'].sum()
def get_neighbor_targets_lagged(row):
    """
    Number of neighboring cells with fires in PREVIOUS month.
    This avoids data leakage by not using concurrent fire information.
    """
    cell_ctuid = row['CTUID']
    current_period = row['year_month']
    neighbors = neighbor_map.get(cell_ctuid, [])
    if len(neighbors) == 0:
        return 0
    # Get previous month (temporal lag)
    current_date = pd.to_datetime(str(current_period))
    previous_period = (current_date - pd.DateOffset(months=1)).to_period('M')
    neighbor_data = final_df[
        (final_df['CTUID'].isin(neighbors)) &
        (final_df['year_month'] == previous_period)
    ]
    return neighbor_data['Target_Variable'].sum()
def get_neighbor_count(row):
    """Number of valid neighbors for this cell (to identify border cells)"""
    cell_ctuid = row['CTUID']
    neighbors = neighbor_map.get(cell_ctuid, [])
    return len(neighbors)
print("Computing spatial lag features...")
final_df['neighbors_fires_lag1m'] = final_df.apply(get_neighbor_fires_lagged, axis=1)
final_df['neighbors_target_lag1m'] = final_df.apply(get_neighbor_targets_lagged, axis=1)
final_df['neighbor_count'] = final_df.apply(get_neighbor_count, axis=1)
final_df['neighbors_fires_avg'] = final_df['neighbors_fires_lag1m'] / final_df['neighbor_count']
final_df['neighbors_fires_avg'] = final_df['neighbors_fires_avg'].fillna(0)
print("\nSpatial lag features created:")
print("   neighbors_fires_lag1m (sum of fires in adjacent cells from previous month)")
print("   neighbors_target_lag1m (count of adjacent cells with fires from previous month)")
print("   neighbor_count (number of valid neighbors - identifies border cells)")
print("   neighbors_fires_avg (average fires per neighbor - normalized for edge effects)")
print(f"\nSpatial lag statistics:")
print(final_df[['neighbors_fires_lag1m', 'neighbors_target_lag1m', 'neighbor_count', 'neighbors_fires_avg']].describe())
import sys, os, shutil
RAPIDS_VERSION = '23.10'
get_ipython().system('pip install -q condacolab')
import condacolab
condacolab.install()
get_ipython().system('conda install -c rapidsai -c nvidia -c conda-forge    rapids={RAPIDS_VERSION} python=3.10 cuda-version=11.8 -y')
print("\n--- Installation Complete ---")
print("You may need to manually restart the Colab runtime (Runtime -> Restart Runtime) for cuDF to be fully available.")
import cudf
import pandas as pd # Still needed for date arithmetic, though cuDF is improving in this area
from datetime import timedelta
print("\nConverting data to cuDF for GPU processing...")
gdf = cudf.DataFrame.from_pandas(final_df)
neighbor_list = []
for cell_ctuid, neighbors in neighbor_map.items():
    for neighbor_ctuid in neighbors:
        neighbor_list.append((cell_ctuid, neighbor_ctuid))
neighbor_map_gdf = cudf.DataFrame(neighbor_list, columns=['CTUID', 'neighbor_CTUID'])
print("Data conversion complete. Starting GPU computation...")
period_series = gdf['year_month'].astype(str).to_pandas().apply(lambda x: pd.to_datetime(x).to_period('M'))
previous_period_series = (period_series.apply(lambda x: pd.to_datetime(str(x))) - pd.DateOffset(months=1)).dt.to_period('M')
gdf['previous_period'] = cudf.Series(previous_period_series.astype(str)) # Convert to string for an efficient join key
gdf['year_month_str'] = gdf['year_month'].astype(str) # Convert current period to string for matching
print("Computing spatial lag features with temporal lag...")
lagged_source_gdf = gdf[['CTUID', 'year_month_str', 'number_of_fires', 'Target_Variable']].copy()
lagged_source_gdf.rename(columns={
    'CTUID': 'neighbor_CTUID',
    'year_month_str': 'lagged_year_month_str',
    'number_of_fires': 'neighbor_fires_lagged',
    'Target_Variable': 'neighbor_target_lagged'
}, inplace=True)
joined_neighbors_gdf = gdf.merge(neighbor_map_gdf, on='CTUID', how='left')
final_join_gdf = joined_neighbors_gdf.merge(
    lagged_source_gdf,
    left_on=['neighbor_CTUID', 'previous_period'],
    right_on=['neighbor_CTUID', 'lagged_year_month_str'],
    how='left'
)
spatial_lag_features = final_join_gdf.groupby(['CTUID', 'year_month']).agg({
    'neighbor_fires_lagged': 'sum',
    'neighbor_target_lagged': 'sum',
    'neighbor_CTUID': 'count' # Count of neighbors for the cell-period
}).reset_index()
spatial_lag_features.rename(columns={
    'neighbor_fires_lagged': 'neighbors_fires_lag1m',
    'neighbor_target_lagged': 'neighbors_target_lag1m',
    'neighbor_CTUID': 'neighbor_count'
}, inplace=True)
gdf = gdf.merge(
    spatial_lag_features[['CTUID', 'year_month', 'neighbors_fires_lag1m', 'neighbors_target_lag1m', 'neighbor_count']],
    on=['CTUID', 'year_month'],
    how='left'
)
gdf[['neighbors_fires_lag1m', 'neighbors_target_lag1m', 'neighbor_count']] = \
    gdf[['neighbors_fires_lag1m', 'neighbors_target_lag1m', 'neighbor_count']].fillna(0)
gdf['neighbors_fires_avg'] = gdf['neighbors_fires_lag1m'] / gdf['neighbor_count']
gdf['neighbors_fires_avg'] = gdf['neighbors_fires_avg'].fillna(0)
print("\nSpatial lag features created:")
print("   neighbors_fires_lag1m (sum of fires in adjacent cells from previous month)")
print("   neighbors_target_lag1m (count of adjacent cells with fires from previous month)")
print("   neighbor_count (number of valid neighbors - identifies border cells)")
print("   neighbors_fires_avg (average fires per neighbor - normalized for edge effects)")
print(f"\nSpatial lag statistics:")
print(gdf[['neighbors_fires_lag1m', 'neighbors_target_lag1m', 'neighbor_count', 'neighbors_fires_avg']].describe())
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
ax1 = axes[0, 0]
neighbor_dist = final_df.groupby('cell_id')['neighbor_count'].first()
counts = neighbor_dist.value_counts().sort_index()
ax1.bar(counts.index, counts.values, color=['#e74c3c', '#f39c12', '#3498db', '#2ecc71'])
ax1.set_xlabel('Number of Neighbors', fontsize=12)
ax1.set_ylabel('Number of Cells', fontsize=12)
ax1.set_title('Spatial Grid: Neighbor Count Distribution\n(Shows Edge Effects)', fontsize=14, fontweight='bold')
ax1.set_xticks([2, 3, 4])
for i, v in enumerate(counts.values):
    ax1.text(counts.index[i], v + 0.5, str(v), ha='center', va='bottom', fontsize=11, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
ax2 = axes[0, 1]
sample_data = final_df.sample(min(1000, len(final_df)), random_state=42)
ax2.scatter(sample_data['neighbors_fires_lag1m'], sample_data['number_of_fires'],
           alpha=0.5, s=30, color='#3498db')
ax2.set_xlabel('Neighbor Fires (Previous Month)', fontsize=12)
ax2.set_ylabel('Current Cell Fires', fontsize=12)
ax2.set_title('Spatial-Temporal Relationship\n(Lagged Neighbor Fires vs Current Fires)',
             fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
corr = sample_data[['neighbors_fires_lag1m', 'number_of_fires']].corr().iloc[0, 1]
ax2.text(0.05, 0.95, f'Correlation: {corr:.3f}',
        transform=ax2.transAxes, fontsize=11,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        verticalalignment='top')
ax3 = axes[1, 0]
edge_analysis = final_df.groupby('neighbor_count').agg({
    'number_of_fires': 'mean',
    'neighbors_fires_lag1m': 'mean',
    'neighbors_fires_avg': 'mean'
}).reset_index()
x = range(len(edge_analysis))
width = 0.35
ax3.bar([i - width/2 for i in x], edge_analysis['number_of_fires'],
       width, label='Avg Fires in Cell', color='#e74c3c', alpha=0.8)
ax3.bar([i + width/2 for i in x], edge_analysis['neighbors_fires_avg'].fillna(0) * 4,
       width, label='Avg Neighbor Fires (scaled)', color='#3498db', alpha=0.8)
ax3.set_xlabel('Number of Neighbors', fontsize=12)
ax3.set_ylabel('Average Fire Count', fontsize=12)
ax3.set_title('Fire Risk by Neighbor Count\n(Border vs Interior Cells)',
             fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(edge_analysis['neighbor_count'])
ax3.legend()
ax3.grid(axis='y', alpha=0.3)
ax4 = axes[1, 1]
monthly_spatial = final_df.groupby('year_month').agg({
    'neighbors_fires_lag1m': 'mean',
    'number_of_fires': 'mean'
}).reset_index()
monthly_spatial['date'] = monthly_spatial['year_month'].dt.to_timestamp()
ax4.plot(monthly_spatial['date'], monthly_spatial['number_of_fires'],
        label='Current Cell Fires', linewidth=2, color='#e74c3c')
ax4.plot(monthly_spatial['date'], monthly_spatial['neighbors_fires_lag1m'],
        label='Neighbor Fires (Lag 1m)', linewidth=2, color='#3498db', linestyle='--')
ax4.set_xlabel('Time', fontsize=12)
ax4.set_ylabel('Average Fire Count', fontsize=12)
ax4.set_title('Temporal Evolution: Spatial Lag Features\n(Monthly Averages)',
             fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print("\n" + "="*80)
print("EDGE EFFECTS ANALYSIS")
print("="*80)
print("\nFire statistics by neighbor count:")
edge_stats = final_df.groupby('neighbor_count').agg({
    'number_of_fires': ['mean', 'median', 'std'],
    'neighbors_fires_lag1m': ['mean', 'median'],
    'cell_id': 'nunique'
}).round(2)
edge_stats.columns = ['_'.join(col).strip() for col in edge_stats.columns.values]
print(edge_stats)
print("\nMissing values:")
missing = final_df.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("   No missing values found")
print("\nDuplicate records:")
duplicates = final_df.duplicated(subset=['geometry', 'year_month']).sum()
print(f"   Duplicate rows: {duplicates}")
if duplicates > 0:
    print("   Removing duplicates...")
    final_df = final_df.drop_duplicates(subset=['geometry', 'year_month'])
print("\nTemporal coverage:")
coverage = final_df.groupby('geometry')['year_month'].count()
print(f"   Min observations per cell: {coverage.min()}")
print(f"   Max observations per cell: {coverage.max()}")
print(f"   Mean observations per cell: {coverage.mean():.2f}")
print("\nTarget variable class balance:")
class_dist = final_df['Target_Variable'].value_counts(normalize=True)
print(f"   No fire (0): {class_dist[0]:.2%}")
print(f"   Fire (1): {class_dist[1]:.2%}")
print("\nData types:")
print(final_df.dtypes)
print("\nSummary statistics:")
print(final_df.describe())
print("="*80)
print("FINAL DATASET SUMMARY")
print("="*80)
print(f"\nDataset shape: {final_df.shape[0]:,} rows x {final_df.shape[1]} columns")
print(f"\nSpatial coverage:")
print(f"  - Unique cells: {final_df['geometry'].nunique():,}")
print(f"  - Cell size: {CELL_SIZE}m x {CELL_SIZE}m")
print(f"\nTemporal coverage:")
print(f"  - Start date: {final_df['year_month'].min()}")
print(f"  - End date: {final_df['year_month'].max()}")
print(f"  - Total months: {final_df['year_month'].nunique()}")
print(f"\nTarget variable:")
print(f"  - rows with number_of_fires>7: {final_df['Target_Variable'].sum():,}")
print(f"  - rate of number_of_fires>7: {final_df['Target_Variable'].mean():.2%}")
print(f"\nFeature categories:")
print(f"  Spatial identifiers: cell_id")
print(f"  Temporal identifiers: year_month, year, month")
print(f"  Target: Target_Variable, number_of_fires")
print(f"  Building features: num_buildings, avg_floors, median_construction_year")
print(f"  Temporal lags: fires_lag_1m, fires_lag_2m, fires_lag_3m, fires_lag_6m, fires_lag_12m")
print(f"  Target lags: target_lag_1m, target_lag_2m, target_lag_3m")
print(f"  Seasonality: month_sin, month_cos")
print(f"  Spatial lags (with temporal lag): neighbors_fires_lag1m, neighbors_target_lag1m")
print(f"  Spatial features: neighbor_count, neighbors_fires_avg")
print(f"\nColumn list:")
for i, col in enumerate(final_df.columns, 1):
    print(f"  {i:2d}. {col}")
print("\n" + "="*80)
print("Data preparation complete!")
print("="*80)
final_df_export = final_df.copy()
final_df_export['year_month'] = final_df_export['year_month'].astype(str)
csv_path = 'fire_risk_panel_data_251025.csv'
print(f"Exporting to CSV: {csv_path}")
final_df_export.to_csv(csv_path, index=False)
print(f"CSV export complete. File size: {pd.read_csv(csv_path).memory_usage(deep=True).sum() / 1024**2:.2f} MB")
parquet_path = 'fire_risk_panel_data_251025.parquet'
print(f"\nExporting to Parquet: {parquet_path}")
final_df_export.to_parquet(parquet_path, index=False, compression='snappy', engine='pyarrow')
print(f"Parquet export complete.")
print(f"\nDataset successfully exported to:")
print(f"  {csv_path}")
print(f"  {parquet_path}")
print(f"\nAll done- now ready for model training.")
print("Sample data (first 20 rows):")
print(final_df.head(20))
print("\nSample data (random 10 rows with fires):")
print(final_df[final_df['Target_Variable'] == 1].sample(min(10, len(final_df[final_df['Target_Variable'] == 1]))))