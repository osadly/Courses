# Compute spatial lag with TEMPORAL LAG to avoid data leakage

print("\nComputing spatial lag features with temporal lag (this may take a few minutes)...")

import multiprocessing as mp
from functools import partial

def get_neighbor_fires_lagged(row, df, neighbor_map):
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

    neighbor_data = df[
        (df['CTUID'].isin(neighbors)) &
        (df['year_month'] == previous_period)
    ]

    return neighbor_data['number_of_fires'].sum()

def get_neighbor_targets_lagged(row, df, neighbor_map):
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

    neighbor_data = df[
        (df['CTUID'].isin(neighbors)) &
        (df['year_month'] == previous_period)
    ]

    return neighbor_data['Target_Variable'].sum()

def get_neighbor_count(row, neighbor_map):
    """Number of valid neighbors for this cell (to identify border cells)"""
    cell_ctuid = row['CTUID']
    neighbors = neighbor_map.get(cell_ctuid, [])
    return len(neighbors)

# Parallel processing
def process_fires_chunk(rows, df, neighbor_map):
    return [get_neighbor_fires_lagged(row, df, neighbor_map) for _, row in rows.iterrows()]

def process_targets_chunk(rows, df, neighbor_map):
    return [get_neighbor_targets_lagged(row, df, neighbor_map) for _, row in rows.iterrows()]

def process_count_chunk(rows, neighbor_map):
    return [get_neighbor_count(row, neighbor_map) for _, row in rows.iterrows()]

# Spatial lag computation with parallel processing
print("Computing spatial lag features...")

# Determine number of cores to use (leave one free)
n_cores = max(1, mp.cpu_count() - 1)
print(f"Using {n_cores} cores for parallel processing...")

# Split dataframe into chunks
chunk_size = len(final_df) // n_cores
chunks = [final_df.iloc[i:i + chunk_size] for i in range(0, len(final_df), chunk_size)]

# Process in parallel
with mp.Pool(processes=n_cores) as pool:
    # Neighbors fires lag
    fires_results = pool.starmap(process_fires_chunk, 
                                  [(chunk, final_df, neighbor_map) for chunk in chunks])
    final_df['neighbors_fires_lag1m'] = [item for sublist in fires_results for item in sublist]
    
    # Neighbors target lag
    targets_results = pool.starmap(process_targets_chunk,
                                    [(chunk, final_df, neighbor_map) for chunk in chunks])
    final_df['neighbors_target_lag1m'] = [item for sublist in targets_results for item in sublist]
    
    # Neighbor count
    count_results = pool.starmap(process_count_chunk,
                                  [(chunk, neighbor_map) for chunk in chunks])
    final_df['neighbor_count'] = [item for sublist in count_results for item in sublist]

# Normalized version (average per neighbor to handle edge effects)
final_df['neighbors_fires_avg'] = final_df['neighbors_fires_lag1m'] / final_df['neighbor_count']
final_df['neighbors_fires_avg'] = final_df['neighbors_fires_avg'].fillna(0)

print("\nSpatial lag features created:")
print("   neighbors_fires_lag1m (sum of fires in adjacent cells from previous month)")
print("   neighbors_target_lag1m (count of adjacent cells with fires from previous month)")
print("   neighbor_count (number of valid neighbors - identifies border cells)")
print("   neighbors_fires_avg (average fires per neighbor - normalized for edge effects)")

print(f"\nSpatial lag statistics:")
print(final_df[['neighbors_fires_lag1m', 'neighbors_target_lag1m', 'neighbor_count', 'neighbors_fires_avg']].describe())