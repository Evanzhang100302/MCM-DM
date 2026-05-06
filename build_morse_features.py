import math, torch, numpy as np, networkx as nx
from morse_function import (
    build_clique_complex, initialize_vertex_weights,
    construct_discrete_morse_function, identify_critical_simplices
)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--emb_file", required=True)
parser.add_argument("--out_file", required=True)
parser.add_argument("--k", type=int, default=5)
args = parser.parse_args()
EMB_FILE = args.emb_file
OUT_FILE = args.out_file
K = args.k

# Convert tile (zoom, tx, ty) to center (lon, lat)
def tile_to_lonlat(zoom, tx, ty):
    n = 2 ** zoom
    lon = tx / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ty / n)))
    lat = math.degrees(lat_rad)
    return lon, lat

# Step 1: Load tile keys and compute lon/lat from tile coords
emb_cache = torch.load(EMB_FILE, map_location="cpu")
keys = list(emb_cache.keys())
lons = [tile_to_lonlat(k[0], k[1], k[2])[0] for k in keys]
lats = [tile_to_lonlat(k[0], k[1], k[2])[1] for k in keys]
N = len(keys)
print(f"Tiles: {N}")

# Step 2: Haversine distance (km)
def haversine(lon1, lat1, lon2, lat2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2)
    return R * 2 * math.asin(math.sqrt(a))

dist = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if i != j:
            dist[i][j] = haversine(lons[i], lats[i], lons[j], lats[j])

# Step 3: KNN graph
G = nx.Graph()
G.add_nodes_from(range(N))
for i in range(N):
    neighbors = np.argsort(dist[i])
    for j in neighbors[1:K+1]:
        G.add_edge(i, j, weight=1.0/(dist[i][j]+1e-6))
print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Step 4: Morse pipeline
complex_dict = build_clique_complex(G, max_dimension=1)
g = initialize_vertex_weights(G, seed=42)
f, Flag = construct_discrete_morse_function(G, complex_dict, g, max_dimension=1)
IsCritical = identify_critical_simplices(complex_dict, f, max_dimension=1)

# Step 5: Save features
morse_features = {}
for i, key in enumerate(keys):
    simplex = frozenset([i])
    morse_features[key] = {
        "morse_value": f.get(simplex, 0.0),
        "is_critical": int(IsCritical.get(simplex, False))
    }

n_critical = sum(v["is_critical"] for v in morse_features.values())
vals = [v["morse_value"] for v in morse_features.values()]
print(f"Critical nodes: {n_critical}/{N}")
print(f"Morse value range: [{min(vals):.3f}, {max(vals):.3f}]")
torch.save(morse_features, OUT_FILE)
print(f"Saved -> {OUT_FILE}")
