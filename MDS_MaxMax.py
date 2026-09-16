import os
import pandas as pd
import numpy as np
from collections import Counter
from rdkit import Chem
from rdkit.Chem.AtomPairs import Pairs
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from adjustText import adjust_text
from matplotlib.lines import Line2D

# *** Matplotlib Font Settings ***
mpl.rcParams['pdf.fonttype'] = 42     # embed TrueType font → editable text in Illustrator
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'

# ==========================================
# INPUT / OUTPUT CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_CSV_PATH  = os.path.join(BASE_DIR, "SMILES.csv")
OUTPUT_MDS_CSV  = os.path.join(BASE_DIR, "MDS_coordinates_AP.csv")
OUTPUT_PDF      = os.path.join(BASE_DIR, "MDS_scatterplot_MaxMax.pdf")
OUTPUT_DIST_CSV = os.path.join(BASE_DIR, "Dissimilarity_Matrix.csv")

# ==========================================
# EXCLUSION & PICKER CONFIGURATION
# ==========================================

# 2. Picker Exclusion: List antagonists to exclude from MAXMAX PICKING ONLY (they stay on the MDS plot)
MAXMIN_EXCLUDE_COMPOUNDS = []

# --- MaxMax Multi-Seed Settings ---
NUM_TO_PICK         = 5    # Number of compounds picked per seed iteration
NUM_SEEDS           = 1000  # Total seed iterations to run
TOP_N_HIGHLIGHT     = 5     # Number of top recurring picks to highlight on the plot
# ==========================================

# === Step 1. Data Loading & Global Exclusion ===
df = pd.read_csv(INPUT_CSV_PATH)

required_cols = {"SMILES", "Compounds", "Behavior"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"CSV must contain columns: {required_cols}")


mols = [Chem.MolFromSmiles(smi) for smi in df["SMILES"]]
valid = [i for i, m in enumerate(mols) if m is not None]
df = df.iloc[valid].reset_index(drop=True)
mols = [m for m in mols if m is not None]
n = len(mols)
print(f"✅ Loaded {n} valid molecules.")

# === Step 2. Compute binary atom-pair fingerprints ===
def atom_pair_set(mol):
    fp = Pairs.GetAtomPairFingerprint(mol)
    return set(fp.GetNonzeroElements().keys())

ap_sets = [atom_pair_set(m) for m in mols]

# === Step 3. Compute binary Tanimoto similarities ===
def tanimoto_binary(set1, set2):
    inter = len(set1 & set2)
    union = len(set1 | set2)
    return inter / union if union else 0.0

sim_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        sim_matrix[i, j] = tanimoto_binary(ap_sets[i], ap_sets[j])

# === Step 4. Compute Distance Matrix ===
dist_matrix = 1 - sim_matrix

# === Step 4.5. Multi-Seed MaxMax Pick Frequency Loop ===
print(f"Running MaxMax over {NUM_SEEDS} random seeds...")

# Identify valid eligible antagonists (excluding compounds in MAXMIN_EXCLUDE_COMPOUNDS)
eligible_mask = (df["Behavior"] == "Antagonist") & (~df["Compounds"].isin(MAXMIN_EXCLUDE_COMPOUNDS))
eligible_indices = df[eligible_mask].index.tolist()
n_eligible = len(eligible_indices)

pick_size = min(NUM_TO_PICK, n_eligible)
pick_counts = Counter()

if pick_size > 0:
    eligible_dist = dist_matrix[np.ix_(eligible_indices, eligible_indices)]
    
    for seed in range(NUM_SEEDS):
        # Set seed to determine the starting seed molecule randomly
        np.random.seed(seed)
        first_pick = np.random.randint(0, n_eligible)
        local_picks = [first_pick]
        
        # MaxMax Selection Loop
        for _ in range(pick_size - 1):
            unpicked = [i for i in range(n_eligible) if i not in local_picks]
            
            # For each unpicked candidate, find the MAXIMUM distance to ANY already-picked compound
            max_scores = [np.max(eligible_dist[u, local_picks]) for u in unpicked]
            
            # Select candidate with the largest MAX distance
            next_pick = unpicked[np.argmax(max_scores)]
            local_picks.append(next_pick)
            
        pick_counts.update(local_picks)

# Assign selection counts and frequency percentages to DataFrame
df["PickCount"] = 0
df["PickFrequency_%"] = 0.0

for local_idx, count in pick_counts.items():
    global_idx = eligible_indices[local_idx]
    df.loc[global_idx, "PickCount"] = count
    df.loc[global_idx, "PickFrequency_%"] = round((count / NUM_SEEDS) * 100, 2)

# Get top N recurring antagonist indices overall
top_local_picks = [idx for idx, _ in pick_counts.most_common(TOP_N_HIGHLIGHT)]
top_global_indices = [eligible_indices[i] for i in top_local_picks]

print("✅ MaxMax pick frequency tally complete.")
print(df.loc[top_global_indices, ["Compounds", "PickCount", "PickFrequency_%"]])

# === Step 5. Classical MDS ===
def classical_mds(D, n_components=2):
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    eigvals, eigvecs = eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    pos = eigvals > 0
    eigvals, eigvecs = eigvals[pos], eigvecs[:, pos]
    coords = eigvecs[:, :n_components] * np.sqrt(eigvals[:n_components])
    return coords

coords = classical_mds(dist_matrix, n_components=2)

# Save MDS coordinates and statistics to CSV
df_out = df.copy()
df_out["MDS1"] = coords[:, 0]
df_out["MDS2"] = coords[:, 1]

cols_to_save = [c for c in ["Compounds", "SMILES", "Behavior", "PickCount", "PickFrequency_%", "MDS1", "MDS2"] if c in df_out.columns]
df_out[cols_to_save].to_csv(OUTPUT_MDS_CSV, index=False)
print(f"✅ MDS coordinates and pick frequencies saved to: {OUTPUT_MDS_CSV}")


# ==========================================
# PLOTTING
# ==========================================
fig, ax = plt.subplots(figsize=(9, 7))

agonists = df[df["Behavior"] == "Agonist"]
antagonists = df[df["Behavior"] == "Antagonist"]

# --- Agonist markers: TRIANGLES ---
ax.scatter(
    coords[agonists.index, 0],
    coords[agonists.index, 1],
    s=35,
    marker="^",
    facecolors="#339933",
    edgecolors="#339933",
    linewidths=1.4,
    label="Agonist"
)

# --- Antagonist markers: CIRCLES ---
ax.scatter(
    coords[antagonists.index, 0],
    coords[antagonists.index, 1],
    s=35,
    marker="o",
    facecolors="#a92223",
    edgecolors="#a92223",
    linewidths=1.4,
    label="Antagonist"
)

# --- HIGHLIGHT TOP RECURRING ANTAGONISTS WITH GOLD HALOS ---
if top_global_indices:
    freqs = df.loc[top_global_indices, "PickFrequency_%"].values
    ring_sizes = 80 + (freqs * 3)  # Halo size scales directly with selection %

    ax.scatter(
        coords[top_global_indices, 0],
        coords[top_global_indices, 1],
        s=ring_sizes,
        marker="o",
        facecolors="none",
        edgecolors="gold",
        linewidths=2.5,
        label=f"Top {TOP_N_HIGHLIGHT} Recurring Picks"
    )

# --- Add Labels ---
texts = []
top_set = set(top_global_indices)

for i, row in df.iterrows():
    compound_name = row["Compounds"]
    if i in top_set:
        label = f"{compound_name} ({row['PickFrequency_%']}%)"
    else:
        label = compound_name
        
    t = ax.text(coords[i, 0], coords[i, 1], label, fontsize=9)
    texts.append(t)

print("Adjusting text labels to prevent overlap...")
adjust_text(texts, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5))

plt.title(f"Classical MDS (MaxMax Selection Frequency over {NUM_SEEDS} Seeds)", fontsize=13)
plt.xlabel("MDS Dimension 1")
plt.ylabel("MDS Dimension 2")

# Create custom legend
legend_elems = [
    Line2D([0], [0], marker="^", color="w", label="Agonist", markerfacecolor="#339933", markersize=9, markeredgecolor="#339933"),
    Line2D([0], [0], marker="o", color="w", label="Antagonist", markerfacecolor="#a92223", markersize=9, markeredgecolor="#a92223"),
    Line2D([0], [0], marker="o", color="w", label=f"Top {TOP_N_HIGHLIGHT} Recurring Picks", markerfacecolor="none", markersize=11, markeredgecolor="gold", markeredgewidth=2.5)
]
plt.legend(handles=legend_elems, title="Behavior / Selection", loc="best")

plt.tight_layout()

# Save PDF figure
plt.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight")
print(f"✅ PDF saved to: {OUTPUT_PDF}")

plt.show()

# Save dissimilarity matrix
dist_df = pd.DataFrame(dist_matrix, index=df["Compounds"], columns=df["Compounds"])
dist_df.to_csv(OUTPUT_DIST_CSV)
print("✅ Similarity matrix saved successfully.")