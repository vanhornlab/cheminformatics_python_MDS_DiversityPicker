# Classical MDS with MaxMin and MaxMax Diversity Analysis

**Developed by Matthew Derrick and Wade Van Horn, 2026**

This repository provides a Python pipeline for visualizing chemical space. It converts SMILES strings into binary Atom-Pair fingerprints, calculates a Tanimoto dissimilarity matrix, and applies **Classical Multidimensional Scaling (MDS)** to map molecules into a 2D coordinate system. MaxMin or MaxMax algorithms applied to the Tanimoto dissimilarity matrix map which ligands are maximally diverse or maximally dissimilar, respectively. MaxMin will maximize the *minimum* distance to any already picked compound, while MaxMax will maximize the *maximum* distance to any already picked compound. 

## Features
* **Fingerprinting:** Uses RDKit Atom-Pair fingerprints for structural comparison.
* **Classical MDS:** Implements metric scaling via eigendecomposition.
* **Automated Path Recognition:** Uses os.path to ensure operating system functionality and file path recognition. 
* **Illustrator Ready:** Generates a vector PDF with embedded TrueType fonts, allowing for direct text editing in Adobe Illustrator. Utilizes adjustText for label overlap.
* **MaxMinPicker Script:** Performs the MaxMin algorithm on the dissimilarity matrix. Since there is a single-run bias associated with the random seed selection, the algorithm is run n times (n=1000), and the top 5 most frequent ligands are highlighted on the scatterplot.
* **MaxMaxPicker Script:** MaxMaxPicker is run identically to the MaxMinPicker; however, the algorithm is different. Instead of finding diversity, MaxMax will find the highest dissimilarity.

## Input Requirements
The script expects a file named `SMILES.csv` (or your specified path) containing the following columns:
* `SMILES`: The molecular structure.
* `Compounds`: The name or identifier for the molecule.
* `Behavior`: Category used for plotting (e.g., "Agonist" or "Antagonist").

## File Structure & Outputs

**MaxMinPicker**
| File | Description |
| :--- | :--- |
| **MDS_MaxMin.py** | The main Python script for MaxMin. |
| **Dissimalarity_Matrix.csv** | The $N \times N$ Tanimoto distance matrix ($1 - \text{similarity}$). |
| **MDS_coordinates.csv** | CSV containing original data plus 2D MDS coordinates (`MDS1`, `MDS2`). |
| **MDS_scatterplot_MaxMin.pdf** | A high-resolution scatterplot of the chemical space. |

**MaxMaxPicker**
| File | Description |
| :--- | :--- |
| **MDS_MaxMax.py** | The main Python script for MaxMax. |
| **Dissimalarity_Matrix.csv** | The $N \times N$ Tanimoto distance matrix ($1 - \text{similarity}$). |
| **MDS_coordinates.csv** | CSV containing original data plus 2D MDS coordinates (`MDS1`, `MDS2`). |
| **MDS_scatterplot_MaxMax.pdf** | A high-resolution scatterplot of the chemical space. |

Note: The input requirements, dissimilarity matrix, and MDS coordinates will be identical for both scripts. The only differences will be which ligands are highlighted and the python scripts. 

## Installation & Dependencies
Ensure you have a Python environment (Conda is recommended) with the following libraries installed:
```bash
pip install pandas numpy matplotlib rdkit scipy adjustText
