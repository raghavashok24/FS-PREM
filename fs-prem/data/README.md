# Data

Place raw dataset files in `data/raw/`. They are excluded from version control via `.gitignore`.

## Required Files

| File | Source | Description |
|---|---|---|
| `master_port_storm_dataset_portex_encoded.csv` | Internal (IBTrACS + AIS merge) | 120k+ advisory–port observations with pre-computed PORTEX features |

## Download Instructions

### IBTrACS Storm Advisories
- URL: https://www.ncei.noaa.gov/products/international-best-track-archive
- Select North Atlantic basin (NA), 2020–2024
- Download `ibtracs.NA.list.v04r00.csv`

### AIS Vessel Records
- URL: https://hub.marinecadastre.gov/
- Select Gulf of Mexico zone, 2020–2024
- Merge with IBTrACS on timestamp and spatial proximity

## Preprocessing

After placing raw files, run **Notebook 01** (`notebooks/01_data_setup.ipynb`) to:
1. Merge IBTrACS advisories with AIS records
2. Recompute the five PORTEX dimensions
3. Create binary event labels (95th percentile threshold)
4. Save `data/processed/portex_labeled.parquet`
