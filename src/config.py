from pathlib import Path

# Monomer database for fitting (structure lookup during descriptor calculation)
MONOMER_DB_INPUT = Path("test_data/example_monomer_db.csv")

# Monomer database for prediction/enumeration output
MONO_DB_OUTPUT = Path("test_data/example_monomer_db.csv")

# Note: Both files must have at least 'Partial_SMILES' and 'symbol' columns.
