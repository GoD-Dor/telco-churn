from pathlib import Path
import hashlib

source = Path("data/raw/telco.csv")

source_hash = hashlib.sha256(source.read_bytes()).hexdigest()

print("Dataset:", source)
print("SHA-256:", source_hash)