from inf8239_u03.config import MOVIELENS_URL, ROOT
from inf8239_u03.data import download_zip

destination, digest = download_zip(MOVIELENS_URL, ROOT / "data/raw")
print("Destino:", destination)
print("SHA-256:", digest)
