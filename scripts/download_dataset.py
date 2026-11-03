"""Unduh snapshot MetroPT-3 resmi dan periksa SHA-256 sebelum ekstraksi."""
from pathlib import Path
import argparse
import hashlib
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
URL = "https://archive.ics.uci.edu/static/public/791/metropt%2B3%2Bdataset.zip"
EXPECTED_ZIP = "aab991a970e58210de853bb8078ce0e63abb4d9412fdc5c79792dae3d8e1721a"
EXPECTED_CSV = "db30ccb4ea402e3c8bf2c99db06e288d4f2a772f6928f9dbe26a920d69793e24"

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Pakai ZIP lokal sebagai pengganti unduhan.")
    args = parser.parse_args()
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    archive = data / "metropt+3+dataset.zip"
    if args.source:
        source = args.source.expanduser().resolve()
        if digest(source) != EXPECTED_ZIP:
            raise SystemExit("SHA-256 ZIP lokal berbeda dari snapshot eksperimen. Periksa sumber dataset.")
        if source != archive:
            shutil.copyfile(source, archive)
    if not archive.exists():
        partial = archive.with_suffix(".zip.part")
        request = urllib.request.Request(URL, headers={"User-Agent":"MetroPT3-B10/1.0"})
        print("Mengunduh dataset resmi sekitar 208 MiB ...", flush=True)
        try:
            with urllib.request.urlopen(request, timeout=180) as response, partial.open("wb") as dest:
                shutil.copyfileobj(response, dest, length=4*1024*1024)
            if digest(partial) != EXPECTED_ZIP:
                raise ValueError("Snapshot UCI berbeda dari checksum yang dicatat. Periksa perubahan sumber.")
            partial.replace(archive)
        finally:
            partial.unlink(missing_ok=True)
    if digest(archive) != EXPECTED_ZIP:
        raise SystemExit("Checksum arsip tidak cocok; jangan melanjutkan dengan sumber yang berbeda.")
    with zipfile.ZipFile(archive) as stored:
        for name in ["MetroPT3(AirCompressor).csv", "Data Description_Metro.pdf"]:
            target = data / ("Data_Description_Metro.pdf" if name.endswith(".pdf") else name)
            if name.endswith(".csv") and target.exists() and digest(target) == EXPECTED_CSV:
                continue
            with stored.open(name) as src, target.open("wb") as dest:
                shutil.copyfileobj(src, dest, length=4*1024*1024)
    if digest(data / "MetroPT3(AirCompressor).csv") != EXPECTED_CSV:
        raise SystemExit("Checksum CSV hasil ekstraksi tidak cocok.")
    print("Dataset lengkap tersedia dan checksum cocok:", data)

if __name__ == "__main__":
    main()
