"""Ejecuta el notebook con este Python y mide el kernel, no el tamaño joblib."""

import json, sys, threading
from pathlib import Path
import nbformat
import psutil
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "notebooks/05_recomendador.ipynb"
nb = nbformat.read(path, as_version=4)
km = KernelManager(kernel_name="python3")
km.kernel_spec.argv = [
    sys.executable,
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}",
]
client = NotebookClient(
    nb, km=km, timeout=7200, resources={"metadata": {"path": str(ROOT)}}
)
stop = threading.Event()


def monitor():
    peak = 0
    samples = 0
    while not stop.wait(0.5):
        for p in psutil.Process().children(recursive=True):
            try:
                if "ipykernel_launcher" not in " ".join(p.cmdline()):
                    continue
                m = p.memory_info()
                peak = max(peak, getattr(m, "peak_wset", m.rss))
                samples += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if samples:
            result = {
                "kernel_peak_working_set_mib": peak / 1024**2,
                "samples": samples,
                "method": "Muestreo 0.5 s; Windows peak_wset desde creación; RSS muestreada en otros sistemas",
                "scope": "máximo del proceso principal del kernel; no suma de picos de procesos ni RAM de todo el sistema",
            }
            dest = ROOT / "reports/memory.json"
            dest.parent.mkdir(exist_ok=True)
            tmp = dest.with_suffix(".tmp")
            tmp.write_text(json.dumps(result, indent=2), encoding="utf-8")
            tmp.replace(dest)


thread = threading.Thread(target=monitor, daemon=True)
thread.start()
try:
    client.execute()
finally:
    stop.set()
    thread.join(timeout=2)
    nbformat.write(nb, path)
print("Notebook ejecutado y guardado:", path)
