import ezkl
import asyncio
import inspect
import os
import json

# Read the actual logrows calculated by setup.py
with open("outputs/zkp/settings.json", "r") as f:
    settings = json.load(f)
logrows = settings["run_args"]["logrows"]

path = os.path.join(os.getcwd(), "outputs", "zkp", f"kzg{logrows}.srs")
os.makedirs(os.path.dirname(path), exist_ok=True)

print(f"⚙️ Generating SRS locally for logrows={logrows}...")
try:
    res = ezkl.gen_srs(srs_path=path, logrows=logrows)
    if inspect.isawaitable(res):
        asyncio.run(res)
    print(f"✅ Success! File generated at: {path}")
except Exception as e:
    print(f"❌ Error generating SRS: {e}")