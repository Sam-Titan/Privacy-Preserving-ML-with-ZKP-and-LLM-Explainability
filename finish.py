import os
import ezkl
import asyncio
import inspect
import json

os.environ["HOME"] = os.environ.get("USERPROFILE", "C:\\")
os.environ["XDG_DATA_HOME"] = os.environ.get("APPDATA", "C:\\")
os.environ["XDG_CONFIG_HOME"] = os.environ.get("LOCALAPPDATA", "C:\\")

with open("outputs/zkp/settings.json", "r") as f:
    settings = json.load(f)
logrows = settings["run_args"]["logrows"]

circuit_path = "outputs/zkp/model.compiled"
vk_path = "outputs/zkp/vk.key"
pk_path = "outputs/zkp/pk.key"
srs_path = f"outputs/zkp/kzg{logrows}.srs"

print(f"⚙️ Generating Keys using {srs_path}...")

try:
    res = ezkl.setup(model=circuit_path, vk_path=vk_path, pk_path=pk_path, srs_path=srs_path)
    if inspect.isawaitable(res):
        asyncio.run(res)
    print("🎉 Setup is officially 100% complete!")
except Exception as e:
    print(f"❌ Error: {e}")