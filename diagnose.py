import os
import asyncio
import inspect
import ezkl

os.environ["RAYON_NUM_THREADS"] = "1"
os.environ["RUST_LOG"] = "info"

witness = "outputs/zkp/witness.json"
model = "outputs/zkp/model.compiled"
pk = "outputs/zkp/pk.key"
proof = "outputs/zkp/proof.json"
srs = "outputs/zkp/kzg12.srs"  # We point exactly to your local SRS file

async def test_prove():
    print("⚙️ Starting isolated proof generation test...")
    try:
        # We now know 5th arg is SRS path! 
        # We will try passing "single" as the 6th arg, or omit it if it fails.
        try:
            res = ezkl.prove(witness, model, pk, proof, srs, "single")
        except TypeError:
            res = ezkl.prove(witness, model, pk, proof, srs)
            
        if inspect.isawaitable(res):
            await res
            
        print("✅ SUCCESS! The proof generated perfectly.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_prove())