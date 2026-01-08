import os
import json
import torch
import boto3
import s3fs
import numpy as np
from botocore import UNSIGNED
from botocore.client import Config
from torch import nn
from tqdm import tqdm

# -------------------------------
# Configuration
# -------------------------------
BUCKET = "chimera-challenge"
PREFIX = "v2/task3/data/"
LOCAL_OUT_DIR = "./rna_embeddings"
EMBED_DIM = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(LOCAL_OUT_DIR, exist_ok=True)

# -------------------------------
# S3 (anonymous)
# -------------------------------
s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
fs = s3fs.S3FileSystem(anon=True)

# -------------------------------
# RNA Encoder (MLP)
# -------------------------------
class RNAEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_dim)
        )

    def forward(self, x):
        return self.net(x)

# -------------------------------
# Discover patient folders
# -------------------------------
def list_patient_ids():
    objects = s3.list_objects_v2(
        Bucket=BUCKET,
        Prefix=PREFIX,
        Delimiter="/"
    )
    return [
        obj["Prefix"].split("/")[-2]
        for obj in objects.get("CommonPrefixes", [])
    ]

# -------------------------------
# Load RNA JSON from S3
# -------------------------------
def load_rna_json(patient_id):
    path = f"s3://{BUCKET}/{PREFIX}{patient_id}/{patient_id}_RNA.json"
    with fs.open(path, "r") as f:
        return json.load(f)

# -------------------------------
# Main
# -------------------------------
def main():
    patient_ids = list_patient_ids()
    print(f"Found {len(patient_ids)} patients")

    # --- infer gene order from first patient ---
    sample_rna = load_rna_json(patient_ids[0])
    genes = sorted(sample_rna.keys())
    gene_index = {g: i for i, g in enumerate(genes)}
    in_dim = len(genes)

    print(f"RNA gene dimension: {in_dim}")

    # --- init encoder ---
    encoder = RNAEncoder(in_dim, EMBED_DIM).to(DEVICE)
    encoder.eval()  # no training here

    for pid in tqdm(patient_ids):
        try:
            rna_dict = load_rna_json(pid)

            # convert to ordered vector
            rna_vec = np.zeros(in_dim, dtype=np.float32)
            for g, v in rna_dict.items():
                if g in gene_index:
                    rna_vec[gene_index[g]] = v

            rna_tensor = torch.from_numpy(rna_vec).to(DEVICE)

            with torch.no_grad():
                embedding = encoder(rna_tensor).cpu()

            # save as .pt
            torch.save(
                {
                    "patient_id": pid,
                    "embedding": embedding,
                    "genes": genes,
                    "encoder": "MLP_1024x256",
                },
                os.path.join(LOCAL_OUT_DIR, f"{pid}.pt")
            )

        except Exception as e:
            print(f"[ERROR] {pid}: {e}")

    print("Done.")

if __name__ == "__main__":
    main()
