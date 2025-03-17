import os
import sys
from typing import List
import torch
import pandas as pd
from torch.nn import functional as F
from oml import datasets as d
from oml.inference import inference
from oml.models import ViTExtractor
from oml.registry import get_transforms_for_pretrained

def create_sample_sub(pair_ids: List[str], sim_scores: List[float]):
    sub_sim_column = "similarity"
    id_column = "pair_id"
    return pd.DataFrame({id_column: pair_ids, sub_sim_column: sim_scores})

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python make_submition.py <model_path>")
        sys.exit(1)

    model_path = sys.argv[1]

    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        sys.exit(1)

    if not os.path.exists("data"):
        os.makedirs("data")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_path = "test.csv"

    model = ViTExtractor(model_path, "vitb16", True).to(device).eval()
    transform, _ = get_transforms_for_pretrained("vitb16_dino")

    df_test = pd.read_csv(test_path)
    test = d.ImageQueryGalleryLabeledDataset(df_test, transform=transform)
    embeddings = inference(model, test, batch_size=32, num_workers=0, verbose=True)

    e1 = embeddings[::2]
    e2 = embeddings[1::2]
    sim_scores = F.cosine_similarity(e1, e2).detach().cpu().numpy()

    pair_ids = df_test["label"].apply(lambda x: f"{x:08d}").to_list()
    pair_ids = pair_ids[::2]

    sub_df = create_sample_sub(pair_ids, sim_scores)
    sub_df.to_csv("data/submission.csv", index=False)

    print("Submission file saved to data/submission.csv")
