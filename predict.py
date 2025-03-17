import torch
import pandas as pd

from oml import datasets as d
from oml.inference import inference
from oml.metrics import calc_retrieval_metrics_rr

from oml.models import ViTExtractor
from oml.registry import get_transforms_for_pretrained
from oml.retrieval import RetrievalResults, AdaptiveThresholding

device = "cuda"


if __name__ == "__main__":
    model = ViTExtractor.from_pretrained("vits16_dino")
    state_dict = torch.load("model.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    transform, _ = get_transforms_for_pretrained("vits16_dino")

    df_test = pd.read_csv("test.csv")
    test = d.ImageQueryGalleryLabeledDataset(df_test, transform=transform)

    def predict():
        embeddings = inference(model, test, batch_size=32, num_workers=6, verbose=True)
        rr = RetrievalResults.from_embeddings(embeddings, test, n_items=10)
        rr = AdaptiveThresholding(n_std=2).process(rr)
        rr.visualize(query_ids=[2, 1], dataset=test, show=True)
        results = calc_retrieval_metrics_rr(rr, map_top_k=(10,), cmc_top_k=(1, 5, 10))

        for metric_name in results.keys():
            for k, v in results[metric_name].items():
                print(f"{metric_name}@{k}: {v.item()}")

    predict()
