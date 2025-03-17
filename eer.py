import pandas as pd
import argparse
import json
import numpy as np

from sklearn.metrics import roc_curve


def compute_eer(y_true, y_score):
    fpr, tpr, threshold = roc_curve(y_true, y_score)

    # заменяем np.inf на max + eps
    eps = 1e-3
    threshold[0] = max(threshold[1:]) + eps

    fnr = 1 - tpr
    eer_index = np.nanargmin(np.absolute((fnr - fpr)))
    eer = fnr[eer_index]
    return eer


def main(private_test_url, private_prediction_url):
    gt_label_column = "label"
    sub_sim_column = "similarity"
    id_column = "pair_id"

    sub_df = pd.read_csv(private_prediction_url)
    gt_df = pd.read_csv(private_test_url)

    gt_df = gt_df.astype({id_column: int})
    sub_df = sub_df.astype({id_column: int})

    gt_df = gt_df.join(sub_df.set_index(id_column), on=id_column, how="left")

    if gt_df[sub_sim_column].isna().any():
        print("Не все `pair_id` присутствуют в сабмите")

    y_score = sub_df[sub_sim_column].tolist()
    y_true = gt_df[gt_label_column].tolist()

    return compute_eer(y_true, y_score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--public_test_url", type=str, required=True)
    parser.add_argument("--public_prediction_url", type=str, required=True)
    parser.add_argument("--private_test_url", type=str, required=False)
    parser.add_argument("--private_prediction_url", type=str, required=False)
    args = parser.parse_args()
    public_score = main(args.public_test_url, args.public_prediction_url)

    private_score = None
    if args.private_test_url and args.private_prediction_url:
        private_score = main(args.private_test_url, args.private_prediction_url)

    print(
        json.dumps(
            {
                "public_score": public_score,
                "private_score": private_score,
            }
        )
    )
