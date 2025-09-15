# convert_to_json.py

import os
import pickle
import json

# Directory containing your PKL files and where JSON will be written
MODEL_DIR      = "model"
RFC_PKL_PATH   = os.path.join(MODEL_DIR, "rfc.pkl")
PCA_PKL_PATH   = os.path.join(MODEL_DIR, "trained_pca.pkl")
RFC_JSON_PATH  = os.path.join(MODEL_DIR, "rfc.json")
PCA_JSON_PATH  = os.path.join(MODEL_DIR, "trained_pca.json")


def convert_rfc():
    with open(RFC_PKL_PATH, "rb") as f:
        pipeline = pickle.load(f)

    rfc_data = {}
    for digit_key, model in pipeline.items():
        rfc_data[digit_key] = {
            "alpha":        model["alpha"].tolist(),
            "beta":         model["beta"].tolist(),
            "n_components": model["n_components"],
            "degree_n":     model["degree_n"],
            "degree_d":     model["degree_d"]
        }

    with open(RFC_JSON_PATH, "w") as f:
        json.dump(rfc_data, f, indent=2)
    print(f"Generated {RFC_JSON_PATH}")


def convert_pca():
    with open(PCA_PKL_PATH, "rb") as f:
        pca = pickle.load(f)

    pca_data = {
        "components":        pca.components_.tolist(),            # should be [17][784]
        "mean":              pca.mean_.tolist(),                  # [784]
        "explained_variance": pca.explained_variance_ratio_.tolist()  # [17]
    }

    with open(PCA_JSON_PATH, "w") as f:
        json.dump(pca_data, f, indent=2)
    print(f"Generated {PCA_JSON_PATH}")


if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    convert_rfc()
    convert_pca()