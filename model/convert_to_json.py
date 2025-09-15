import os
import pickle
import json

MODEL_DIR       = "model"
RFC_PKL_PATH    = os.path.join(MODEL_DIR, "rfc.pkl")
PCA_PKL_PATH    = os.path.join(MODEL_DIR, "trained_pca.pkl")
RFC_JSON_PATH   = os.path.join(MODEL_DIR, "rfc.json")
PCA_JSON_PATH   = os.path.join(MODEL_DIR, "trained_pca.json")


def to_list(x):
    # Convert NumPy arrays or lists to plain Python lists
    try:
        return x.tolist()
    except AttributeError:
        return list(x)


def convert_rfc():
    with open(RFC_PKL_PATH, "rb") as f:
        pipeline = pickle.load(f)

    rfc_data = {}
    for digit_key, model in pipeline.items():
        rfc_data[digit_key] = {
            "alpha":        to_list(model["alpha"]),
            "beta":         to_list(model["beta"]),
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
        "components":        to_list(pca.components_),
        "mean":              to_list(pca.mean_),
        "explained_variance": to_list(pca.explained_variance_ratio_)
    }

    with open(PCA_JSON_PATH, "w") as f:
        json.dump(pca_data, f, indent=2)
    print(f"Generated {PCA_JSON_PATH}")


if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    convert_rfc()
    convert_pca()