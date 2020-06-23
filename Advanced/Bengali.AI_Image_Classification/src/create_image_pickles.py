import pandas as pd
import glob
from tqdm import tqdm
import joblib

if __name__ == "__main__":
    files = glob.glob("../input/train_*.parquet")
    for f in files:
        df = pd.read_parquet(f)
        image_ids = df["image_id"].values
        df = df.drop("image_id", axis=1)
        image_array = df.values
        for j, img_id in tqdm(enumerate(image_ids), total=len(image_ids)):
            joblib.dump(image_array[j, :], f"../input/image_pickles/{img_id}.pkl")
