import pandas as pd
from collections import Counter
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv("../input/train.csv")
    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df["target"].values
    kf = model_selection.StratifiedKFold(n_splits=5)

    for fold_, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, "kfold"] = fold_
    
    print(df['kfold'].value_counts())

    df.to_csv("../input/train_folds.csv", index=False)


