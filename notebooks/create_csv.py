import pandas as pd

df = pd.DataFrame([
    {"filename": "sample1.png", "has_contraband": 0},
    {"filename": "sample8.png", "has_contraband": 1},
    {"filename": "sample2.png", "has_contraband": 0},
])

df.to_csv("data/raw/metadata_sample.csv", index=False)
