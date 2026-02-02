import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="ISO-8859-1")
    df = df[["text", "label"]].copy()
    df = df.rename(columns={"text": "raw_text", "label": "Spam Label"})
    return df

def clean_text(text: pd.Series) -> pd.Series:
    text = text.str.lower()
    text = text.str.replace(r"http\S+|www\S+", " ", regex=True)
    return text
