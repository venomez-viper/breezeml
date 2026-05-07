import pandas as pd

def embed(df: pd.DataFrame, text_columns: list | str, model_name: str = "all-MiniLM-L6-v2") -> pd.DataFrame:
    """
    Convert raw text columns into dense semantic embeddings using sentence-transformers.
    Returns a new DataFrame with the original text columns dropped and replaced by embedding features.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    text_columns : list or str
        The name(s) of the column(s) containing raw text to embed.
    model_name : str
        The sentence-transformers model to use. Default is "all-MiniLM-L6-v2" (fast and highly accurate).

    Returns
    -------
    pd.DataFrame
        A new DataFrame ready for machine learning.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "The 'sentence-transformers' library is required for semantic embeddings.\n"
            "Please install it using: pip install breezeml[nlp]  OR  pip install sentence-transformers"
        )
    
    if isinstance(text_columns, str):
        text_columns = [text_columns]
        
    df_new = df.copy()
    
    # Load model (downloads only once, runs offline thereafter)
    print(f"BreezeML 🌬️ Loading NLP model '{model_name}'...")
    model = SentenceTransformer(model_name)
    
    for col in text_columns:
        if col not in df_new.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
            
        print(f"BreezeML 🌬️ Generating semantic embeddings for column '{col}'...")
        texts = df_new[col].fillna("").astype(str).tolist()
        
        embeddings = model.encode(texts, show_progress_bar=True)
        
        df_new = df_new.drop(columns=[col])
        emb_df = pd.DataFrame(
            embeddings, 
            columns=[f"{col}_emb_{i}" for i in range(embeddings.shape[1])],
            index=df_new.index
        )
        df_new = pd.concat([df_new, emb_df], axis=1)
        
    return df_new
