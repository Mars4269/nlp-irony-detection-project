from typing import List, Literal, Optional
import numpy as np
import spacy
from spacy.tokens import Doc
import pandas as pd
from vocabulary import BaseTokenizer
from nltk.tokenize import TweetTokenizer

class PosTagEnrichmentEngine:
    """Manages offline POS tagging on a DataFrame, adding columns with tags and one-hot encodings."""

    def __init__(self, size: Literal["sm", "md", "lg"] = "md",  tokenizer: Optional[BaseTokenizer | TweetTokenizer] = None):
        """Loads a spaCy model for tagging, sets tokenizer, and defines universal/fine-grained mappings."""

        # SpaCy setup
        self.tokenizer = tokenizer
        self.nlp = self._load_nlp(size)

        # Universal tags (they are the same for all languages)
        self.universal_tags = [
            "ADJ", "ADP", "ADV", "AUX", "CONJ", "CCONJ", "DET",
            "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN",
            "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE"
        ]
        self.universal_idx = {tag: i for i, tag in enumerate(self.universal_tags)}

        # Italian tags (language-specific tags present in all)
        self.italian_tags = [
            "A", "AP", "B", "BN", "B_PC", "CC", "CS", "DD", "DE", "DI", "DQ",
            "DR", "E", "E_RD", "FB", "FC", "FF", "FS", "I", "N", "NO", "PART",
            "PC", "PC_PC", "PD", "PE", "PI", "PP", "PQ", "PR", "RD", "RI", "S",
            "SP", "SW", "SYM", "T", "V", "VA", "VA_PC", "VM", "VM_PC",
            "VM_PC_PC", "V_B", "V_PC", "V_PC_PC", "X", "_SP"
        ]
        self.italian_idx = {tag: i for i, tag in enumerate(self.italian_tags)}

        # Dataset placeholder
        self.dataset = None
        self.is_tokenized = False

    def load_dataset(self, df: pd.DataFrame):
        """Stores the dataset for subsequent POS enrichment."""

        # Check presence of text column
        assert "text" in df.columns, "Missing 'text' column"

        # Check that if tokenizer is not passed, text is already tokenized
        if not self.tokenizer and isinstance(df["text"].iloc[0], str):
            raise ValueError("Text is detected to be not tokenized. Please use set_tokenizer() to pass one.")
        if self.tokenizer and isinstance(df["text"].iloc[0], list):
            print("WARNING: Text is detected to be tokenized. Passed tokenizer won't be used.")
        
        # checks passed
        self.dataset = df.copy(deep=True)
        self.is_tokenized = isinstance(self.dataset["text"].iloc[0], list)
        return self

    def set_tokenizer(self, tokenizer: BaseTokenizer | TweetTokenizer):
        self.tokenizer = tokenizer

    def enrich(self, use_universal_tags=False):
        """Adds string_tags and ohe_tags columns to the dataset."""

        # Add the string tags, the int tags and the one-hot tags for each tweet
        self.dataset[["string_tags", "int_tags", "ohe_tags"]] = self.dataset.apply(
            lambda row: pd.Series(self._add_tags_for_tweet(row, use_universal_tags)),
            axis=1
        )

        return self.dataset

    def _add_tags_for_tweet(self, row, use_universal_tags):
        """Generates tag strings and one-hot vectors for a single row's text."""

        # Tokenize with given tokenizer and instantiate doc with the tokens
        tokens = self.tokenizer.tokenize(row["text"]) if not self.is_tokenized else row["text"]
        doc = Doc(self.nlp.vocab, words=tokens)
        self.nlp(doc)
        tag_strs, tag_ints, tag_vectors = [], [], []

        # Get tags for all tokens
        for token in doc:
            # String tag
            tag = token.pos_ if use_universal_tags else token.tag_
            tag_strs.append(tag)

            # Int and One-hot tag
            idx_of_tag, vec = self._one_hot_encode(tag, use_universal_tags=use_universal_tags)
            tag_ints.append(idx_of_tag)
            tag_vectors.append(vec)
        
        tag_ints = np.array(tag_ints)
        tag_vectors = np.array(tag_vectors)

        return [tag_strs, tag_ints, tag_vectors]
    
    def _load_nlp(self, size):
        assert size in ["sm", "md", "lg"], 'size must be one among: "sm", "md", "lg".'
        try:
            nlp = spacy.load(f"it_core_news_{size}", disable=["parser", "ner", "attribute_ruler", "lemmatizer"])
            return nlp
        except OSError:
            raise ValueError(
                f"Model it_core_news_{size} not found. Please run: python -m spacy download it_core_news_{size}"
            )
        
    def _one_hot_encode(self, tag: str, use_universal_tags: bool):
        # Choose lookup dict based on the passed boolean
        if use_universal_tags:
            idx = self.universal_idx
        else:
            if tag not in self.italian_idx:
                self.italian_idx[tag] = len(self.italian_idx)
            idx = self.italian_idx

        # Instantiate ohe vector as vector of zeros, e.g. vec = [0,0,0,0,0]
        vec = np.array([0] * len(idx))

        # Find index of tag from the lookup and set vec[idx] to 1
        # e.g. idx["B"] = 2 => vec = [0,0,1,0,0]
        if tag in idx:
            idx_of_tag = idx[tag]
            vec[idx_of_tag] = 1

        return idx_of_tag, vec
        
