import numpy as np
from torch.utils.data import Dataset
from nltk.tokenize import TweetTokenizer
from tqdm import tqdm
from text_enrichment.pos_tags import PosTagEnrichmentEngine
from vocabulary import BaseTokenizer, Vocabulary
from typing import Dict, Literal, Optional, Union
from preprocessing import segment_hashtags
import pandas as pd
import torch


class CustomDataset(Dataset):
    def __init__(
            self,
            dataframe,
            tokenizer,
            max_len,
            text_enrichment: bool =False,
            hashtag_segmentation: bool =False,
            pos_tags: Optional[Literal['universal', 'italian']] = None,
            parent_dir: Optional[str] = None
    ):
        self.dataframe = dataframe
        self.targets = dataframe['iro'].values

        self.tokenizer = tokenizer
        self.max_len = max_len

        self.text_enrichment = text_enrichment
        self.pos_tags = None

        if text_enrichment:
            self.prob = dataframe.prob.values

        # Segmenting hashtags if asked
        if hashtag_segmentation:
            parent_dir_path = f"{parent_dir}/" if parent_dir else ""
            with open(f"{parent_dir_path}vocab/ALBERTo_vocab.txt", "r", encoding="utf-8") as f:
                vocab = set(line.strip() for line in f)

            with tqdm(total=len(self.dataframe), desc="Segmenting hashtags") as pbar:
                self.dataframe["text"] = self.dataframe["text"].apply(
                    lambda text: segment_hashtags(text=text, dictionary=vocab, pbar=pbar)
                )

        self.data = dataframe['text'].values

        if pos_tags:
            ptee = PosTagEnrichmentEngine(tokenizer = self.tokenizer)
            use_universal_tags = pos_tags == "universal"
            tokenized_data_with_pt = ptee.load_dataset(self.dataframe).enrich(use_universal_tags=use_universal_tags)
            self.pos_tags = tokenized_data_with_pt['ohe_tags'].values

    def __len__(self): # this is needed for the dataloader class
        # The __len__ function returns the number of samples in our dataset.
        return len(self.targets)

    def __getitem__(self, index):
        # The __getitem__ function loads and returns a sample from the dataset at the given index idx

        text = str(self.data[index])
        #text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            return_token_type_ids=True,

            # anche se è un modo di merda perchè stampa 30milioni di warning, l'alternatica
            # attualmente supportata fa SCHIZZARE in una maniera FASTIDIOSAMENTE VERGONOSA
            # l'utilizzo di RAM. Ci terremo i warning, ma con un modello che funziona. tradeoff!
            max_length=self.max_len,
            truncation=True,

            padding='max_length',
        )

        pos_tags = self.pos_tags[index] if self.pos_tags is not None else None

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
        
        out = {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float),
        }

        if pos_tags is not None:
            pos_tags = self._pad_and_truncate_2d(pos_tags)
            out['pos_tags'] = torch.tensor(pos_tags, dtype=torch.long)


        if self.text_enrichment:
            out['probs'] = torch.tensor(self.prob[index], dtype=torch.float)

        return out
    
    def _pad_and_truncate_2d(self, array):
        if array.shape[0] > self.max_len:
            array = array[:self.max_len]
        else:
            padded_zeros = (self.max_len - array.shape[0], array.shape[1])
            array = np.concatenate([array, np.zeros(padded_zeros)])
        return array
    

class GRUDataset(Dataset):
    """
    Custom Dataset for GRU Model using a pre-built vocabulary and embedding matrix.

    :param dataframe: DataFrame containing 'text' and 'iro' columns.
    :type dataframe: pd.DataFrame
    :param word_to_idx: Dictionary mapping words to indices.
    :type word_to_idx: Dict[str, int]
    :param max_len: Maximum length for sequences.
    :type max_len: int
    """
    def __init__(
            self,
            dataframe: pd.DataFrame,
            vocabulary: Vocabulary,
            max_len: int,
            tokenizer: Union[TweetTokenizer, BaseTokenizer],
            hashtag_segmentation: bool = False,
            text_enrichment: bool = False,
            pos_tags: Optional[Literal['universal', 'italian']] = None
        ):

        self.dataframe = dataframe
        #self.data = dataframe['text']
        self.targets = dataframe['iro'].values
        self.text_enrichment = text_enrichment
        self.pos_tags = None

        if text_enrichment:
            self.prob = dataframe.prob.values
        
        self.word_to_idx = vocabulary.word_to_idx
        self.training_words = vocabulary.training_words
        self.max_len = max_len
        self.tokenizer = tokenizer

        # Tokenizing and converting text to indices offline, so that we avoid being slow at runtime
        tokenized_dataframe = pd.DataFrame(self.dataframe["text"].apply(lambda x: tokenizer.tokenize(x)), columns=["text"])
        
        # Segmenting hashtags if asked
        if hashtag_segmentation:
            tokenized_dataframe["text"] = tokenized_dataframe["text"].apply(
                lambda tokens: segment_hashtags(tokens=tokens, dictionary=self.training_words)
            )


        # Mapping tokens to their indices
        tokenized_text = tokenized_dataframe["text"]
        self.token_indices = [
            [self.word_to_idx.get(token, self.word_to_idx.get('static_oov', 0)) for token in sentence]
            for sentence in tokenized_text
            ]
        
        # Getting pos tags if asked
        if pos_tags:
            ptee = PosTagEnrichmentEngine()
            use_universal_tags = pos_tags == "universal"
            tokenized_data_with_pt = ptee.load_dataset(tokenized_dataframe).enrich(use_universal_tags=use_universal_tags)
            self.pos_tags = tokenized_data_with_pt['ohe_tags'].values


    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.targets)

    def __getitem__(self, index):
        """
        Returns a single sample from the dataset.

        :param index: Index of the sample to fetch.
        :type index: int
        :return: Dictionary with 'inputs' (padded token indices) and 'targets'.
        :rtype: Dict[str, torch.Tensor]
        """
        tokenized_and_encoded_inputs = self.token_indices[index]
        pos_tags = self.pos_tags[index] if self.pos_tags is not None else None
        target = self.targets[index]

        # Pad or truncate to max_len
        tokenized_and_encoded_inputs = self._pad_and_truncate_1d(tokenized_and_encoded_inputs)

        if pos_tags is not None:
            pos_tags = self._pad_and_truncate_2d(pos_tags)


        # Format output based on requested enrichment
        out = {
            'inputs': torch.tensor(tokenized_and_encoded_inputs, dtype=torch.long),
            'targets': torch.tensor(target, dtype=torch.float)
        }

        if self.text_enrichment:
            out['probs'] = torch.tensor(self.prob[index], dtype=torch.float)

        if self.pos_tags is not None:
            out['pos_tags'] = torch.tensor(pos_tags, dtype=torch.long)
        
        return out
    
    def _pad_and_truncate_1d(self, array):
        if len(array) > self.max_len:
            array = array[:self.max_len]
        else:
            array += [0] * (self.max_len - len(array))  # Padding with index 0
        return array
    
    def _pad_and_truncate_2d(self, array):
        if array.shape[0] > self.max_len:
            array = array[:self.max_len]
        else:
            padded_zeros = (self.max_len - array.shape[0], array.shape[1])
            array = np.concatenate([array, np.zeros(padded_zeros)])
        return array

        
if __name__ == "__main__":
    from utils import load_datasets
    import transformers

    train, _, _ = load_datasets("data", 42)
    tokenizer = transformers.AutoTokenizer.from_pretrained("m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0")

    cd = CustomDataset(
            train,
            tokenizer,
            50,
            pos_tags = "italian"
    )

    print(cd[0])

