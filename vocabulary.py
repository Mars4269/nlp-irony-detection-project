from typing import List, Union, Optional, Literal
from gensim.models.keyedvectors import KeyedVectors
from numpy import float32, zeros
from pandas import DataFrame
from sklearn.decomposition import PCA
from numpy.random import uniform
from nltk.tokenize import word_tokenize, TweetTokenizer
from nltk import download
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import json

# Download the tokenizer models (run this once)
# download('punkt')
# download('punkt_tab')

# I am creating a wrapper of word_tokenize to have it in the same form of TweetTokenizer
class BaseTokenizer:
    @staticmethod
    def tokenize(text, language="italian", preserve_line=False):
        return word_tokenize(text, language, preserve_line)

class Vocabulary:
    def __init__(
        self,
        dataset: DataFrame,
        embedding_model: KeyedVectors,
        tokenizer: Union[BaseTokenizer, TweetTokenizer],
        embedding_size: int,
        cache_path: Optional[str] = None,
        cache_mode: Optional[Literal["dump", "load"]] = None
    ) -> None:
        self.embedding_model = embedding_model
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.embedding_size = embedding_size

        self.word_to_idx = None
        self.idx_to_word = None
        self.embedding_matrix = None
        self.vocab_size = None
        self.training_words = None

        self.number_of_unique_tokens = None
        self.number_of_new_terms = None
        self.new_terms = None

        self.cache_path = cache_path
        self.cache_mode = cache_mode

    def create_vocabulary(
        self,
        verbose: int = False
    ):
        """
        Creates a vocabulary and embedding matrix for an Italian tweet dataset.
        If cache_mode == "load" and cache_path is not None, tries to load from cache directly.
        If cache_mode == "dump" and cache_path is not None, after creation, dumps to cache.
        If cache_mode is None, no caching is performed.
        """

        # Check caching logic
        if self.cache_mode == "load" and self.cache_path is not None:
            # Attempt to load from cache
            if self.load_vocabulary(self.cache_path, verbose=verbose):
                # If loading was successful, we can return immediately
                if verbose: print("")
                return
            else:
                if verbose: print("Failed to load from cache. Proceeding without cached values...")

        # If we're here, we need to actually build the vocabulary
        if verbose: print("No cached vocabulary loaded. Building vocabulary from scratch...")

        num_steps = (
            6 if self.cache_mode == "dump" and self.cache_path is not None
            else 5
        )

        # STEP 1
        if verbose: print(f"[1/{num_steps}] Tokenizing...")
        dataset_tokenized = self.dataset.copy(deep=True)
        dataset_tokenized['text'] = dataset_tokenized['text'].apply(lambda x: self.tokenizer.tokenize(x))

        # STEP 2
        if verbose: print(f"[2/{num_steps}] Checking for new terms...")
        self.training_words = self._retrieve_words(dataset_tokenized)
        self.number_of_unique_tokens = len(self.training_words)
        self._check_new_terms(self.training_words)
        self.number_of_new_terms = len(self.new_terms)
        new_percentage = float(len(self.new_terms)) * 100 / len(self.training_words)
        if verbose: print(f"\tTotal new terms: {len(self.new_terms)} ({new_percentage:.2f}%)")

        # STEP 3
        if verbose: print(f"[3/{num_steps}] Merging new terms...")
        self._union_training_embedding_model()

        # STEP 4
        if verbose: print(f"[4/{num_steps}] Getting word_to_idx and idx_to_word...")
        self.word_to_idx = self.embedding_model.key_to_index
        self.idx_to_word = self.embedding_model.index_to_key
        self.vocab_size = len(self.word_to_idx) - 1

        # STEP 5        
        if verbose: print(f"[5/{num_steps}] Building embedding matrix...")
        self.build_embedding_matrix()

        # STEP 6 - if dumping is requested
        if self.cache_mode == "dump" and self.cache_path is not None:
            if verbose: print(f"[6/{num_steps}] Saving in cache path...")
            self.dump_vocabulary(self.cache_path)

        if verbose: print("Done!")

    def dump_vocabulary(self, cache_path: str, verbose: bool = False) -> None:
        """
        Dump the current vocabulary state and embedding model to the specified cache path.
        """
        os.makedirs(cache_path, exist_ok=True)

        # Save embedding model
        extended_embedding_model_path = os.path.join(cache_path, "extended_embedding_model.kv")
        self.embedding_model.save(extended_embedding_model_path)

        # If we have an embedding_matrix, save it as a .pt file
        embedding_matrix_path = None
        if self.embedding_matrix is not None:
            embedding_matrix_path = "embedding_matrix.pt"
            full_embedding_matrix_path = os.path.join(cache_path, embedding_matrix_path)
            torch.save(self.embedding_matrix, full_embedding_matrix_path)
            if verbose: print(f"Embedding matrix saved to {full_embedding_matrix_path}")

        # Save JSON
        vocab_json_path = os.path.join(cache_path, "vocab.json")
        self._to_json(vocab_json_path, "extended_embedding_model.kv", embedding_matrix_path)
        if verbose: print(f"Vocabulary metadata saved to {vocab_json_path}")

    def load_vocabulary(self, cache_path: str, verbose: bool) -> bool:
        """
        Load the vocabulary state and embedding model from the specified cache path.
        Returns True if successful, False otherwise.
        """
        vocab_json_path = os.path.join(cache_path, "vocab.json")
        if not os.path.exists(vocab_json_path):
            if verbose: print(f"File not found: {vocab_json_path}")
            return False

        try:
            with open(vocab_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            tokenizer_name = data.get("tokenizer_name")
            if tokenizer_name == "BaseTokenizer":
                self.tokenizer = BaseTokenizer()
            elif tokenizer_name == "TweetTokenizer":
                self.tokenizer = TweetTokenizer()
            else:
                raise ValueError(f"Unknown tokenizer name: {tokenizer_name}")

            self.embedding_size = data.get("embedding_size")
            self.word_to_idx = data.get("word_to_idx")
            self.idx_to_word = data.get("idx_to_word")
            self.vocab_size = data.get("vocab_size")
            self.number_of_unique_tokens = data.get("number_of_unique_tokens")
            self.number_of_new_terms = data.get("number_of_new_terms")
            self.training_words = data.get("training_words")
            new_terms = data.get("new_terms")
            self.new_terms = set(new_terms) if new_terms is not None else None

            extended_embedding_model_path = data.get("extended_embedding_model_path")
            if extended_embedding_model_path is None:
                raise ValueError("No 'extended_embedding_model_path' found in the JSON.")

            # Load embedding model
            full_model_path = os.path.join(cache_path, extended_embedding_model_path)
            if not os.path.exists(full_model_path):
                raise FileNotFoundError(f"Embedding model file not found: {full_model_path}")

            self.embedding_model = KeyedVectors.load(full_model_path, mmap='r')

            # Load embedding matrix if present
            embedding_matrix_path = data.get("embedding_matrix_path")
            if embedding_matrix_path is not None:
                full_embedding_matrix_path = os.path.join(cache_path, embedding_matrix_path)
                if not os.path.exists(full_embedding_matrix_path):
                    raise FileNotFoundError(f"Embedding matrix file not found: {full_embedding_matrix_path}")

                self.embedding_matrix = torch.load(full_embedding_matrix_path)
                if verbose: print("Embedding matrix successfully loaded from cache!")

            if verbose: print("Vocabulary and embedding model successfully loaded from cache!")
            return True

        except FileNotFoundError as e:
            if verbose: print(f"File not found: {e}")
        except json.JSONDecodeError as e:
            if verbose: print(f"Error decoding JSON: {e}")
        except Exception as e:
            if verbose: print(f"An error occurred while loading vocabulary: {e}")

        return False

    def _to_json(self, path: str, extended_embedding_model_path: str, embedding_matrix_path: Optional[str]) -> None:
        """
        Saves the vocabulary's relevant internal state as a JSON file.
        Includes the path to the extended embedding model and embedding matrix.
        """
        data = {
            "tokenizer_name": type(self.tokenizer).__name__,
            "embedding_size": self.embedding_size,
            "vocab_size": self.vocab_size,
            "number_of_new_terms": self.number_of_new_terms,
            "number_of_unique_tokens": self.number_of_unique_tokens,
            "extended_embedding_model_path": extended_embedding_model_path,
            "embedding_matrix_path": embedding_matrix_path,
            "new_terms": list(self.new_terms) if self.new_terms is not None else None,
            "training_words": self.training_words,
            "word_to_idx": self.word_to_idx,
            "idx_to_word": self.idx_to_word
            # We no longer store embedding_matrix directly in JSON
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def build_embedding_matrix(self):
        assert self.vocab_size and self.embedding_size, "Vocabulary missing: run create_vocabulary first"
        embedding_matrix = zeros((self.vocab_size + 1, self.embedding_size), dtype=float32)

        for word, idx in self.word_to_idx.items():
            if idx != 0:
                embedding_matrix[idx] = self.embedding_model[word]  # leaving the first row for pad tokens

        # turn np array into a torch tensor
        self.embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)

    def _union_training_embedding_model(self):
        # Bulk addition of new terms using add_vectors (if available)
        new_terms_list = list(self.new_terms)
        new_vectors = uniform(low=-0.05, high=0.05, size=(len(new_terms_list), self.embedding_size))
        self.embedding_model.add_vectors(new_terms_list, new_vectors)

        static_oov_vector = uniform(low=-0.05, high=0.05, size=(1, self.embedding_size))
        self.embedding_model.add_vectors(['static_oov'], static_oov_vector)
        static_start_of_hashtag = uniform(low=-0.05, high=0.05, size=(1, self.embedding_size))
        static_end_of_hashtag = uniform(low=-0.05, high=0.05, size=(1, self.embedding_size))
        self.embedding_model.add_vectors(['static_start_of_hashtag'], static_start_of_hashtag)
        self.embedding_model.add_vectors(['static_end_of_hashtag'], static_end_of_hashtag)

    def _retrieve_words(self, dataset_tokenized) -> List[str]:
        return list(set([token for sentence in dataset_tokenized.text for token in sentence]))

    def _check_new_terms(self, word_listing: List[str]) -> None:
        embedding_vocabulary = set(self.embedding_model.key_to_index.keys())
        self.new_terms = set(word_listing).difference(embedding_vocabulary)

    ############################################################################
    #                                PLOT METHODS                               #
    ############################################################################

    def plot_tokens_and_neighbors(self, tokens: List[str], k: int = 5) -> None:
        """
        Plots a 2D PCA projection of the given tokens and their k nearest neighbors 
        according to the current embedding model. Each token's group (the token itself 
        plus its neighbors) is plotted in a distinct color, and all points are annotated 
        with their corresponding token text.

        :param tokens: A list of token strings to explore (e.g., ["ciao", "amore", "pizza"]).
        :param k: Number of nearest neighbors to retrieve for each token.
        """
        # 1. Gather k neighbors for each token
        neighbors_dict = self._get_neighbors_for_tokens(tokens, k)

        # 2. Prepare data for plotting (embeddings, labels, color indices)
        all_tokens_list, embeddings, labels, color_indices = self._prepare_plot_data(tokens, neighbors_dict)

        # 3. Reduce embeddings to 2D via PCA
        pca = PCA(n_components=2)
        points_2d = pca.fit_transform(embeddings)

        # 4. Plot in a scatter diagram
        self._plot_2d_points(points_2d, labels, color_indices, tokens, k)


    def _get_neighbors_for_tokens(self, tokens: List[str], k: int):
        """
        Retrieves the top-k most similar words for each token in `tokens`,
        if they exist in the extended embedding model. Returns a dictionary:
            {
              token: [(neighbor1, sim_score1), (neighbor2, sim_score2), ...],
              ...
            }
        """
        neighbors_dict = {}
        for t in tokens:
            if t in self.embedding_model.key_to_index:
                # Gensim's most_similar returns (neighbor_token, similarity_score)
                neighbors_dict[t] = self.embedding_model.most_similar(t, topn=k)
            else:
                neighbors_dict[t] = []  # No neighbors if token not in vocabulary
        return neighbors_dict

    def _prepare_plot_data(self, tokens: List[str], neighbors_dict: dict):
        """
        Prepares data needed for the PCA plot:
         - Consolidates tokens and neighbors into one list
         - Extracts their embedding vectors
         - Assigns each token (and its neighbors) to a 'color group'
        Returns:
         - all_tokens_list (List[str]): The ordered list of all tokens to plot
         - embeddings (np.ndarray): The stacked embeddings for all tokens
         - labels (List[str]): The same token strings for annotation
         - color_indices (List[int]): A color index for each token
        """
        # 1. Collect all tokens to plot
        all_tokens = set(tokens)
        for t in tokens:
            for neighbor, _ in neighbors_dict[t]:
                all_tokens.add(neighbor)

        all_tokens_list = list(all_tokens)
        
        # 2. Build a dictionary that maps each token (and its neighbors) 
        #    to the color index of its "parent" token.
        token2color = {}
        # We want each token in `tokens` to have a unique color index
        for idx, t in enumerate(tokens):
            # Assign color idx to the main token
            token2color[t] = idx
            # Assign same color to its neighbors
            for neighbor, _ in neighbors_dict[t]:
                token2color[neighbor] = idx

        # 3. Extract embeddings and compute color indices
        embeddings = []
        color_indices = []
        labels = []

        for t in all_tokens_list:
            # If there's no assignment, default to -1 (or some fallback)
            c_idx = token2color[t] if t in token2color else -1
            color_indices.append(c_idx)
            labels.append(t)
            embeddings.append(self.embedding_model[t])  # vector from Gensim

        embeddings = np.array(embeddings)

        return all_tokens_list, embeddings, labels, color_indices

    def _plot_2d_points(self, points_2d, labels, color_indices, tokens, k):
        """
        Given 2D points (points_2d), labels, and color indices, creates a scatter plot.
        Each (token + neighbors) group is assigned a unique color. Points are annotated 
        with their corresponding tokens.
        """
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.figure(figsize=(10, 8))

        # We'll define a color map. If tokens are fewer than 10, tab10 suffices.
        # If more, you might want to pick a bigger palette or dynamically generate colors.
        color_map = plt.cm.get_cmap("tab10", len(tokens))

        for i, (x, y) in enumerate(points_2d):
            c_idx = color_indices[i]
            # If c_idx is invalid, we'll just default to gray
            if c_idx < 0 or c_idx >= len(tokens):
                color = "gray"
            else:
                color = color_map(c_idx)

            # Plot the point
            plt.scatter(x, y, color=color, alpha=0.7, edgecolors='k', s=80)
            # Annotate with the token string
            plt.text(x + 0.02, y + 0.02, labels[i], fontsize=9)

        plt.title(f"PCA (2D) of Tokens and their {k} Neighbors", fontsize=14)
        plt.xlabel("PC1", fontsize=12)
        plt.ylabel("PC2", fontsize=12)
        plt.tight_layout()
        plt.show()

    ############################################################################
    #                          COMPATIBILITY CHECK METHODS                      #
    ############################################################################

    def check_coverage(self) -> None:
        """
        Prints the coverage of dataset tokens in the embedding model, using
        number_of_unique_tokens and number_of_new_terms. Helps confirm
        how many tokens were already known vs. how many were OOV.
        """
        total_unique_tokens = self.number_of_unique_tokens or 0
        new_terms_count = self.number_of_new_terms or 0

        known_tokens_count = total_unique_tokens - new_terms_count
        coverage_ratio = 0.0
        if total_unique_tokens > 0:
            coverage_ratio = (known_tokens_count / total_unique_tokens) * 100.0

        print("----- Coverage Check -----")
        print(f"Total unique tokens in dataset: {total_unique_tokens}")
        print(f"Number of OOV (new) terms:      {new_terms_count}")
        print(f"Number of known tokens:         {known_tokens_count}")
        print(f"Coverage ratio:                 {coverage_ratio:.2f}%\n")

    def check_semantics(self, words: List[str]) -> None:
        """
        Shows the top-5 most similar words to each word in `words`, as a quick
        qualitative check to confirm that the embedding model is producing
        sensible neighbors (and thus aligned with your domain + tokenizer).
        """
        print("----- Semantic Check -----")
        for word in words:
            if word in self.embedding_model.key_to_index:
                print(f"\nMost similar words to '{word}':")
                for neighbor, sim_score in self.embedding_model.most_similar(word, topn=5):
                    print(f"   {neighbor} (score={sim_score:.3f})")
            else:
                print(f"\nWord '{word}' is not in the (extended) embedding model's vocabulary.")
        print("")

    def justify_random_oov(self) -> None:
        """
        Prints a short explanation of why randomly initializing vectors for OOV tokens
        can be an acceptable fallback in many NLP pipelines, highlighting the actual
        OOV ratio in this dataset.
        """
        total_unique_tokens = self.number_of_unique_tokens or 0
        new_terms_count = self.number_of_new_terms or 0

        oov_ratio = 0.0
        if total_unique_tokens > 0:
            oov_ratio = (new_terms_count / total_unique_tokens) * 100.0

        print("----- Justification for Random OOV Handling -----")
        print(f"Out of {total_unique_tokens} total unique tokens, "
              f"{new_terms_count} ({oov_ratio:.2f}%) are new/OOV.\n")
        print("1. Since the OOV percentage is relatively small, assigning random vectors "
              "to these tokens is not likely to drastically impact overall performance.")
        print("2. This approach keeps all dimensionalities consistent and prevents the "
              "model from discarding or ignoring unseen tokens.")
        print("3. If we later fine-tune the model, these random embeddings can adapt "
              "to reflect actual usage of OOV terms.")
        print("4. This balances practicality (no data loss) with a minimal overhead for "
              "a small fraction of unknown words.\n")



