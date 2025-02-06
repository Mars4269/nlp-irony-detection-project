# fundamentals
import pandas as pd
import torch
import os
from typing import Callable, List, Literal, Optional, Tuple, Dict
from abc import ABC, abstractmethod
from itertools import product
from tqdm.notebook import tqdm
from nltk.tokenize import TweetTokenizer
from gensim.models import KeyedVectors

# torch and transformers
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from transformers import AutoTokenizer

# our libs
from CustomDataset import CustomDataset, GRUDataset
from ModelOps import save_loss_plot, bce, test_predict
from metrics import best_threshold, zero_one_f1_score
from training_session import TrainingSession
from utils import load_datasets
from vocabulary import BaseTokenizer, Vocabulary
from preprocessing import fundamental_preprocessing, emoji_and_emoticons_preprocessing


SEED = 42

class GridSearch(ABC):
    """
    Abstract base class for performing grid search.

    :param params: Dictionary of grid search parameters.
    :type params: dict
    :param train_set: The train set to perform the search on.
    :type train_set: pandas.DataFrame
    :param val_set: The validation set to compute the metrics on.
    :type val_set: pandas.dataFrame
    :param device: The device for the computation. It must be either 'cpu' or 'cuda'.
    :param max_len: The max len of an input sequence.
    :type max_len: int
    """
    def __init__(
            self,
            params: Dict[str, List], 
            data_folder: str,
            device: Literal["cpu", "cuda"],
            max_len: Optional[int] = None,
            parent_dir: Optional[str] = None
        ) -> None:

        self.model_class = None
        self.session_names = set()

        # Passable args
        self.params = params
        self.data_folder = data_folder
        self.max_len = max_len
        self.param_dicts = self._generate_param_dicts(params)
        self.num_combinations = len(self.param_dicts)
        self.device = device
        self.parent_dir = parent_dir

        # Default parameters
        self.param_configuration = {
            "loss_fn": bce,
            "lr": 1e-4,
            "batch_size": 8,
            "weight_decay": 0.1,
            "label_smoothing": 0.0,
            "optimizer_class": AdamW,
            "preprocessing": "fundamental",
            "tokenizer": None,
        }

        # Expected types for parameters
        self.allowed_params_and_types = {
            "loss_fn": Callable,
            "lr": (float, str),
            "batch_size": int,
            "weight_decay": float,
            "label_smoothing": float,
            "optimizer_class": type,
            "preprocessing": str,
            "tokenizer": any,
        }

        # Allowed specific values for certain parameters
        self.allowed_values = {
            "lr": ["cosine", "one_cycle", "reduce_on_plateau"],
            "optimizer_class": [Adam, AdamW],
            "preprocessing": ["fundamental", "emoji_and_emoticons"],
        }

        # Results df
        self.results = None

    def search(self, epochs: int = 30, verbose: bool = False) -> None:
        """
        Conduct the grid search. Orchestrates the workflow by initializing the model,
        creating dataloaders, configuring training components, and training.
        """
        # Initialize the results dataframe with the allowed params
        self._initialize_results()

        with tqdm(total=len(self.param_dicts), desc="Grid Search Progress", disable=False, leave=False) as pbar:
            for i, param_dict in enumerate(self.param_dicts):
                try:
                    # validate params
                    self._validate_params(param_dict)

                    # set params for the iterate
                    self._set_param_configuration(param_dict)

                    # Initialize model and dataloaders
                    training_loader, validation_loader = self._create_dataloaders()

                    # Train
                    parent_dir = f"{self.parent_dir}/" if self.parent_dir else ""
                    training_session = TrainingSession(
                        model_class=self.model_class,
                        model_init_kwargs=self._get_model_init_kwargs(),
                        training_loader=training_loader,
                        validation_loader=validation_loader,
                        loss_fn=self.param_configuration['loss_fn'],
                        seeds=[SEED],
                        weights_path=f"{parent_dir}weights/{self.model_class}/grid_search",
                        optimizer_class=self.param_configuration['optimizer_class'],
                        lr=self.param_configuration['lr'],
                        weight_decay=self.param_configuration['weight_decay'],
                        session_name=self._get_session_name(i)
                    )

                    model, loss_histories = training_session.train(
                        epochs=epochs, 
                        verbose=verbose,
                        device=self.device,
                        unlock_backbone=epochs
                    )

                    # Evaluate the model
                    weights_path = f"{parent_dir}weights/{self.model_class}/grid_search/{SEED}/{self.model_class}_{self._get_session_name(i)}.pth"
                    zero_one_f1 = self._evaluate(model, weights_path, validation_loader, SEED)

                    # Save results
                    self._save_results(
                        f1_score=zero_one_f1,
                        loss_histories=loss_histories,
                        seed=SEED,
                        i=i
                    )
                    
                except Exception as e:
                    print(f"Warning: iteration {i} failed with error: {e}")
                    raise e

                # Update progress bar
                pbar.update(1)


    def _initialize_results(self) -> None:
        """
        Initializes the results DataFrame with columns based on allowed parameters,
        plus additional columns for tracking performance metrics.

        Columns:
            - All parameters in param_configuration
            - "last_save": Stores the last epoch where weights were saved.
            - "0/1_f1_score": Stores the computed 0/1 F1 score.
        """
        self.results = pd.DataFrame(columns=list(self.param_configuration.keys()) + ["last_save", "0/1_f1_score"])


    @staticmethod
    def _generate_param_dicts(params: Dict[str, List]) -> List[Dict[str, any]]:
        """
        Generates all combinations of parameter values into a list of dictionaries.  

        :param params: Dictionary of grid search parameters.
        :type params: dict
        :return: List of parameter dictionaries.
        :rtype: list
        """
        param_combinations = list(product(*params.values()))
        return [dict(zip(params.keys(), combo)) for combo in param_combinations]
    
    def _validate_params(self, param_dict: Dict[str, any]) -> None:
        """
        Validates that all keys in the param_dict are within the allowed parameters
        and that their values conform to the expected types or constraints.

        :param param_dict: Dictionary containing parameters to validate.
        :type param_dict: Dict[str, any]
        :raises AssertionError: If any parameter is not allowed or its value is of an incorrect type.
        """
        # Validate keys
        invalid_params = [key for key in param_dict.keys() if key not in self.param_configuration]
        assert not invalid_params, f"Invalid parameters: {invalid_params}. Allowed parameters are: {list(self.param_configuration.keys())}"

        # Validate types and specific values
        for key, value in param_dict.items():
            expected_type = self.allowed_params_and_types.get(key)

            # Check type
            assert isinstance(value, expected_type), (
                f"Parameter '{key}' must be of type {expected_type}, but got {type(value).__name__}."
            )

            # Check specific allowed values, if applicable
            if key in self.allowed_values and isinstance(value, str):
                assert value in self.allowed_values[key], (
                    f"Parameter '{key}' has invalid value '{value}'. Allowed values are: {self.allowed_values[key]}"
                )
    
    def _set_param_configuration(self, param_dict: Dict[str, any]) -> None:
        """
        Configures searchable parameters for the grid search. Updates `self.param_configuration` with `param_dict`.

        :param param_dict: Dictionary containing parameters to search for and their values.
        :type param_dict: Dict[str, any]
        """
        self.param_configuration.update(param_dict)

    @abstractmethod
    def _create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        pass

    @abstractmethod
    def _get_model_init_kwargs(self) -> Dict[str, any]:
        pass

    def _get_session_name(self, i: int) -> str:
        return f"grid_search_{i}"
    
    def _evaluate(self, model: torch.nn.Module, weights_path: str, val_loader: DataLoader, seed: int) -> float:
        """
        Evaluates the model on the validation dataset and computes the 0/1 F1 score.

        :param model: Trained model to evaluate.
        :type model: torch.nn.Module
        :param weights_path: The path to the model weights.
        :type weights_path: str
        :param val_loader: DataLoader for validation data.
        :type val_loader: torch.utils.data.DataLoader
        :param seed: Random seed used for training (used for generating weight file names).
        :type seed: int
        :return: The 0/1 F1 score computed for the model on the validation data.
        :rtype: float
        """

        # Predict values using the test_predict function
        y_true, y_pred = test_predict(model=model, dataloader=val_loader, device=self.device, weights_path=weights_path)

        # Get best threshold
        thr = best_threshold(y_true, y_pred)

        # Compute zero-one F1 scores
        _, _, zero_one_f1 = zero_one_f1_score(y_true, y_pred, thr=thr)
        
        return zero_one_f1

    def _save_results(
        self,
        f1_score: float,
        loss_histories: Dict[str, List[float]],
        seed: int,
        i: int
    ) -> None:
        """
        Adds a new row to the results DataFrame, updates it with the current parameters, last epoch, and F1 score,
        sorts the DataFrame by "0/1_f1_score" in descending order, and saves plots.

        :param params: Current parameter configuration.
        :type params: Dict[str, any]
        :param f1_score: Computed 0/1 F1 score for the model.
        :type f1_score: float
        :param loss_histories: Training and validation loss histories.
        :type loss_histories: Dict[str, List[float]]
        :param seed: Current seed.
        :type seed: int
        """
        # Add a new row to results DataFrame
        new_row = self.param_configuration.copy()
        new_row["loss_fn"] = (
            str(self.param_configuration["loss_fn"].__name__)
            if isinstance(self.param_configuration["loss_fn"], Callable)
            else str(self.param_configuration["loss_fn"])
        )
        new_row["tokenizer"] = type(self.param_configuration["tokenizer"]).__name__
        new_row["optimizer_class"] = self.param_configuration["optimizer_class"].__name__
        new_row["last_save"] = loss_histories[seed]["last_save"]
        new_row["0/1_f1_score"] = f1_score

        if self.model_class == "gru":
            new_row["embedding_model"] = "italian_word2vec_100.bin"

        if self.results.empty:
            self.results.loc[0] = new_row
        else:
            self.results = pd.concat([self.results, pd.DataFrame([new_row], index=[i])], ignore_index=False)

        # Sort results DataFrame by "0/1_f1_score" in descending order
        self.results.sort_values(by="0/1_f1_score", ascending=False, inplace=True)

        # Save results
        if self.parent_dir:
            os.makedirs(os.path.join(self.parent_dir, "grid_search"), exist_ok=True)
            results_path = os.path.join(self.parent_dir, "grid_search", f"{self.model_class}_grid_search_results.json")
        else:
            os.makedirs("grid_search", exist_ok=True)
            results_path = os.path.join("grid_search", f"{self.model_class}_grid_search_results.json")

        self.results.to_json(results_path, indent=4)

        # Save loss plots
        if self.parent_dir:
            os.makedirs(os.path.join(self.parent_dir, "grid_search", "ïmgs"), exist_ok=True)
            plot_path = os.path.join(self.parent_dir, "grid_search", "imgs", f"{self._get_session_name(i)}_loss.jpg")
        else:
            os.makedirs(os.path.join("grid_search", "ïmgs"), exist_ok=True)
            plot_path = os.path.join("grid_search", "imgs", f"{self._get_session_name(i)}_loss.jpg")

        save_loss_plot(
            loss_histories[seed]["train"],
            loss_histories[seed]["val"],
            file_path=plot_path,
        )

    def get_results(self) -> pd.DataFrame:
        """
        Returns the current state of the results DataFrame.

        :return: The results DataFrame containing parameters, last update, and F1 scores.
        :rtype: pd.DataFrame
        """
        if self.results is None:
            raise ValueError("The results DataFrame has not been initialized. Please run the search method first.")
        return self.results



class GRUGridSearch(GridSearch):
    """
    Grid Search Interface for the Gru model.  
    It requires having passed a dictionary,
    where the keys are the params to make the search for and the values are lists of values to try.  

    The possible params to search for are the following (example usage immediately below):   

    ### Training parameters
    `loss_fn` The losses to try. If it is not passed as key of param_dict, it is defaulted to BCE.  
    Values must be Callable.

    `lr` The learning rates or learning rates schedules to try. If it is not passed as key of param_dict, it is defaulted to 0.0001.  
    Values must be either a float or a string among the following: "cosine", "one_cycle" or "reduce_on_plateau".

    `batch_size`  The batch sizes to try. If it is not passed as key of param_dict, it is defaulted to 8.  
    Values must be integers.

    `weight_decay`  The amount of L2 regularization weight decay. If it is not passed as key of param_dict, it is defaulted to 0.0.  
    Values must be floats.

    `label_smoothing`  The epsilon value of the label smoothing. If it is not passed as key of param_dict, it is defaulted to 0.0.  
    Values must be floats.

    `optimizer` The optimizers to try. If it is not passed as key of param_dict, it is defaulted to Adam.  
    Values must be either "Adam" or "AdamW".

    `preprocessing` The type of preprocessing. If it is not passed as key of param_dict, it is defaulted to "classic".  
    Values must be either "classic" or "for_tweets"

    `gru_hidden_size`  The number of hidden units in the GRU. If it is not passed as key of param_dict, it is defaulted to 32.  
    Values must be integers.

    `num_gru_layers`  The number of hidden layers of the GRU. If it is not passed as key of param_dict, it is defaulted to 2.  
    Values must be integers.

    `gru_dropout`  The amount of dropout. If it is not passed as key of param_dict, it is defaulted to 0.1.  
    Values must be floats.

    `tokenizer`  The tokenizer to use. If it is not passed as key of param_dict, it is defaulted to TweetTokenizer.  
    Values must be either TweetTokenizer() or BaseTokenizer().

    Example
        ```python
            # Mock train and validation sets
            train_set = pd.DataFrame({"iro": [1, 0, 1, 0]})
            val_set = pd.DataFrame({"iro": [1, 0]})
            embedding_matrix = np.random.rand(100, 32)

            # Grid search parameters for GRU
            gru_params = {
                "loss_fn": [bce, WeightedFocalLoss().forward],
                "lr": [1e-4, 1e-5],
                "batch_size": [16, 32],
                "num_gru_layers": [2, 3],
                "gru_dropout": [0.1, 0.2],
                "label_smoothing": [0.0, 0.1],
                "optimizer": ["Adam", "AdamW"]
            }

            # Initialize the GridSearch object
            gru_search = GRUGridSearch(
                params=gru_params,
                train_set=train_set,
                val_set=val_set,
                device="cuda",
                max_len=100,
                vocabulary=vocabulary_instance  # assuming you have a Vocabulary instance
            )

            # Execute GRU grid search
            gru_search.search()

            # Get the results
            gru_search.get_results()
        ```
    :param params: Dictionary of grid search parameters.
    :type params: dict
    :param train_set: The train set to perform the search on.
    :type train_set: pandas.DataFrame
    :param val_set: The validation set to compute the metrics on.
    :type val_set: pandas.dataFrame
    :param device: The device for the computation. It must be either 'cpu' or 'cuda'.
    :param max_len: The max len of an input sequence.
    :type max_len: int
    """
    def __init__(
            self,
            params: Dict[str, List], 
            data_folder: str,
            max_len: int,
            device: Literal["cpu", "cuda"],
            parent_dir: Optional[str] = None
        ) -> None:
        

        # Init parent class
        super().__init__(
            params=params,
            data_folder=data_folder,
            device=device,
            max_len=max_len,
            parent_dir=parent_dir
        )

        self.model_class = "gru"

        # Update param_configuration with GRU-specific parameters
        parent_dir = f"{self.parent_dir}/" if self.parent_dir else ""
        embedding_model = KeyedVectors.load_word2vec_format(f'{parent_dir}embedding_models\italian_word2vec_100.bin', binary=True)
        self.param_configuration.update({
            # gru params
            "scale_grad_by_freq": False,
            "gru_hidden_size": 32,
            "num_gru_layers": 2,
            "gru_dropout": 0.1,
            "text_enrichment": False,

            # vocab params
            "embedding_model": embedding_model,
            "tokenizer": BaseTokenizer(),
            "embedding_size": 100
        })

        # Update allowed_params_and_types with GRU-specific parameters
        self.allowed_params_and_types.update({
            # gru params
            "scale_grad_by_freq": bool,
            "gru_hidden_size": int,
            "num_gru_layers": int,
            "gru_dropout": float,
            "text_enrichment": bool,

            # vocab params
            "embedding_model": KeyedVectors,
            "tokenizer": (TweetTokenizer, type(BaseTokenizer())),
            "embedding_size": int
        })

        # Placeholder for vocabulary
        self.vocabulary = None
    
    def _create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        # Get train and val set
        train_set, val_set, _ = load_datasets(folder=self.data_folder)

        # Loader params
        loader_params = {
            'batch_size': self.param_configuration["batch_size"],
            'pin_memory': True,
        }

        # Preprocess both datasets
        for subset in (train_set, val_set):
            fundamental_preprocessing(
                data=subset,
                lowercase=True,
                remove_urls=True,
                remove_video_placeholders=True,
                remove_html_entities=True,
                remove_retweet_markers=True,
                normalize_whitespace=True
            )

            if self.param_configuration["preprocessing"] == "emoji_and_emoticons":
                emoji_and_emoticons_preprocessing(subset)

        # Create vocabulary
        self.vocabulary: Vocabulary = self._create_vocabulary(train_set)

        # Apply label smoothing
        train_set_with_eps = train_set.copy(deep=True)
        label_smoothing = self.param_configuration["label_smoothing"]
        train_set_with_eps["iro"] = train_set_with_eps["iro"].apply(
            lambda x: 1 - label_smoothing if x == 1 else label_smoothing
        )

        # Create Dataset objects
        custom_dataset_train = GRUDataset(
            dataframe=train_set_with_eps,
            vocabulary=self.vocabulary,
            max_len=self.max_len,
            tokenizer=self.param_configuration["tokenizer"],
            text_enrichment=self.param_configuration["text_enrichment"]
        )
        
        custom_dataset_val = GRUDataset(
            dataframe=val_set,
            vocabulary=self.vocabulary,
            max_len=self.max_len,
            tokenizer=self.param_configuration["tokenizer"],
            text_enrichment=self.param_configuration["text_enrichment"]
        )

        # Create Dataloaders
        training_loader = DataLoader(custom_dataset_train, **loader_params)
        validation_loader = DataLoader(custom_dataset_val, **loader_params)

        return training_loader, validation_loader
    
    def _create_vocabulary(self, train_set):
        vocabulary = Vocabulary(
            dataset=train_set,
            embedding_model=self.param_configuration["embedding_model"],
            tokenizer=self.param_configuration["tokenizer"],
            embedding_size=self.param_configuration["embedding_size"]
        )
        vocabulary.create_vocabulary()
        return vocabulary
    
    def _get_model_init_kwargs(self):
        return {'embedding_matrix': self.vocabulary.embedding_matrix,
                'scale_grad_by_freq': self.param_configuration['scale_grad_by_freq'],
                'gru_hidden_size': self.param_configuration['gru_hidden_size'],
                'num_gru_layers': self.param_configuration['num_gru_layers'],
                'gru_dropout': self.param_configuration['gru_dropout']
                }
    
class GruBertGridSearch(GridSearch):
    """
    Grid Search Interface for the GruBert model.
    It requires a dictionary where the keys are the parameters to search over and the values are lists of values to try.

    The possible parameters to search for are:

    - `loss_fn`: The loss functions to try. Default is `bce`. Values must be callables.
    - `lr`: Learning rates or learning rate schedules to try. Default is `0.0001`. Values must be floats or strings among: `"cosine"`, `"one_cycle"`, `"reduce_on_plateau"`.
    - `batch_size`: Batch sizes to try. Default is `8`. Values must be integers.
    - `weight_decay`: L2 regularization weight decay. Default is `0.0`. Values must be floats.
    - `label_smoothing`: Epsilon value for label smoothing. Default is `0.0`. Values must be floats.
    - `optimizer`: Optimizers to try. Default is `"Adam"`. Values must be `"Adam"` or `"AdamW"`.
    - `preprocessing`: Type of preprocessing. Default is `"classic"`. Values must be `"classic"` or `"for_tweets"`.
    - `embedding_size`: Size of the BERT embeddings. Default is `1024`. Values must be integers.
    - `hidden_size`: Hidden size for the GRU layer. Default is `256`. Values must be integers.
    - `model_name`: Name of the BERT model. Default is `'bert-large-cased'`. Values must be valid BERT model names.

    Example:
    ```python
    # Grid search parameters for GruBert
    grubert_params = {
        "loss_fn": [bce],
        "lr": [1e-4, "cosine"],
        "batch_size": [16],
        "embedding_size": [768, 1024],
        "hidden_size": [256, 512],
        "model_name": ["bert-base-uncased", "Musixmatch/alberto"]
    }

    # Initialize the GridSearch object
    grubert_search = GruBertGridSearch(
        params=grubert_params,
        train_set=train_set,
        val_set=val_set,
        tokenizer=bert_tokenizer,
        device="cuda",
        max_len=128  # Specify max sequence length
    )

    # Execute GruBert grid search
    grubert_search.search()

    # Get the results
    grubert_search.get_results()
    ```
    """

    def __init__(
            self,
            params: Dict[str, List], 
            data_folder: str,
            device: Literal["cpu", "cuda"],
            max_len: Optional[int] = None,
            parent_dir: Optional[str] = None
        ) -> None:

        # Initialize the parent class
        super().__init__(
            params=params,
            data_folder=data_folder,
            device=device,
            max_len=max_len,
            parent_dir=parent_dir
        )

        self.model_class = "grubert"

        # Update param_configuration with GruBert-specific parameters
        self.param_configuration.update({
            "model_name": "m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0", #'nickprock/sentence-bert-base-italian-uncased'
            "embedding_size": 768, 
            "gru_hidden_size": 256, 
            "num_gru_layers": 2, 
            "gru_dropout": 0.1, 
            "output_size": 1, 
            "text_enrichment": False, 
        })

        self.param_configuration.update({
            "tokenizer": AutoTokenizer.from_pretrained(self.param_configuration["model_name"])
        })

        # Update allowed_params_and_types with GruBert-specific parameters
        self.allowed_params_and_types.update({
            "model_name": str,
            "embedding_size": int,
            "gru_hidden_size": int,
            "num_gru_layers": int, 
            "gru_dropout": float, 
            "output_size": int, 
            "text_enrichment": bool,
            "tokenizer":  BertTokenizerFast
        })

        # Update allowed_values with GruBert-specific allowed values
        self.allowed_values.update({
            "model_name": [
                "bert-base-uncased",
                "bert-large-uncased",
                "roberta-base",
                "roberta-large",
                "Musixmatch/alberto",
                "nickprock/sentence-bert-base-italian-uncased",
                "m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0"
            ]
        })

    def _create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        # Get train and val set
        train_set, val_set, _ = load_datasets(folder=self.data_folder)

        # Loader params
        loader_params = {
            'batch_size': self.param_configuration["batch_size"],
            'pin_memory': True,
        }

        # Preprocess both datasets
        for subset in (train_set, val_set):
            fundamental_preprocessing(
                data=subset,
                lowercase=True,
                remove_urls=True,
                remove_video_placeholders=True,
                remove_html_entities=True,
                remove_retweet_markers=True,
                normalize_whitespace=True
            )

            if self.param_configuration["preprocessing"] == "emoji_and_emoticons":
                emoji_and_emoticons_preprocessing(subset)

        # Apply label smoothing
        train_set_with_eps = train_set.copy(deep=True)
        label_smoothing = self.param_configuration["label_smoothing"]
        train_set_with_eps["iro"] = train_set_with_eps["iro"].apply(
            lambda x: 1 - label_smoothing if x == 1 else label_smoothing
        )

        # Create Dataset objects
        custom_dataset_train = CustomDataset(
            dataframe=train_set_with_eps,
            max_len=self.max_len,
            tokenizer=self.param_configuration["tokenizer"],
        )
        
        custom_dataset_val = CustomDataset(
            dataframe=val_set,
            max_len=self.max_len,
            tokenizer=self.param_configuration["tokenizer"],
        )

        # Create Dataloaders
        training_loader = DataLoader(custom_dataset_train, **loader_params)
        validation_loader = DataLoader(custom_dataset_val, **loader_params)

        return training_loader, validation_loader
    
    def _get_model_init_kwargs(self):
        return {
            "embedding_size": self.param_configuration["embedding_size"],
            "gru_hidden_size": self.param_configuration["gru_hidden_size"],
            "model_name": self.param_configuration["model_name"],
            "gru_dropout": self.param_configuration["gru_dropout"],
            "num_gru_layers": self.param_configuration["num_gru_layers"],
            "output_size": 1
            }
