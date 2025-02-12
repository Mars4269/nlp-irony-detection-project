import os
import re
import numpy as np
import pandas as pd
import torch

from typing import List, Dict, Optional, Tuple, Union, Literal
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from gensim.models import KeyedVectors
from tqdm import tqdm
from transformers import AutoTokenizer

# === Your local modules ===
from models import Gru, GruBERT
from ModelOps import test_predict, load_weights
from CustomDataset import GRUDataset, CustomDataset
import vocabulary as v  # module with your Vocabulary class

def zero_one_f1_score(y_true, y_pred, evaluate_baselines=False, thr=None):
    '''Calculates the zero-one F1 score for a binary classification task.'''
    y_true_label = np.array(y_true)
    y_pred_label = np.array(y_pred)

    # Baselines outputs already contain hard labels,
    # while our models output logits that need to be thresholded
    if not evaluate_baselines:
        try:
            y_pred_label = (y_pred_label >= thr).astype(int)
        except Exception as e:
            raise ValueError("You need to provide a threshold for the logits.\n If you are evaluating a dummy baseline, make sure to specify ``evaluate_baselines=True``.")

    # Iro f1

    f1_1 = f1_score(y_true_label, y_pred_label)

    # Non-Iro labels
    y_true_negative = 1 - y_true_label
    y_pred_negative = 1 - y_pred_label
    
    # Non-Iro f1 score
    f1_0 = f1_score(np.array(y_true_negative), np.array(y_pred_negative))

    # Avg f1 score
    zero_one_f1_score = (f1_1 + f1_0)/2

    return f1_0, f1_1, zero_one_f1_score

def compute_baseline_metrics(y_true: np.ndarray, num_samples: int, baseline_type: Literal["random", "majority"], seeds: List[int]) -> tuple:
    '''Compute the baseline metrics for the given baseline type (random or majority).'''

    # Check the baseline type and generate predictions accordingly
    if baseline_type == "random":
        predictions = []

        for seed in seeds:
            random_state = np.random.RandomState(seed=seed)
            y_pred = random_state.randint(2, size=num_samples)
            predictions.append( zero_one_f1_score(y_true, y_pred, evaluate_baselines=True) )

        predictions = np.array(predictions)
        return np.mean(predictions, axis=0)
        
    y_pred = np.zeros(num_samples)  # All predictions are 0 for Majority
     
    # Compute zero-one F1 scores for the baseline model
    f1_0, f1_1, zero_one_f1 = zero_one_f1_score(y_true, y_pred, evaluate_baselines=True)
    
    return f1_0, f1_1, zero_one_f1

##############################################################################
# MODEL FACTORY
##############################################################################
class ModelFactory:
    """
    Responsible for:
     1. Parsing the .pth filename to get (model_class, configuration).
     2. Interpreting configuration strings to determine which architecture flags to set.
     3. Instantiating the correct model (Gru or GruBERT) with the right hyperparameters.
     4. Loading the .pth weights.
    """

    @staticmethod
    def parse_model_filename(file_name: str) -> Tuple[str, str]:
        """
        E.g. "Gru_base_case.pth" => ("gru", "base_case")
             "GruBERT_hashtag_enrichment.pth" => ("grubert", "hashtag_enrichment")
        """
        f_lower = file_name.lower()

        if f_lower.startswith("gru_"):
            model_class = "gru"
            config_part = file_name[4:]  # everything after "Gru_"
        elif f_lower.startswith("grubert_"):
            model_class = "grubert"
            config_part = file_name[8:]  # everything after "GruBERT_"
        else:
            raise ValueError(
                f"File '{file_name}' does not start with 'Gru_' or 'GruBERT_'."
            )

        # Strip .pth if present
        if config_part.endswith(".pth"):
            config_part = config_part[:-4]

        return model_class, config_part

    @staticmethod
    def interpret_configuration(config_name: str) -> Dict[str, Optional[Union[bool, int]]]:
        """
        The user-provided structure:
          - if "hashtag_segmentation" in config => hashtag_segmentation=True
          - if "pos_tags_enrichment" in config => also hashtag_segmentation=True, pos_tags=True, num_pos_tags=48
          - if "hashtag_enrichment" in config  => also hashtag_segmentation=True, pos_tags=True, num_pos_tags=48, text_enrichment=True
        Anything else => default is base case.

        Returns a dict with:
          {
            "hashtag_segmentation": bool,
            "pos_tags": Optional[bool],
            "num_tags": Optional[int],
            "text_enrichment": bool
          }
        """
        cfg_lower = config_name.lower()

        # Default
        hashtag_segmentation = False
        text_enrichment = False
        pos_tags = None
        num_tags = None

        if "hashtag_segmentation" in cfg_lower:
            hashtag_segmentation = False

        if "pos_tags_enrichment" in cfg_lower:
            hashtag_segmentation = False
            pos_tags = True
            num_tags = 48

        if "hashtag_enrichment" in cfg_lower:
            hashtag_segmentation = False
            text_enrichment = True

        return {
            "hashtag_segmentation": hashtag_segmentation,
            "pos_tags": pos_tags,
            "num_tags": num_tags,
            "text_enrichment": text_enrichment
        }

    @staticmethod
    def create_model(
        model_class: str,
        config_flags: Dict[str,  Optional[Union[bool, int]]],
        path_to_weights: str,
        device: torch.device,
        # Additional hyperparams used in training
        gru_hidden_size: int = 32,
        num_gru_layers: int = 2,
        gru_dropout: float = 0.2,
        embedding_dim_gru: int = 100
    ) -> torch.nn.Module:
        """
        Create and load the model, returning an nn.Module on the specified device.

        :param model_class: "gru" or "grubert"
        :param config_flags: dictionary from interpret_configuration
        :param path_to_weights: path to the .pth file with saved state_dict
        :param device: CPU or GPU
        :param gru_hidden_size, num_gru_layers, gru_dropout, embedding_dim_gru: hyperparams you used
        :return: Instantiated and weight-loaded model
        """
        # Extract config flags
        text_enrichment = config_flags["text_enrichment"]
        num_tags = config_flags["num_tags"]

        if model_class == "gru":
            # We can pass a dummy embedding matrix to Gru; the real weights are in the checkpoint
            # but we must ensure shape matches training-time shape.
            dummy_vocab_size = 2471955 
            dummy_embedding = torch.randn((dummy_vocab_size, embedding_dim_gru), dtype=torch.float32)

            model = Gru(
                embedding_matrix=dummy_embedding,
                scale_grad_by_freq=False,  # or True, depending
                gru_hidden_size=gru_hidden_size,
                num_gru_layers=num_gru_layers,
                gru_dropout=gru_dropout,
                text_enrichment=text_enrichment,
                num_tags=num_tags
            )
        else:
            # "grubert"
            embedding_size = 768  # typical BERT hidden size
            model = GruBERT(
                embedding_size=embedding_size,
                gru_hidden_size=gru_hidden_size,
                num_gru_layers=num_gru_layers,
                gru_dropout=gru_dropout,
                model_name="m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0",
                text_enrichment=text_enrichment,
                num_tags=num_tags
            )

        # Load checkpoint
        load_weights(model, path_to_weights)
        model.to(device)

        return model


##############################################################################
# DATA FACTORY
##############################################################################
class DataFactory:
    """
    Responsible for creating the correct Datasets and DataLoaders
    given the model class, config flags, and user-provided data (DataFrames).
    """

    def __init__(
        self,
        embedding_model_path: str,
        bert_model_name: str = "m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0",
        tokenizer_for_gru=None
    ):
        """
        :param embedding_model_path: path to word2vec embeddings for GRU
        :param bert_model_name: huggingface model name for BERT-based tokenizer
        :param tokenizer_for_gru: optional custom tokenizer (if you want a BaseTokenizer from vocabulary)
        """
        self.embedding_model_path = embedding_model_path
        self.bert_model_name = bert_model_name
        self.tokenizer_for_gru = tokenizer_for_gru if tokenizer_for_gru else v.BaseTokenizer()

    def create_dataloaders(
        self,
        model_class: str,
        config_flags: Dict[str,Optional[Union[bool, int]]],
        threshold_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        batch_size: int = 8,
        max_len: int = 50
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Build and return (threshold_loader, eval_loader) for the given model class.
        """
        if model_class == "gru":
            return self._create_gru_dataloaders(config_flags, threshold_df, eval_df, batch_size, max_len)
        else:
            return self._create_grubert_dataloaders(config_flags, threshold_df, eval_df, batch_size, max_len)

    def _create_gru_dataloaders(
        self,
        config_flags: Dict[str, Optional[Union[bool, int]]],
        threshold_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        batch_size: int,
        max_len: int
    ) -> Tuple[DataLoader, DataLoader]:

        # Load KeyedVectors and build a Vocabulary
        w2v_model = KeyedVectors.load_word2vec_format(self.embedding_model_path, binary=True)
        vocab_obj = v.Vocabulary(
            dataset=threshold_df, 
            embedding_model=w2v_model,
            embedding_size=100, 
            tokenizer=self.tokenizer_for_gru
        )
        vocab_obj.create_vocabulary()

        # Pull out flags
        hashtag_seg = config_flags["hashtag_segmentation"]
        pos_tags = config_flags["pos_tags"]
        text_enrichment = config_flags["text_enrichment"]

        threshold_dataset = GRUDataset(
            dataframe=threshold_df,
            vocabulary=vocab_obj,
            tokenizer=self.tokenizer_for_gru,
            max_len=max_len,
            hashtag_segmentation=hashtag_seg,
            pos_tags=pos_tags,
            text_enrichment=text_enrichment
        )
        threshold_loader = DataLoader(threshold_dataset, batch_size=batch_size, pin_memory=True)

        eval_dataset = GRUDataset(
            dataframe=eval_df,
            vocabulary=vocab_obj,
            tokenizer=self.tokenizer_for_gru,
            max_len=max_len,
            hashtag_segmentation=hashtag_seg,
            pos_tags=pos_tags,
            text_enrichment=text_enrichment
        )
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, pin_memory=True)

        return threshold_loader, eval_loader

    def _create_grubert_dataloaders(
        self,
        config_flags: Dict[str, Optional[Union[bool, int]]],
        threshold_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        batch_size: int,
        max_len: int
    ) -> Tuple[DataLoader, DataLoader]:

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name, token="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")

        hashtag_seg = config_flags["hashtag_segmentation"]
        pos_tags = config_flags["pos_tags"]
        text_enrichment = config_flags["text_enrichment"]

        threshold_dataset = CustomDataset(
            dataframe=threshold_df,
            tokenizer=tokenizer,
            max_len=max_len,
            hashtag_segmentation=hashtag_seg,
            pos_tags=pos_tags,
            text_enrichment=text_enrichment
        )
        threshold_loader = DataLoader(threshold_dataset, batch_size=batch_size, pin_memory=True)

        eval_dataset = CustomDataset(
            dataframe=eval_df,
            tokenizer=tokenizer,
            max_len=max_len,
            hashtag_segmentation=hashtag_seg,
            pos_tags=pos_tags,
            text_enrichment=text_enrichment
        )
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, pin_memory=True)

        return threshold_loader, eval_loader


##############################################################################
# EVALUATOR
##############################################################################
class Evaluator:
    """
    High-level orchestrator class that:
      1) Iterates over a list of files (model checkpoints).
      2) Uses ModelFactory to parse the file name, interpret the config, and create the model.
      3) Uses DataFactory to create threshold & eval Dataloaders.
      4) Finds best threshold and computes final metrics.
      5) Returns a summary DataFrame.
    """

    def __init__(
        self,
        data_factory: DataFactory,
        device: torch.device = torch.device("cpu"),
        # Model hyperparams:
        gru_hidden_size: int = 32,
        num_gru_layers: int = 2,
        gru_dropout: float = 0.2,
        embedding_dim_gru: int = 100,
        batch_size: int = 8,
        max_len: int = 50
    ):
        """
        :param data_factory: An instance of DataFactory to create Datasets & Dataloaders.
        :param device: CPU or GPU
        :param gru_hidden_size, num_gru_layers, etc.: The defaults to use for model creation.
        :param batch_size, max_len: Dataloader & tokenization parameters.
        """
        self.data_factory = data_factory
        self.device = device

        self.gru_hidden_size = gru_hidden_size
        self.num_gru_layers = num_gru_layers
        self.gru_dropout = gru_dropout
        self.embedding_dim_gru = embedding_dim_gru

        self.batch_size = batch_size
        self.max_len = max_len

    def compute_metrics_over_files(
        self,
        file_list: List[str],
        threshold_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        parent_dir: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Main entry point: for each file in file_list,
          1) parse the file
          2) create the model & dataloaders
          3) find best threshold
          4) compute final F1

        Returns a DataFrame of results.
        """
        results = []

        for file_name in tqdm(file_list, desc="Evaluating models"):
            # 1) Create the full path
            full_path = file_name
            if parent_dir is not None:
                full_path = os.path.join(parent_dir, file_name)

            # 2) Parse the file => model_class, config_name
            model_class, config_name = ModelFactory.parse_model_filename(file_name)

            # 3) Interpret the config => flags
            config_flags = ModelFactory.interpret_configuration(config_name)

            # 4) Create dataloaders for threshold & evaluation sets
            thr_loader, eval_loader = self.data_factory.create_dataloaders(
                model_class=model_class,
                config_flags=config_flags,
                threshold_df=threshold_df,
                eval_df=eval_df,
                batch_size=self.batch_size,
                max_len=self.max_len
            )

            # 5) Instantiate the model & load weights
            model = ModelFactory.create_model(
                model_class=model_class,
                config_flags=config_flags,
                path_to_weights=full_path,
                device=self.device,
                gru_hidden_size=self.gru_hidden_size,
                num_gru_layers=self.num_gru_layers,
                gru_dropout=self.gru_dropout,
                embedding_dim_gru=self.embedding_dim_gru
            )

            # 6) Compute best threshold
            y_thr_true, y_thr_pred = test_predict(model, None, thr_loader, self.device)
            best_thr = self.find_best_threshold(y_thr_true, y_thr_pred)

            # 7) Evaluate final metrics
            y_eval_true, y_eval_pred = test_predict(model, None, eval_loader, self.device)
            f1_0, f1_1, avg_f1 = self.zero_one_f1_score(y_eval_true, y_eval_pred, threshold=best_thr)

            # Gather result row
            results.append({
                "ModelClass": model_class,
                "Configuration": config_name,
                "F1_not_ironic": f1_0,
                "F1_ironic": f1_1,
                "F1_avg_0_1": avg_f1,
                "BestThreshold": best_thr
            })

        return pd.DataFrame(results)
        
    def _get_predictions_for_file(
        self,
        file_name: str,
        threshold_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        parent_dir: Optional[str] = None
    ) -> Tuple[str, str, np.ndarray, np.ndarray, float]:
        """
        Private helper method to load a model from a weights file, create dataloaders,
        and compute predictions needed for error analysis.

        Steps:
        1. Create the full file path.
        2. Parse the file name to extract model class and configuration name.
        3. Interpret configuration flags.
        4. Create threshold and evaluation DataLoaders.
        5. Instantiate the model and load weights.
        6. Compute predictions on the threshold set to determine the best threshold.
        7. Compute predictions on the evaluation set.
        
        Returns:
        model_class (str): The model class (e.g., "gru" or "grubert").
        config_name (str): The configuration name extracted from the file.
        y_eval_true (np.ndarray): True labels on the evaluation set.
        y_eval_pred (np.ndarray): Predicted scores on the evaluation set.
        best_thr (float): The best threshold computed from the threshold set.
        """
        # 1. Full path construction.
        full_path = file_name if parent_dir is None else os.path.join(parent_dir, file_name)
        
        # 2. Parse the file name.
        model_class, config_name = ModelFactory.parse_model_filename(file_name)
        
        # 3. Interpret configuration flags.
        config_flags = ModelFactory.interpret_configuration(config_name)
        
        # 4. Create dataloaders for threshold and evaluation datasets.
        thr_loader, eval_loader = self.data_factory.create_dataloaders(
            model_class=model_class,
            config_flags=config_flags,
            threshold_df=threshold_df,
            eval_df=eval_df,
            batch_size=self.batch_size,
            max_len=self.max_len
        )
        
        # 5. Instantiate the model and load weights.
        model = ModelFactory.create_model(
            model_class=model_class,
            config_flags=config_flags,
            path_to_weights=full_path,
            device=self.device,
            gru_hidden_size=self.gru_hidden_size,
            num_gru_layers=self.num_gru_layers,
            gru_dropout=self.gru_dropout,
            embedding_dim_gru=self.embedding_dim_gru
        )
        
        # 6. Compute predictions on the threshold set and find the best threshold.
        y_thr_true, y_thr_pred = test_predict(model, None, thr_loader, self.device)
        best_thr = self.find_best_threshold(y_thr_true, y_thr_pred)
        
        # 7. Compute predictions on the evaluation set.
        y_eval_true, y_eval_pred = test_predict(model, None, eval_loader, self.device)
        
        return model_class, config_name, y_eval_true, y_eval_pred, best_thr


    def plot_precision_recall_curves_over_files(
        self,
        weights_file_path: str,
        threshold_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        title: Optional[str] = None,
        parent_dir: Optional[str] = None,
        label: int = 1
    ) -> None:
        """
        For each weights file in file_list, load the corresponding model, obtain evaluation predictions,
        compute the Precision-Recall curve, and plot all curves as subplots in a single figure.
        """

        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve, average_precision_score
        import seaborn as sns
        sns.set_style("darkgrid")
        sns.set_palette("pastel")

        # Load model predictions and related information.
        model_class, config_name, y_eval_true, y_eval_pred, _ = self._get_predictions_for_file(
            weights_file_path, threshold_df, eval_df, parent_dir
        )

        y_eval_true = np.array(y_eval_true)
        y_eval_pred = np.array(y_eval_pred)

        y_eval_true = np.abs(1 - label - y_eval_true)
        y_eval_pred = np.abs(1 - label - y_eval_pred)
        
        # Compute precision-recall curve and average precision.
        precision, recall, _ = precision_recall_curve(y_eval_true, y_eval_pred)
        ap_score = average_precision_score(y_eval_true, y_eval_pred)
        
        # Plot the curve in the corresponding subplot.
        # ax = plt.axes()
        plt.plot(recall, precision, label=f'AP = {ap_score:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        if title:
            plt.title(f"{title}\nAverage Precision = {ap_score:.2f}")
        else:
            plt.title(f"Precision-Recall Curve\nAverage Precision = {ap_score:.2f}")
        plt.legend(loc='lower left')
        plt.grid(True)

        # plt.tight_layout()
        # plt.show()


    def plot_confusion_matrix_for_file(
        self,
        weights_file_path: str,
        threshold_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        parent_dir: Optional[str] = None,
        normalize: bool = False
    ) -> None:
        """
        Loads the model, gets predictions, computes a confusion matrix (optionally normalized),
        and plots it as a single figure.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import itertools
        from sklearn.metrics import confusion_matrix
        # Turn off any automatic gridlines
        # 1. Retrieve predictions and threshold
        model_class, config_name, y_eval_true, y_eval_pred, best_thr = self._get_predictions_for_file(
            weights_file_path, threshold_df, eval_df, parent_dir
        )
        
        # 2. Binarize predictions using the best threshold
        y_pred_label = (np.array(y_eval_pred) >= best_thr).astype(int)

        # 3. Compute confusion matrix
        cm = confusion_matrix(y_eval_true, y_pred_label)
        if normalize:
            # Normalize by row sums (add a small epsilon to avoid division by zero)
            cm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

        # 4. Plot
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.grid(False)
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
        ax.set_title(f"Confusion Matrix - {model_class.upper()}", fontsize=14)

        # 5. Set up axis ticks
        tick_marks = np.arange(2)  # for binary classification: 0,1
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(["Not Ironic", "Ironic"], rotation=45, ha="right")
        ax.set_yticklabels(["Not Ironic", "Ironic"])

        # 6. Annotate each cell
        fmt = ".2f" if normalize else "d"
        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")

        plt.tight_layout()
        plt.show()

        return cm

    def print_mistaken_tweets(
        self,
        true_label: int,
        weights_file_path: str,
        threshold_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        parent_dir: Optional[str] = None,
    ) -> None:
        # 1. Retrieve predictions and threshold
        _, _, _, y_eval_pred, best_thr = self._get_predictions_for_file(
            weights_file_path, threshold_df, eval_df, parent_dir
        )
        
        # 2. Binarize predictions using the best threshold
        y_pred_label = (np.array(y_eval_pred) >= best_thr).astype(int)
        eval_df['prediction'] = y_pred_label

        # 3. Get tweets whose true label is true_label and predicted label is the opposite
        mistaken_mask = (eval_df['iro'] == true_label) & (eval_df['prediction'] == np.abs(1 - true_label))
        mistaken_tweets = eval_df[mistaken_mask]

        # 4. Print
        print(f"\nMistaken tweets where true label = {true_label} but predicted = {1 - true_label}:")
        for _, row in mistaken_tweets.iterrows():
            print(row["text"])

        



    @staticmethod
    def plot_error_frequencies(cm):
        """
        Plots a stacked bar chart from a confusion matrix (2x2) for binary classification:
        - cm[0, 0]: True Negatives
        - cm[0, 1]: False Positives
        - cm[1, 0]: False Negatives
        - cm[1, 1]: True Positives

        Args:
            cm (numpy.ndarray): A 2x2 confusion matrix.
        """

        import numpy as np
        import seaborn as sns
        import matplotlib.pyplot as plt
        # Use Seaborn darkgrid style and pastel color palette
        sns.set_theme(style="darkgrid", palette="pastel")

        # Classes: Index 0 = "Not Ironic", Index 1 = "Ironic"
        categories = ["Not Ironic", "Ironic"]

        # Extract counts for correct vs. incorrect by true class
        # correct[0] = True Negative count,  correct[1] = True Positive count
        # incorrect[0] = False Positive count, incorrect[1] = False Negative count
        correct = [cm[0, 0], cm[1, 1]]
        incorrect = [cm[0, 1], cm[1, 0]]

        x = np.array([0, 0.3])

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.margins(y=0.2)
        
        # Width of the bars
        bar_width = 0.1

        # Plot the "correct" portion
        ax.bar(x, correct, width=bar_width, label='Correct')

        # Plot the "incorrect" portion on top (stacked)
        ax.bar(x, incorrect, width=bar_width, bottom=correct, label='Incorrect')

        # Annotate the bars with % correct vs. % incorrect
        for i in range(len(categories)):
            total = correct[i] + incorrect[i]
            if total > 0:
                correct_pct = 100.0 * correct[i] / total
                incorrect_pct = 100.0 * incorrect[i] / total
                # Position the text above the stacked bar
                ax.text(
                    x[i], 
                    correct[i] + incorrect[i] + (0.02 * total), 
                    f"{correct_pct:.0f}% - {incorrect_pct:.0f}%", 
                    ha='center', 
                    va='bottom', 
                    fontsize=11
                )

        # Set x-axis labels, y-axis label, and legend
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_ylabel('Count')
        ax.set_title('Stacked Bar of Correct vs. Incorrect Predictions')
        ax.legend()

        plt.tight_layout()
        plt.show()





    @staticmethod
    def zero_one_f1_score(y_true, y_pred, threshold: float):
        # Convert lists (or tensors) to NumPy arrays
        y_true = np.array(y_true, dtype=int)
        y_pred = np.array(y_pred)

        # Binarize predictions based on threshold
        y_pred_label = (y_pred >= threshold).astype(int)

        # F1 for class=1 (ironic)
        f1_1 = f1_score(y_true, y_pred_label)

        # F1 for class=0 (not ironic)
        y_true_neg = 1 - y_true       # now works fine because they're arrays
        y_pred_neg = 1 - y_pred_label
        f1_0 = f1_score(y_true_neg, y_pred_neg)

        avg_f1 = 0.5 * (f1_0 + f1_1)
        return f1_0, f1_1, avg_f1


    @staticmethod
    def find_best_threshold(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Search thresholds in [0, 1) by steps of 0.01,
        pick that which yields the best 0-1 F1 average.
        """
        thresholds = np.arange(0, 1, 0.01)
        best_thr, best_score = 0.0, 0.0
        for thr in thresholds:
            f1_0, f1_1, avg_f1 = Evaluator.zero_one_f1_score(y_true, y_pred, threshold=thr)
            if avg_f1 > best_score:
                best_score = avg_f1
                best_thr = thr
        return best_thr


import pandas as pd

def average_out_scores_for_seeds(
    results_per_seed: dict[int, pd.DataFrame]
) -> pd.DataFrame:
    """
    Given a dictionary mapping seed -> DataFrame (each with columns:
       "ModelClass", "Configuration", "F1_not_ironic", "F1_ironic", "F1_avg_0_1", "BestThreshold"),
    combine them and compute mean & std grouped by (ModelClass, Configuration).
    
    The returned DataFrame has MultiIndex columns:
      F1_not_ironic       F1_ironic      F1_avg_0_1     BestThreshold
         mean    std        mean   std      mean   std     mean   std
    """

    # 1) Concatenate all per-seed DataFrames into one
    #    ignoring_index so we have a single continuous row index
    combined_df = pd.concat(results_per_seed.values(), ignore_index=True)

    # 2) Group by ModelClass & Configuration, then aggregate numeric columns with mean & std
    agg_df = (
        combined_df
        .groupby(["ModelClass", "Configuration"], as_index=False)
        .agg({
            "F1_not_ironic": ["mean", "std"],
            "F1_ironic": ["mean", "std"],
            "F1_avg_0_1": ["mean", "std"],
            "BestThreshold": ["mean", "std"]
        })
    )

    # 3) Set (ModelClass, Configuration) as the index so it’s out of the columns
    agg_df.set_index(["ModelClass", "Configuration"], inplace=True)

    return agg_df
