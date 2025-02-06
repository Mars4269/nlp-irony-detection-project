from matplotlib.pyplot import (gca, title, show, figure, imshow, axis, tight_layout, subplot, plot, xlabel, ylabel, subplots_adjust, xticks)
from pandas import DataFrame, Series
from seaborn import histplot, set_theme  # type: ignore
from wordcloud import WordCloud, STOPWORDS  # type: ignore
from numpy import array, mean
from nltk.corpus import stopwords 
from collections import Counter
from typing import Optional, Tuple
import os

import seaborn as sns


# Constants for figure sizes
FIGSIZE = (12,6)
HIST_FIGSIZE = (12,8)
SENTENCE_LENGHT_FIGSIZE = (8,6)

# List of pastel colors for visualization
pastel_colors = [
    "#BAE1FF",  # Light Blue
    "#FFE0B2",  # Pastel Orange
    "#D1C4E9",  # Pastel Purple
    "#BAFFC9",  # Light Green
    "#D4A5A5",  # Pastel Red
    "#FFB3BA",  # Light Pink
    "#FFDFBA",  # Light Peach
    "#FFFFBA",  # Light Yellow
    "#F8BBD0",  # Pastel Pink
    "#DCEDC8",  # Pastel Green
    "#FFCCBC",  # Pastel Coral
    "#C5CAE9",  # Pastel Lavender
    "#FFF9C4",  # Pastel Lemon
    "#E1BEE7",  # Pastel Lilac
    "#FFECB3"   # Pastel Gold
]

def labels_hist(data: DataFrame, is_subplot: bool = False):
    """
    Visualize the data distribution of the "iro" column through all splits
    
    ### Args
        data (DataFrame): DataFrame containing the data to be plotted. It must contain a "split" column and a "iro" column.
        is_subplot (bool): Boolean indicating if the plot is part of a subplot.
    """
    if not is_subplot:
        figure(figsize=HIST_FIGSIZE)
    else:
        gca()  # Get current axes if it's a subplot
    
    set_theme()
    title('iro')
    
    histplot(data=data, x='split', hue='iro', multiple="dodge", shrink=.5, zorder=2)
    
    # Annotating the histogram with counts
    for _, p in enumerate(gca().patches):
        gca().annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                       textcoords='offset points')
    
    if not is_subplot:
        show()
import vocabulary as v
def wordcloud(
        data: DataFrame, 
        iro: Optional[int] = None, 
        is_subplot: bool = False, 
        parent_dir: Optional[str] = None,
        remove_words: Optional[list] = None,
        custom_title: Optional[str] = None
):
    """
    Creates a word cloud for the text in the data filtered by the 'iro' column.
    
    ### Args
        data (DataFrame): DataFrame containing the data to be plotted.
        iro (int): Integer indicating the filter for the 'iro' column (0 or 1).
        is_subplot (bool): Boolean indicating if the plot is part of a subplot.
    """

    if not is_subplot:
        figure(figsize=HIST_FIGSIZE)
        
    else:
        gca()  # Get current axes if it's a subplot

    italian_stopwords = set(stopwords.words('italian'))

    font_name = 'Doto-VariableFont_ROND,wght.ttf'
    font_path = font_name if parent_dir is None else os.path.join(parent_dir, font_name)

    # WordCloud for ironic tweets
    if iro is not None:
        ironic_text = " ".join(data[data['iro'] == iro]['text'])
    else:
        ironic_text = " ".join(data['text'])

    if remove_words is not None:
        tokenizer = v.BaseTokenizer()
        tokens = tokenizer.tokenize(ironic_text)
        # Filter out tokens that match the remove_words list
        filtered_tokens = [
            tok for tok in tokens 
            if tok not in remove_words
        ]
        ironic_text = " ".join(filtered_tokens)

    wordcloud_ironic = WordCloud(stopwords=italian_stopwords,
                        font_path=font_path
    ).generate(ironic_text)
    
    if custom_title:
        title(custom_title)
    else:
        title(f"iro = {iro}")
    imshow(wordcloud_ironic, interpolation='bilinear')
    axis('off')
    gca().set_frame_on(False)
    tight_layout()
    
    if not is_subplot:
        show()

def sentence_lenght(data: DataFrame):
    """
    Plots the distribution of sentence lengths in the 'text' column of the data.
    
    ### Args
        data (DataFrame): DataFrame containing the data to be analyzed.
    """
    x = data.text.str.split(' ')
    set_theme()
    
    figure(figsize=SENTENCE_LENGHT_FIGSIZE)
    subplot(2, 1, 1)
    
    p = {}
    for t in x:
        if len(t) in p:
            p[len(t)] += 1
        else:
            p[len(t)] = 1
    
    total_num = sum([p[key] for key in sorted(p.keys())])
    maxLenPercentageDiscarded = {}
    
    for i, t in enumerate(sorted(p.keys())):
        maxLenPercentageDiscarded[t] = sum([p[i] for i in array(sorted(p.keys()))[array(sorted(p.keys())) >= t]]) / total_num
    
    # Plotting the sentence length distribution
    plot(sorted(p.keys()), [p[i] for i in sorted(p.keys())])
    title('Sentence length distribution')
    xlabel('Words per tweet')
    ylabel('# Sentences')
    
    subplot(2, 1, 2)
    plot(sorted(p.keys()), [maxLenPercentageDiscarded[i] for i in sorted(p.keys())])
    title('Percentage of discarded sentences')
    xlabel('Sentence length')
    ylabel('Percentage')
    
    subplots_adjust(hspace=0.5)
    show()

def plot_number_of_characters(data_to_plot: Series, cathegory: str = "ALL"):
    data_to_plot.apply(len).hist()
    title(f'Tweet Length for {cathegory} Tweets')
    xlabel('# of characters')
    ylabel('# of Tweets')
    show()

def plot_average_word_length(data_to_plot: Series, cathegory: str = "ALL"):
    avg_word_lengths = data_to_plot.str.split().apply(lambda x : [len(i) for i in x]).map(lambda x: mean(x))
    max_len = int(avg_word_lengths.max())
    print(max_len)
    avg_word_lengths.hist(range=(1, 20), bins=20)
    title(f'Average word Length for {cathegory} Tweets')
    xlabel('Average word length')
    ylabel('# of Tweets')
    xticks(range(1, 21, 1))
    show()

def plot_most_frequent_words(data_to_plot: Series, cathegory: str = "ALL"):
    italian_stopwords = set(stopwords.words('italian'))

    new=data_to_plot.str.split()
    new=new.values.tolist()
    corpus=[word for i in new for word in i]
    counter=Counter(corpus)
    most=counter.most_common()
    x, y=[], []

    for word,count in most[:50]:
        if (word not in italian_stopwords):
            x.append(word)
            y.append(count)
            
    sns.barplot(x=y,y=x)



# !pip install transformers scikit-learn pandas matplotlib --quiet
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def plot_tsne_alberto(
    df: pd.DataFrame,
    text_column: str = "text",
    label_column: str = "iro",
    model_name: str = "m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0",
    model_token: str = "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",
    batch_size: int = 16,
    pca_components: int = 50,
    tsne_components: int = 2,
    random_state: int = 42,
    return_data: bool = False,
    circle: Optional[Tuple[float, float, float]] = None  # (X, Y, R)
):
    """
    Loads AlBERTo, generates [CLS] embeddings for your DataFrame, applies PCA + t-SNE, and plots the result.
    """
    # 1. Extract text and labels
    texts = df[text_column].tolist()
    labels = df[label_column].tolist()
    
    # 2. Load AlBERTo tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=model_token)
    alberto = AutoModel.from_pretrained(model_name, token=model_token)
    alberto.eval()
    
    # 3. Batch inference to get [CLS] embeddings
    all_embeddings = []
    for start_idx in range(0, len(texts), batch_size):
        batch_texts = texts[start_idx:start_idx+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=50)
        
        with torch.no_grad():
            outputs = alberto(**inputs, return_dict=True)
            # shape: (batch_size, seq_len, hidden_dim)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_dim)
        
        all_embeddings.append(cls_embeddings)
    
    embeddings = torch.cat(all_embeddings, dim=0)  # (num_samples, hidden_dim)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # 4. Apply PCA -> TSNE
    pca = PCA(n_components=pca_components, random_state=random_state)
    pca_out = pca.fit_transform(embeddings.numpy())
    
    tsne = TSNE(n_components=tsne_components, random_state=random_state)
    tsne_out = tsne.fit_transform(pca_out)
    
    # 5. Plot in 2D
    if tsne_components == 2:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(tsne_out[:, 0], tsne_out[:, 1], 
                              c=labels, cmap="coolwarm", alpha=0.7)
        plt.title("t-SNE visualization of AlBERTo Embeddings")
        plt.xlabel("t-SNE dimension 1")
        plt.ylabel("t-SNE dimension 2")
        cbar = plt.colorbar(scatter)
        cbar.set_label("Label (0 = Non-ironic, 1 = Ironic)")
        
        # If circle is not None, draw the point and circle
        if circle is not None:
            circle_x, circle_y, radius = circle
            
            # # Plot a green point at (X, Y)
            # plt.scatter([circle_x], [circle_y], c='green', s=80, marker='o', edgecolors='black', label='Circle Center')
            
            # Create a filled, semi-transparent circle
            ax = plt.gca()
            circle_patch = plt.Circle(
                (circle_x, circle_y), radius, 
                color='green', alpha=0.5, fill=True
            )
            ax.add_patch(circle_patch)
            plt.legend()
        
        plt.show()
    else:
        print(f"t-SNE with {tsne_components}D is computed, but you need a custom {tsne_components}D plot or more.")
    
    # 6. Return data if requested
    if return_data:
        return tsne_out, labels


def get_knn_tweets(
    tsne_out: np.ndarray,
    df: pd.DataFrame,
    coords: tuple[float, float],
    k: int = 5,
    text_column: str = "text",
    label_column: str = "iro",
    prediction_column: str = "prediction",
):
    """
    Given a 2D t-SNE output array (tsne_out) and a coordinate (coords),
    find the k nearest tweets in the t-SNE space and print them.
    """
    # Ensure the shapes align
    assert len(tsne_out) == len(df), (
        f"Mismatch: tsne_out has {len(tsne_out)} rows, but df has {len(df)}"
    )
    
    # Calculate Euclidean distances from coords
    distances = np.sqrt(np.sum((tsne_out - np.array(coords))**2, axis=1))
    
    # Sort by distance and get indices of the k closest points
    nearest_indices = np.argsort(distances)[:k]
    
    # Print the tweets with the smallest distance
    print(f"\nThe {k} nearest tweets to {coords} in t-SNE space:")
    for idx in nearest_indices:
        text_val = df[text_column].iloc[idx]
        label_val = df[label_column].iloc[idx]
        pred_val = df[prediction_column].iloc[idx]
        
        print(f"---\nTweet:\n{text_val}\nLabel:\n{label_val}\nPredicted:\n{pred_val}\n")
    
    return df.iloc[nearest_indices]

import numpy as np
import pandas as pd

def get_tweets_in_circle(
    tsne_out: np.ndarray,
    df: pd.DataFrame,
    x: float,
    y: float,
    r: float,
    text_column: str = "text",
    label_column: str = "iro",
    prediction_column: str = "prediction",
    print_tweets: bool = True
) -> pd.DataFrame:
    """
    Given a 2D t-SNE output array (tsne_out) aligned with the rows in `df`,
    returns the tweets that lie within the circle of center (x, y) and radius r.
    """
    # Ensure matching shapes
    assert len(tsne_out) == len(df), (
        f"Mismatch: tsne_out has {len(tsne_out)} rows, but df has {len(df)}"
    )
    
    # Compute distance of each point in tsne_out to center (x, y)
    center = np.array([x, y])
    distances = np.sqrt(np.sum((tsne_out - center)**2, axis=1))
    
    # Find rows within the given radius
    inside_mask = distances <= r
    inside_indices = np.where(inside_mask)[0]
    
    # Retrieve corresponding DataFrame rows
    inside_df = df.iloc[inside_indices].copy()
    
    if print_tweets:
        print(f"\nTweets inside the circle (center=({x}, {y}), radius={r}):")
        for idx in inside_indices:
            row = df.iloc[idx]
            text_val = row[text_column]
            label_val = row[label_column]
            pred_val = row[prediction_column]
            
            print(f"---\nTweet:\n{text_val}\nLabel:\n{label_val}\nPredicted:\n{pred_val}\n")
    
    return inside_df

