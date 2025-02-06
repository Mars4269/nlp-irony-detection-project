import string
import numpy as np
import regex as re
import pandas as pd
from sklearn.model_selection import train_test_split

def set_seeds(seed_value):
    np.random.seed(seed_value)

# def _replace_abbreviations(text, abbrv_dict = ABBRV_DICT):
#     for key, abbr_list in abbrv_dict.items():
#         for abbr in abbr_list:
#             # abbr_pattern = re.escape(abbr)
#             text = re.sub(r'\b' + abbr + r'\b', key, text)
#     return text

# def _remove_punctuation(text):
#     punctuation = list(string.punctuation)
#     punctuation.remove("'")
#     return "".join([word for word in text if word not in punctuation]).strip()

# def collapse_dots_to_ellipsis(text):
#     # Replace any sequence of four or more dots with an ellipsis (...)
#     return re.sub(r'\.{4,}', '...', text)

# import re
# import pandas as pd

# def remove_punctuation(text):
#     # Dummy implementation for the sake of completeness
#     return text

# def lemmatize_text(text):
#     return ' '.join([lemmatize(word, "italian") for word in text.split()])

# def preprocessing(
#     data, 
#     #preprocessing_type: Optional[str] = "basic",
#     translate_emoticons=False,
#     lowercase=False, 
#     remove_urls=False, 
#     remove_urls_placeholder = '',
#     remove_video_placeholders=False,
#     remove_mentions=False, 
#     remove_html_entities=False, 
#     remove_html_entities_placeholder = '',
#     remove_retweet_markers=False, 
#     replace_abbreviations=False, 
#     remove_duplicates_at_end=False,
#     translate_emojis=False,
#     replace_by_space=False,
#     keep_only_good_symbols=False,
#     remove_punctuation=False, 
#     normalize_whitespace=False, 
#     lemmatize=False, 
#     remove_stopwords=False
#     ):    
#     """
#     Preprocesses the text data in a DataFrame by performing several cleaning steps.

#     :param data: A pandas DataFrame containing a column 'text' with the text data to be preprocessed.
#     :type data: pd.DataFrame
#     :param lowercase: Whether to convert text to lowercase.
#     :type lowercase: bool
#     :param remove_urls: Whether to remove URLs from the text.
#     :type remove_urls: bool
#     :param remove_mentions: Whether to remove mentions (e.g., @username) from the text.
#     :type remove_mentions: bool
#     :param remove_html_entities: Whether to remove HTML entities (e.g., &, <, >) from the text.
#     :type remove_html_entities: bool
#     :param remove_retweet_markers: Whether to remove retweet markers (e.g., #rt, rt) from the text.
#     :type remove_retweet_markers: bool
#     :param replace_abbreviations: Whether to replace abbreviations with their full forms using the `replace_abbreviations` function.
#     :type replace_abbreviations: bool
#     :param remove_punctuation: Whether to remove punctuation using the `remove_punctuation` function.
#     :type remove_punctuation: bool
#     :param normalize_whitespace: Whether to convert multiple consecutive whitespace characters into a single space.
#     :type normalize_whitespace: bool
#     :param lemmatize: Whether to lemmatize the text.
#     :type lemmatize: bool
#     :param remove_stopwords: Whether to remove stop words from the text.
#     :type remove_stopwords: bool

#     :returns: The DataFrame with the preprocessed text data.
#     :rtype: pd.DataFrame

#     :example:
#     >>> import pandas as pd
#     >>> data = pd.DataFrame(
#         {'text': [
#             'Check this out! https://example.com @user #rt',
#             'Another tweet & more text.'
#             ]})

#     >>> clean_data = preprocessing(data, lowercase=True, remove_urls=True, remove_mentions=True, remove_html_entities=True, remove_retweet_markers=True, replace_abbreviations=True, remove_punctuation=True, normalize_whitespace=True, lemmatize=True, remove_stopwords=True)
#     >>> print(clean_data)

#     0  check this out!
#     1  another tweet more text.
#     """

#     # allowed_types = ["basic", "GRU", "BERT"]

#     # if preprocessing_type not in allowed_types:
#     #     raise ValueError(f"Error, type of preprocessing must be one of {allowed_types}")
#     #     return
    

#     lemmatizer = WordNetLemmatizer()
#     stop_words = set(stopwords.words('italian'))

#     # Remove apostrophes
#     # If we do not do this word_tokenize considers things like "c'è" as a single token
#     # data['text'] = data['text'].apply(lambda x: x.replace("'", " ")) 

#     if translate_emoticons:
#         data['text'] = data['text'].apply(lambda x: emoticon.emoticon(x, 'it'))

#     if lowercase:
#         data['text'] = data['text'].str.lower()

#     if remove_urls:
#         data['text'] = data['text'].apply(lambda x: re.sub(r'https?v?:\/\/\S+', remove_urls_placeholder, x))

#     if remove_video_placeholders:
#         data['text'] =  data['text'].apply(lambda x: re.sub(r"\[video\]", '', x))

#     if remove_mentions:
#         data['text'] = data['text'].apply(lambda x: re.sub(r'@\S+\s?', 'nome', x))
#         # data['text'] = data['text'].apply(lambda x: re.sub(r'(nome)+', 'nome', x))

#     if remove_html_entities:
#         data['text'] = data['text'].apply(lambda x: re.sub(r'&[a-z]+;', remove_html_entities_placeholder, x))

#     if remove_retweet_markers:
#         data['text'] = data['text'].apply(lambda x: re.sub(r'(\s#{0,1}[Rr][Tt])|(^#{0,1}[Rr][Tt]\s)', '', x))

#     if replace_abbreviations:
#         data['text'] = data['text'].apply(lambda x: _replace_abbreviations(x))

#     if remove_duplicates_at_end:
#         data['text'] = data['text'].apply(lambda x: re.sub(r'(\w)(\1+)(?=\s|$)', lambda m: m.group(0) if m.group(0) in {"!", "?", "..."} else m.group(1), x))

#     if translate_emojis:
#         data['text'] = data['text'].apply(lambda x: demojize(x, language='it'))

#     if replace_by_space:
#         data['text'] = data['text'].apply(lambda x: re.sub(r'[/(){}\[\]\|@,;_]', ' ', x))

#     if keep_only_good_symbols:
#         data['text'] = data['text'].apply(lambda x: re.sub(r'[^0-9a-zéèòàù #!?.{2,3}]', ' ', x))

#     if remove_punctuation:
#         data['text'] = data['text'].apply(lambda x: _remove_punctuation(x))

#     if normalize_whitespace:
#         data['text'] = data['text'].apply(lambda x: re.sub(r'\s{2,}', ' ', x))

#     if lemmatize:
#         data['text'] = data['text'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(x)]))

#     if remove_stopwords:
#         data['text'] = data['text'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in stop_words]))

#     data['text'] = data['text'].str.strip()

#     return data

def contains_mark(text: str, mark: str):
    return mark in text

# def contains_smileys(text):
#     smileys = [':)', '=)', ':(', ':-)', ':-(', ':D', 'D:', ';)', ';(', ';-)', 'D-:', ':-D', ':))', ':((', '-.-', 'u.u', 'U.U', ':\')', 'c:', ' :c']
#     return any(smiley in text for smiley in smileys)

# def contains_emoji(text):
#     # from https://www.kaggle.com/code/yash3056/data-set-cleaning
#     emoji_pattern = re.compile(
#         "["
#         "\U0001F600-\U0001F64F"  # emoticons
#         "\U0001F300-\U0001F5FF"  # symbols & pictographs
#         "\U0001F680-\U0001F6FF"  # transport & map symbols
#         "\U0001F1E0-\U0001F1FF"  # flags (iOS)
#         "\U00002702-\U000027B0"  # other miscellaneous symbols
#         "\U000024C2-\U0001F251"
#         "]+",
#         flags=re.UNICODE,
#     )
#     return bool(emoji_pattern.search(text))

def show_mark_statistics(df, mark_list):
    mark_df = df.copy(deep=True)
    results_df = pd.DataFrame(columns=[0, 1], index=mark_list)

    for mark in mark_list:
        mark_df[mark] = mark_df['text'].apply(contains_mark, **{"mark": mark})

        # Group by 'iro' column and calculate total tweets and tweets containing the mark
        total_tweets = mark_df.groupby('iro').size()  # Total number of tweets for each group
        mark_tweet_counts = mark_df.groupby('iro')[mark].sum()  # Tweets containing the mark

        # Calculate the percentage of tweets containing the mark
        percentage_mark_tweets = (mark_tweet_counts / total_tweets) * 100
        # print(percentage_mark_tweets)

        results_df.loc[mark] = (f"{percentage_mark_tweets[0]:.2f}", f"{percentage_mark_tweets[1]:.2f}") 

    return results_df

# def load_datasets(folder: str, seed: int): # , preprocessing: Optional[str] = None
#     train_set = pd.read_csv(f"{folder}/training_set_sentipolc16.csv", sep=",")
#     test_set = pd.read_csv(f"{folder}/test_set_sentipolc16_gold2000.csv", sep=",",names=list(train_set.columns))

#     for subset in (train_set, test_set):
#         subset.drop("idtwitter", axis=1, inplace=True)
#         subset.drop("subj", axis=1, inplace=True)
#         subset.drop("opos", axis=1, inplace=True)
#         subset.drop("oneg", axis=1, inplace=True)
#         subset.drop("lpos", axis=1, inplace=True)
#         subset.drop("lneg", axis=1, inplace=True)
#         subset.drop("top", axis=1, inplace=True)

#     test_set, val_set = train_test_split(test_set, test_size=0.5, shuffle=False, random_state=seed)

#     return train_set, val_set, test_set

def load_datasets(folder: str):
    train = pd.read_csv(f"{folder}/train.csv")
    val = pd.read_csv(f"{folder}/val.csv")
    test = pd.read_csv(f"{folder}/test.csv")
    return train, val, test

import vocabulary as v
def intersection_tokens(df1, df2):
    tokenizer = v.BaseTokenizer()
    tokens_df1 = {
        token 
        for text in df1['text'] 
        for token in tokenizer.tokenize(text)
    }
    tokens_df2 = {
        token 
        for text in df2['text'] 
        for token in tokenizer.tokenize(text)
    }
    return list(tokens_df1 & tokens_df2)
