import os
import regex as re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from simplemma import lemmatize
from emoji import demojize
from tqdm.notebook import tqdm
from typing import List, Set, Optional, Union
import pandas as pd
from nltk.tokenize import TweetTokenizer
from mtranslate import translate
import csv
from copy import deepcopy

# Uncomment only if error is raised
# download("stopwords")
# download("punkt_tab")
# download('wordnet')

ABBRV_DICT = {'comunque' : ['cmq'],
                'perché' : ['xk', 'xkè', 'xké', 'xke'],
                'per' : ['x'],
                'non': ['nn'],
                'però' : ['xo'],
                'ti voglio bene' : ['tvb', 'tvtb'],
                'capito' : ['cpt'],
                'che' : ['ke'],
                'niente' : ['nnt'],
                'messaggio' : ['msg'],
                'per favore' : ['plz', 'pls'],
                'tutto' : ['tt'],
                'fratello' : ['frà'],
              }

# ----------------- Public -----------------
def emoji_and_emoticons_preprocessing(
    data: pd.DataFrame, 
    translate_emoticons: bool = True,
    translate_emojis: bool = True,
    verbose: bool = False,
    parent_dir: Optional[str] = None
    ):
    data = deepcopy(data)
    if verbose:
        print("-------Original Text-------")
        for i, row in data[:min(10, len(data))].iterrows():
            print(f"Sentence {i+1}: {row['text']}")
        print("----")

    if translate_emoticons:
        with tqdm(total = len(data["text"])) as pbar:
            data['text'] = data['text'].apply(lambda x: emoticon(x, 'it', parent_dir=parent_dir, pbar=pbar))
        if verbose:
            print_step("translate_emoticons", data)
        
    if translate_emojis:
        data['text'] = data['text'].apply(lambda x: demojize_and_strip(x))
        if verbose:
            print_step("translate_emojis", data)
    return data
def fundamental_preprocessing(
    data: pd.DataFrame, 
    lowercase: bool = False, 
    remove_urls: bool = False, 
    remove_video_placeholders: bool = False,
    remove_mentions: bool = False, 
    remove_html_entities: bool = False, 
    remove_retweet_markers: bool = False, 
    replace_abbreviations: bool = False, 
    remove_duplicates_at_end: bool = False,
    replace_by_space: bool = False,
    keep_only_good_symbols: bool = False,
    remove_punctuation: bool = False, 
    normalize_whitespace: bool = False, 
    lemmatize: bool = False, 
    remove_stopwords: bool = False,
    verbose: bool = False,
):
    """
    Preprocesses the text data in a DataFrame by performing several cleaning steps.
    After each step, prints out how the text looks to show the transformation sequence.
    """
    data = deepcopy(data)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('italian'))

    if verbose:
        print("-------Original Text-------")
        for i, row in data[:min(10, len(data))].iterrows():
            print(f"Sentence {i+1}: {row['text']}")
        print("----")

    # 1) Lowercase
    if lowercase:
        data['text'] = data['text'].str.lower()
        if verbose:
            print_step("lowercase", data)

    # 2) Remove URLs
    if remove_urls:
        data['text'] = data['text'].apply(lambda x: re.sub(r'https?v?:\/\/\S+', '<url>', x))
        if verbose:
            print_step("remove_urls", data)

    # 3) Remove [video] placeholders
    if remove_video_placeholders:
        data['text'] =  data['text'].apply(lambda x: re.sub(r"\[video\]", '', x))
        if verbose:
            print_step("remove_video_placeholders", data)

    # 4) Remove mentions (e.g. @Mario)
    if remove_mentions:
        data['text'] = data['text'].apply(lambda x: re.sub(r'@\S+', '<mention>', x)) # \s?
        if verbose:
            print_step("remove_mentions", data)

    # 5) Remove HTML entities (e.g. &amp;)
    if remove_html_entities:
        data['text'] = data['text'].apply(lambda x: re.sub(r'&[a-z]+;', '', x))
        if verbose:
            print_step("remove_html_entities", data)

    # 6) Remove retweet markers (#rt, rt)
    if remove_retweet_markers:
        data['text'] = data['text'].apply(lambda x: re.sub(r'(\s#{0,1}[Rr][Tt])|(^#{0,1}[Rr][Tt]\s)', '', x))
        if verbose:
            print_step("remove_retweet_markers", data)

    # 7) Replace abbreviations
    if replace_abbreviations:
        data['text'] = data['text'].apply(lambda x: _replace_abbreviations(x))
        if verbose:
            print_step("replace_abbreviations", data)

    # 8) Remove duplicated characters at end (e.g. "helloooo" -> "hello")
    #    but keep sequences like "!!!"
    if remove_duplicates_at_end:
        data['text'] = data['text'].apply(
            lambda x: re.sub(
                r'(\w)(\1+)(?=\s|$)', 
                lambda m: m.group(0) if m.group(0) in {"!", "?", "..."} else m.group(1), 
                x
            )
        )
        if verbose:
            print_step("remove_duplicates_at_end", data)

    # 9) Replace certain symbols by space
    if replace_by_space:
        data['text'] = data['text'].apply(lambda x: re.sub(r'[/(){}\[\]\|@]', ' ', x))
        if verbose:
            print_step("replace_by_space", data)

    # 10) Keep only “good” symbols
    if keep_only_good_symbols: 
        # This is just an example that might keep [0-9a-zéèòàù #!?.] plus 2-3 repeated punctuation
        data['text'] = data['text'].apply(lambda x: re.sub(r'[^0-9a-zéèòàù <>_#!\?\.]{2,3}', ' ', x))
        if verbose:
            print_step("keep_only_good_symbols", data)

    # 11) Remove punctuation
    if remove_punctuation:
        data['text'] = data['text'].apply(lambda x: _remove_punctuation(x))
        if verbose:
            print_step("remove_punctuation", data)

    # 12) Normalize whitespace (replace multiple spaces with single space)
    if normalize_whitespace:
        data['text'] = data['text'].apply(lambda x: re.sub(r'\s{2,}', ' ', x))
        if verbose:
            print_step("normalize_whitespace", data)

    # 13) Lemmatize
    if lemmatize:
        data['text'] = data['text'].apply(
            lambda x: ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(x)])
        )
        if verbose:
            print_step("lemmatize", data)

    # 14) Remove stopwords
    if remove_stopwords:
        data['text'] = data['text'].apply(
            lambda x: ' '.join([word for word in word_tokenize(x) if word not in stop_words])
        )
        if verbose:
            print_step("remove_stopwords", data)

    # Final strip
    data['text'] = data['text'].str.strip()

    if verbose:
        print("\n-------Final Preprocessed Text-------")
        for i, row in data[:min(10, len(data))].iterrows():
            print(f"Sentence {i+1}: {row['text']}")
        print("----")

    return data

def old_segment_hashtags(tokens: List[str] | str, dictionary: set, pbar: tqdm = None) -> list:
    # Iterate over tokens and process hashtags
    local_tokens = deepcopy(tokens)
    out = []
    i = 0
    while i < len(local_tokens) - 1:
        # Look for hashtags
        if local_tokens[i] == '#':
            # Segment
            seg = segment_hashtag_content(local_tokens[i+1], dictionary)
            local_tokens.pop(i+1)
            if seg:
                # append segmentation
                out.append("<hashtag>") 
                out.extend(seg)
                out.append("</hashtag>") 
            else:
                # do not add hashtag if segmentation is not found
                out.append("<hashtag>") 
                out.append("</hashtag>") 
        else:
            # append all other tokens normally
            out.append(local_tokens[i])
        i += 1

    if pbar:
        pbar.update(1)

    return out

def segment_hashtags(
    dictionary: Set[str],
    pbar: Optional[tqdm] = None,
    tokens: Optional[List[str]] = None,
    text: Optional[List[str]] = None,
) -> Union[List[str], str]:
    """
    This function can handle both:
      - A list of tokens
      - A string (untokenized text)

    If 'tokens_or_text' is a list, we scan for a '#' token and segment the 
    immediately following token. The result is returned as a list of tokens 
    with the segmented hashtag enclosed in start_of_hashtag/end_of_hashtag.

    If 'tokens_or_text' is a string, we scan for all occurrences of '#' 
    (without tokenizing the entire text). For each one, we segment it 
    with the same logic and replace the original hashtag with:
       start_of_hashtag token1 token2 ... end_of_hashtag
    or, if no segmentation is found, with
       start_of_hashtag end_of_hashtag

    Args:
        tokens_or_text (Union[List[str], str]): The input to process.
        dictionary (Set[str]): A set of known words for hashtag segmentation.
        pbar (tqdm, optional): A progress bar instance (optional).
    
    Returns:
        Union[List[str], str]: 
            - If input is a list, returns a list of tokens.
            - If input is a string, returns the modified string.
    """
    if (tokens is None) == (text is None):
        raise ValueError("Exactly one of 'tokens' or 'text' must be provided.")
    
    # --- CASE 1: LIST OF TOKENS ---
    if tokens is not None:
        local_tokens = deepcopy(tokens)
        out = []
        i = 0
        while i < len(local_tokens) - 1:
            if local_tokens[i] == '#':
                # Segment the next token
                seg = segment_hashtag_content(local_tokens[i + 1], dictionary)
                # Remove next token from the list
                local_tokens.pop(i + 1)
                if seg:
                    # append segmentation
                    out.append("<hashtag>") 
                    out.extend(seg)
                    out.append("</hashtag>") 
                else:
                    # do not add hashtag if segmentation is not found
                    out.append("<hashtag>") 
                    out.append("</hashtag>") 
            else:
                # Normal token, just copy it
                out.append(local_tokens[i])
            i += 1

        # If there's a last token not processed by the loop, add it
        if i < len(local_tokens):
            out.append(local_tokens[i])

        # pbar update
        if pbar:
            pbar.update(1)

        return out

    # --- CASE 2: UNTOKENIZED STRING ---
    else:
        local_text = deepcopy(text)

        # We'll use a regex to find occurrences of #something.
        # For example: r'#[A-Za-zÀ-ÖØ-öø-ÿ0-9_]+'
        # You can adjust to match your scenario more accurately (hashtags in other alphabets, etc.).
        pattern = re.compile(r'#[^\s]+')

        def replace_hashtag(m: re.Match) -> str:
            """Callback to handle each hashtag match with segmentation."""
            hashtag_text = m.group(0)  # e.g. '#labuonascuola'
            seg = segment_hashtag_content(hashtag_text, dictionary)
            if seg:
                return "<hashtag> " + " ".join(seg) + " </hashtag>"
            else:
                return "<hashtag> </hashtag>"

        # Perform a substitution using the callback
        new_text = pattern.sub(replace_hashtag, local_text)

        # pbar update
        if pbar:
            pbar.update(1)

        return new_text

# ----------------- Private Emoji and emoticons -----------------

def emoticon(text, *argv, parent_dir, pbar):
    # Build the path to the CSV file
    dir_name = os.path.join(parent_dir, "data") if parent_dir else "data"
    file_path = os.path.join(dir_name, "emoticon.csv")

    # Read the CSV once, building a dictionary for emoticon lookups
    emoticon_map = {}
    with open(file_path, encoding="UTF-8") as csv_file:
        reader = csv.reader(csv_file, delimiter="\t")
        for row in reader:
            # row[1] is the raw emoticon, row[2] is the replacement or meaning
            emoticon_map[row[1].strip()] = row[2]

    # Tokenize the input text
    tokenizer = TweetTokenizer()
    tokens = tokenizer.tokenize(text)

    # Check if a language argument was provided
    lang = argv[0] if argv else None
    for token in tokens:
        # If the token is in our emoticon map, replace it
        if token in emoticon_map:
            replaced = emoticon_map[token]
            # If a language was specified, translate the replacement
            if lang:
                replaced = translate(replaced, lang)

            text = text.replace(token, f"{replaced} ")

    # transform it: :faccina_felice: -> faccina felice
    pattern = r":[a-zA-Z0-9]+(_[a-zA-Z0-9]+)*:"
    def replace_with_spaces(match: re.Match):
        return f'{match.group().replace(":", "").replace("_", " ")} '
    pbar.update(1)
    return re.sub(pattern, replace_with_spaces, text)

def demojize_and_strip(text: str, language: str = "it"):  
    # demojize
    demojized_text = demojize(text, language=language)

    # transform it: :faccina_felice: -> faccina felice
    pattern = r":[a-zA-Z0-9]+(_[a-zA-Z0-9]+)*:"
    def replace_with_spaces(match: re.Match):
        return f'{match.group().replace(":", "").replace("_", " ")} '

    result = re.sub(pattern, replace_with_spaces, demojized_text)
    return result

def print_step(step_name, df: pd.DataFrame):
    """ Helper function to print the DataFrame's text after a given step. """
    print(f"\n--- After {step_name} ---")
    for i, row in df.iloc[:min(10, len(df))].iterrows():
        print(f"Sentence {i+1}: {row['text']}")
    print("----")

# ----------------- Private Hashtag -----------------

def segment_hashtag_content(
    hashtag: str,
    dictionary: Set[str],
    lowercase: bool = True
) -> List[str]:
    """
    Segments an Italian hashtag into its constituent words using a 
    dictionary-based dynamic programming approach. Then applies the logic:
      1) Prefer a segmentation of length >= 2 (the shortest among those).
      2) If none with length >= 2, pick one of length 1 if it exists.

    Args:
        hashtag (str): The hashtag string, e.g. '#labuonascuola'.
        dictionary (Set[str]): A set of known Italian words.
        lowercase (bool, optional): Whether to convert the hashtag to lowercase
            before segmentation. Defaults to True.

    Returns:
        List[str]: The chosen segmentation based on the above logic.
    """
    # 1. Remove leading '#' if present
    if hashtag.startswith('#'):
        hashtag = hashtag[1:]
    
    # 2. Normalize (lowercase)
    if lowercase:
        hashtag = hashtag.lower()

    n = len(hashtag)
    if hashtag in dictionary:
        return [hashtag]
    
    # dp[i] = list of all possible segmentations (each segmentation is a list of tokens)
    # that cover hashtag[:i]
    dp = [[] for _ in range(n + 1)]
    dp[0] = [[]]  # There's exactly one way to segment an empty string: an empty list of tokens

    for i in range(1, n + 1):
        for j in range(i):
            # For each segmentation that covers hashtag[:j], try to extend with hashtag[j:i]
            for seg in dp[j]:
                substring = hashtag[j:i]
                if substring in dictionary:
                    # Found a valid extension
                    dp[i].append(seg + [substring])

    # dp[n] now contains all possible segmentations for the entire hashtag
    all_segmentations = dp[n]

    if not all_segmentations:
        # No segmentation found at all
        return
    
    # Separate segmentations with length >= 2 vs. length == 1
    multi_word_segmentations = [s for s in all_segmentations if len(s) >= 2]
    single_word_segmentations = [s for s in all_segmentations if len(s) == 1]

    # 1) If there's at least one segmentation with length >= 2, pick the one with the fewest tokens
    if multi_word_segmentations:
        best_multi = min(multi_word_segmentations, key=len)
        return best_multi
    
    # 2) Otherwise, if there's a single-word segmentation,a return that
    if single_word_segmentations:
        # Just pick the first single-word segmentation (they should all be the same word if there's more than one)
        return single_word_segmentations[0]

    # 3) Otherwise, no segmentation meets the criteria
    return

# ----------------- Private Fundamental -----------------

def _replace_abbreviations(text, abbrv_dict = ABBRV_DICT):
    for key, abbr_list in abbrv_dict.items():
        for abbr in abbr_list:
            # abbr_pattern = re.escape(abbr)
            text = re.sub(r'\b' + abbr + r'\b', key, text)
    return text

def _remove_punctuation(text):
    punctuation = list(string.punctuation)
    punctuation.remove("'")
    punctuation.remove("#")
    return "".join([word for word in text if word not in punctuation]).strip()

def collapse_dots_to_ellipsis(text):
    # Replace any sequence of two or more dots with an ellipsis (...)
    return re.sub(r'\.{2,}', '...', text)

def remove_punctuation(text):
    # Dummy implementation for the sake of completeness
    return text

def lemmatize_text(text):
    return ' '.join([lemmatize(word, "italian") for word in text.split()])


if __name__ == "__main__":
    tweets_data = {
        "text": [
            "Ciao a tutti! 😊 Che giornata meravigliosa oggi. :') #sole #felicità",
            "RT @amico: Non ci credo che hai detto una cosa così! 😱😂😂 https://example.com",
            "Stasera cinema con gli amici [video] non vedo l'ora! <3<3<3 #cinema #divertimento",
            "Xkè nn rispondi ai msg? Tvb ma sei insopportabile... 🙄",
            "@mario Ho visto il tuo post, davvero interessante! Complimenti :D 👏👏",
            "Ma è vero che ci sono sconti del 50% oggi? 🤔 plz fammi sapere #shopping",
            "Un brindisi a tutti i miei fratelli! Cin cin! 🥂🎉",
            "Xo devo andare adesso, nn ho più tempo per parlare! Cpt? 😅",
            "Stamattina ho letto un articolo interessante su &amp; questo argomento, davvero illuminante.",
            "Wow, il tramonto stasera è stato spettacolare...! 🌅 #meraviglia"
        ]
    }

    # Convert to DataFrame
    tweets_df = pd.DataFrame(tweets_data)

    emoji_and_emoticons_preprocessing(data=tweets_df, verbose=True)
    fundamental_preprocessing(
        data=tweets_df, 
        lowercase=True, 
        remove_urls=True, 
        remove_video_placeholders=True,
        remove_mentions=True, 
        remove_html_entities=True, 
        remove_retweet_markers=True, 
        replace_abbreviations=True, 
        remove_duplicates_at_end=True,
        replace_by_space=True,
        keep_only_good_symbols=True,
        remove_punctuation=False, 
        normalize_whitespace=True, 
        lemmatize=False, 
        remove_stopwords=False,
        verbose=True
     )

if __name__ == "__main__": 

    from vocabulary import BaseTokenizer

    with open("vocab/ALBERTo_vocab.txt", "r", encoding="utf-8") as f:
        vocab = set(line.strip() for line in f)

    segmented = segment_hashtags(
    vocab,
    tokens = BaseTokenizer().tokenize("Ciao a tutti! 😊 Che giornata meravigliosa oggi. :') #sole #felicità #senocagnolona"),
    )

    print(segmented)