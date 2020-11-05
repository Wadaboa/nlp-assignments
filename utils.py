import os
import re
import tarfile
import urllib.request

import pandas as pd
import gensim
import gensim.downloader as gloader
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.util import mark_negation
from nltk.tokenize import word_tokenize


# Download NLTK packages
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")


# NLTK utils
STEMMER = nltk.porter.PorterStemmer()
LEMMATIZER = nltk.wordnet.WordNetLemmatizer()
EN_STOPWORDS = set(stopwords.words("english"))

# Regular expressions
DATE_RE = r"\d{1,2}[-\/\.]\d{1,2}[-\/\.]\d{2,4}"
FLOAT_RE = r"(\d*\,)?\d+.\d*"
INT_RE = r"(?<=\s)\d+(?=\s)"
BRACKETS_RE = r"\[[^]]*\]"
HTML_RE = r"<.*?>"
PUNCTUATION_RE = r"[^\w{w}\s\{<>}]+"


class IMDBDataset():
    '''
    IMDB movie review dataset wrapper
    '''
    
    NAME = "imdb"
    URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    
    def __init__(self, dataset_folder=None, dataframe_path=None, preprocessor=None):
        # Create directories
        self.dataset_folder = (
            dataset_folder or 
            os.path.join(os.getcwd(), "datasets", self.NAME)
        )
        self.original_dataset_folder = os.path.join(self.dataset_folder, "original")
        self.original_dataset_path = os.path.join(self.original_dataset_folder, "movies.tar.gz")
        self.dataframe_path = dataframe_path or os.path.join(self.dataset_folder, "dataframe", self.NAME + ".pkl")      
        
        # Download the dataset, preprocess it and load the dataframe
        self._create_dirs()
        self._download()
        self.dataframe = self._prepare()
        
        if preprocessor is not None:
            self.preprocess(preprocessor)

    def _create_dirs(self):
        '''
        Create dataset folders, if they do not exist yet
        '''
        # Create the dataset folder
        if not os.path.exists(self.original_dataset_folder):
            os.makedirs(self.original_dataset_folder)

        # Create the dataframe folder
        if not os.path.exists(self.dataframe_path):
            os.makedirs(os.path.dirname(self.dataframe_path))

    def _download(self):
        '''
        Download and extract the dataset, if it was not already done
        '''
        if not os.path.exists(self.original_dataset_path):
            urllib.request.urlretrieve(self.URL, self.original_dataset_path)
            tar = tarfile.open(self.original_dataset_path)
            tar.extractall(self.original_dataset_folder)
            tar.close()
            return True

        return False

    def _prepare(self):
        '''
        Preprocess the original dataset and wrap it in pandas dataframe,
        if necessary (otherwise load the pickled file)
        '''
        if not os.path.exists(self.dataframe_path):
            dataframe_rows = []
            for split in ["train", "test"]:
                for sentiment in ["pos", "neg"]:
                    folder = os.path.join(self.original_dataset_folder, "aclImdb", split, sentiment)
                    for filename in os.listdir(folder):
                        file_path = os.path.join(folder, filename)
                        try:
                            if os.path.isfile(file_path):
                                with open(file_path, mode="r", encoding="utf-8") as text_file:
                                    # Extract info
                                    text = text_file.read()
                                    score = filename.split("_")[1].split(".")[0]
                                    file_id = filename.split("_")[0]

                                    # Compute sentiment
                                    num_sentiment = (
                                        1 if sentiment == "pos"
                                        else 0 if sentiment == "neg"
                                        else -1
                                    )

                                    # Create single dataframe row
                                    dataframe_row = {
                                        "file_id": file_id,
                                        "score": score,
                                        "sentiment": num_sentiment,
                                        "split": split,
                                        "text": text,
                                    }
                                    dataframe_rows.append(dataframe_row)
                        except Exception as e:
                            return None

            # Transform the list of rows in a proper dataframe
            dataframe = pd.DataFrame(dataframe_rows)
            dataframe_cols = ["file_id", "score", "sentiment", "split", "text"]
            dataframe = dataframe[dataframe_cols]
            dataframe.to_pickle(self.dataframe_path)
            return dataframe
        
        return pd.read_pickle(self.dataframe_path)
    
    def preprocess(self, preprocessor):
        '''
        Apply standard preprocessing to every movie review
        '''
        self.dataframe["text"] = self.dataframe["text"].apply(preprocessor)
        
    

def stem_text(text):
    return [STEMMER.stem(word) for word in text]


def lemmatize_text(text):
    return [LEMMATIZER.lemmatize(word) for word in text]


def preprocess_text(
    text,
    regexes=True,
    start_end_symbols=True,
    mark_neg=False,
    remove_punct=False,
    remove_stopwords=False,
    min_chars=None,
    root=None,
):
    # Strip trailing spaces and remove newlines
    text = text.strip().replace("\n", " ").replace("\r", " ")
    # Convert to lowercase
    text = text.lower()
    # Apply regular expressions
    if regexes:
        # Remove HTML tags
        text = re.sub(HTML_RE, "", text)
        # Remove text in square brackets
        text = re.sub(BRACKETS_RE, "", text)
        # Remove dates
        text = re.sub(DATE_RE, "", text)
        # Remove floating numbers
        text = re.sub(FLOAT_RE, "", text)
    # Mark negation
    if mark_neg:
        text = " ".join(mark_negation(text.split(), double_neg_flip=False))
        EN_STOPWORDS.add("_NEG")
    # Remove punctuation
    if remove_punct:
        text = re.sub(PUNCTUATION_RE, "", text)
    # Leave single whitespace
    text = text.split()
    # Remove words with less than `n` chars
    if min_chars is not None and isinstance(min_chars, int):
        text = [word for word in text if len(word) >= min_chars]
    # Remove stopwords
    if remove_stopwords:
        text = [word for word in text if word not in EN_STOPWORDS]
    # Perform stemming/lemmatization
    if root == "stem":
        text = stem_text(text)
    elif root == "lemmatize":
        text = lemmatize_text(text)
    # Add start/end symbols
    if start_end_symbols:
        text = ["<s>"] + text + ["</s>"]
    # Return the text as a string
    return " ".join(text)


def load_embedding_model(model_type, embedding_dimension=50):
    """
    Loads a pre-trained word embedding model via gensim library
    """
    # Find the correct embedding model name
    download_path = ""
    if model_type.strip().lower() == 'word2vec':
        download_path = "word2vec-google-news-300"
    elif model_type.strip().lower() == 'glove':
        download_path = f"glove-wiki-gigaword-{embedding_dimension}"
    else:
        raise AttributeError("Unsupported embedding model type (choose from {word2vec, glove})")

    # Check download
    try:
        emb_model = gloader.load(download_path)
    except ValueError as e:
        print("Invalid embedding model name. Check the embedding dimension:")
        print("Word2Vec: {300}")
        print("GloVe: {50, 100, 200, 300}")
        raise e

    return emb_model