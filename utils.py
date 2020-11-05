import os
import re
import tarfile
import urllib.request

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gensim
import gensim.downloader as gloader
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.util import mark_negation
from nltk.tokenize import word_tokenize
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD


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
    '''
    Highly-customizable text preprocessing function
    '''
    
    def stem_text(text):
        return [STEMMER.stem(word) for word in text]

    def lemmatize_text(text):
        return [LEMMATIZER.lemmatize(word) for word in text]
    
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
        
    def get_portion(self, amount=500, seed=0):
        '''
        Returns a random subset of the whole dataframe
        '''
        np.random.seed(seed)
        random_indexes = np.random.choice(
            np.arange(self.dataframe.shape[0]), size=amount, replace=False
        )
        return self.dataframe.iloc[random_indexes]
        
    
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


def visualize_embeddings(embeddings, word_annotations=None, word_to_idx=None):
    """
    Plots the given reduced word embeddings (2D). Users can highlight specific 
    words (`word_annotations` list) in order to better analyze the 
    effectiveness of the embedding method.
    """
    fig = plt.figure(figsize=(12, 8), dpi=100)
    ax = plt.gca()
    
    # Annotate given words
    other_embeddings = embeddings
    if word_annotations is not None and word_to_idx is not None:
        word_indexes = []
        for word in word_annotations:
            word_index = word_to_idx[word]
            word_indexes.append(word_index)
        word_indexes = np.array(word_indexes)
        other_embeddings = embeddings[np.setdiff1d(np.arange(embeddings.shape[0]), word_indexes)]
        target_embeddings = embeddings[word_indexes]

        ax.scatter(target_embeddings[:, 0], target_embeddings[:, 1], alpha=1.0, c='red')
        ax.scatter(target_embeddings[:, 0], target_embeddings[:, 1], alpha=1, facecolors='none', edgecolors='r', s=1000)
        for word, word_index in zip(word_annotations, word_indexes):
            word_x, word_y = embeddings[word_index, 0], embeddings[word_index, 1]
            ax.annotate(word, xy=(word_x, word_y))
    
    # Draw word vectors
    ax.scatter(other_embeddings[:, 0], other_embeddings[:, 1], alpha=0.1, c='blue')

    # Avoid outliers ruining the visualization if they are quite far away
    xmin_quantile = np.quantile(embeddings[:, 0], q=0.01)
    xmax_quantile = np.quantile(embeddings[:, 0], q=0.99)
    ymin_quantile = np.quantile(embeddings[:, 1], q=0.01)
    ymax_quantile = np.quantile(embeddings[:, 1], q=0.99)
    ax.set_xlim(xmin_quantile, xmax_quantile)
    ax.set_ylim(ymin_quantile, ymax_quantile)
    
    plt.show()


def reduce_svd(embeddings, seed=0):
    """
    Applies SVD dimensionality reduction
    """
    svd = TruncatedSVD(n_components=2, n_iter=10, random_state=seed)
    return svd.fit_transform(embeddings)


def reduce_tsne(embeddings, seed=0):
    """
    Applies t-SNE dimensionality reduction
    """
    tsne = TSNE(n_components=2, n_iter=1000, metric='cosine', n_jobs=2, random_state=seed)
    return tsne.fit_transform(embeddings)


def get_analogies(
    embedding_matrix,
    idx_to_word,
    word_to_idx,
    similarity_matrix,
    similarity_func,
    positive_listing,
    negative_listing,
    k
):
    """
    Finds the top `k` most similar words following this reasoning:
        - Words that have highest similarity to words in `positive_listing`
        - Words that have highest distance to words in `negative_listing`
    """
    # Positive words (similarity)
    positive_indexes = np.array([word_to_idx[word] for word in positive_listing])
    word_positive_vector = np.sum(embedding_matrix[positive_indexes, :], axis=0)

    # Negative words (distance)
    negative_indexes = np.array([word_to_idx[word] for word in negative_listing])
    word_negative_vector = np.sum(embedding_matrix[negative_indexes, :], axis=0)

    # Find candidate words
    target_vector = (word_positive_vector - word_negative_vector) / (len(positive_listing) + len(negative_listing))
    total_indexes = np.concatenate((positive_indexes, negative_indexes))
    valid_indexes = np.setdiff1d(np.arange(similarity_matrix.shape[0]), total_indexes)
    candidate_vectors = embedding_matrix[valid_indexes]

    candidate_similarities = similarity_func(candidate_vectors, target_vector, transpose_q=True)
    candidate_similarities = candidate_similarities.ravel()
    
    relative_indexes = np.argsort(candidate_similarities, axis=0)[::-1]
    relative_indexes = relative_indexes[:K]
    top_K_indexes = valid_indexes[relative_indexes]
    top_K_words = [idx_to_word[idx] for idx in top_K_indexes]

    return top_K_words, candidate_similarities[relative_indexes]