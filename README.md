# NLP assignments
This repository contains various assignments for the 2020 Natural Language Processing class at UNIBO (M.Sc. in Artificial Intelligence). In particular, assignments are present in the form of the following Jupyter notebooks:
1. [sentiment_analysis.ipynb](sentiment_analysis.ipynb): Contains an approach to address the task of predicting scores from the famous IMDB movie review dataset. In particular, different models, like logistic regression and linear SVM, were tested on a multi-class classification task over a simple TF-IDF feature matrix.
2. [distributed_representations.ipynb](distributed_representations.ipynb): Leverages the IMDB movie review dataset to test various textual embeddings, going from sparse to dense representations (co-occurrence matrix, `PPMI` matrix, `Word2Vec` and `GloVe`). In particular, it tries to inspect how different embeddings reflect in different semantic concepts (e.g. analogies and biases) and how much each embedding can be trusted for semantic purposes.
3. [sequence_labeling.ipynb](sequence_labeling.ipynb): Contains an implementation of various neural models to deal with POS tags on the `dependency treebank` dataset, by leveraging LSTMs, GRUs and additional CRF layers. Moreover, lots of experiments are carried out around the choice of hyperparameters and the addition/deletion of specific tools, like optimizers, learning rate schedulers, dropout, etc.
4. [fact_checking.ipynb](fact_checking.ipynb): Addresses the fact checking problem on the `fever` dataset, by using neural models for sentence embeddings and different claim/evidence merging strategies.

Moreover, additional files in the repo have the following usefulness: 
- `models/`: A folder that contains pre-trained `PyTorch` neural models for both notebooks `3` and `4`.
- `requirements.txt`: A `pip` dependency file that lists all the libraries (and their associated versions) needed to run the notebooks and hopefully reproduce reported results.
- `utils.py`: Small Python module containing some functions and classes used by more than one notebook.
