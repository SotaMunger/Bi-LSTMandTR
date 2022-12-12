# Bidirectional_LSTM_and_TextRank
This code compares the performance of fake news classification models that use whole texts vs. TextRank sentence extractions.

To test code, simply download 'Fake.csv' and 'True.csv' files from https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?select=Fake.csv and place them in the same folder as the notebook.  The 'glove.6B.50d.txt' vector dictionary, available at https://nlp.stanford.edu/data/glove.6B.zip, must also be placed in the notebook's folder.  Compile by running all the cells in the Jupyter notebook.  Required python libraries:
- pandas
- numpy
- nltk
- re
- contractions
- scikit-learn
- tensorflow
- gensim v3.8
