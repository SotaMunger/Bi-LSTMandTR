# Bidirectional_LSTM_and_TextRank Fake News Classifier

This end-to-end natural language processing data science project analyzes the effectiveness of using TextRank  extracted sentences to improve the performance of a fake news classifier. The method involves training four bi-directional long-short term memory (BD-LSTM) neural net classification models - one on three sentence TextRank extracted summaries of news articles, one on four sentence summaries, one on the article titles, and one on the entire article texts - then testing those models on similarly treated test cases for accuracy, precision, and recall. The data set consists of 23481 articles taken from various internet sources deemed untrustworthy by Politifact and 21417 articles pulled from Reuters.com, an established news source with a well-regarded reputation.

Initial exploratory data analyses revealed that both 'fake' and 'real' articles had similar mean sentence counts, on average, but that fake news articles had a wider distribution of sentence counts:

![image](https://user-images.githubusercontent.com/91567553/234726303-8fe09ab1-c430-4f9c-98c0-b0f9c001fb0d.png)

Using NLP libraries such as NLTK and Python Re, both fake and real articles were processed through several cleaning steps to remove such features as twitter pic URLs, twitter handles, hash-tags, and time stamps. Because the goal of the study was to classify articles based solely on semantic patterns in the text, these features were removed, even though they could have improved prediction scores. The summaries TextRank summaries were then created using the GenSim v3.8 (https://radimrehurek.com/gensim_3.8.3/) gensim.summarizer.summarize() function. Word tokens from the articles, summaries, and titles were then converted to 50 dimensional GloVe vector embeddings (https://nlp.stanford.edu/data/glove.6B.zip) and fed into their respective BD-LSTM models. Each model, consisting of the architecture below, was then trained for 20 epochs with a batch size of 512:

![image](https://user-images.githubusercontent.com/91567553/234727655-97f9012b-e23f-4213-a49c-53f530f6615a.png)

Testing of the models produced the following confusion matrices and performance values:

![image](https://user-images.githubusercontent.com/91567553/234727911-eaaba0a9-b78f-4229-93d1-6c5253846441.png)

![image](https://user-images.githubusercontent.com/91567553/234727941-6b7a4e37-7321-48e6-90d2-c7b9a30a9a98.png)

These results revealed that using TextRank sentence summaries did not significantly increase the performance of the fake news classifier.

To test the code yourself, simply download 'Fake.csv' and 'True.csv' files from https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?select=Fake.csv and place them in the same folder as the notebook.  The 'glove.6B.50d.txt' vector dictionary, available at https://nlp.stanford.edu/data/glove.6B.zip, must also be placed in the notebook's folder.  Compile by running all the cells in the Jupyter notebook.  Required python libraries:

- pandas
- numpy
- nltk
- re
- contractions
- scikit-learn
- tensorflow
- gensim v3.8
