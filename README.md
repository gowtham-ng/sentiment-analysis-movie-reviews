# Sentiment Analysis of Movie Reviews on IMDb using Deep Learning and GloVe

This repository contains the code and report for my graduate-level data mining course project at Arizona State University. The project focuses on performing sentiment analysis on movie reviews from IMDb using deep learning techniques and GloVe embeddings.

## Project Overview

The goal of this project is to classify movie reviews as either positive or negative using various machine learning models, including Long Short-Term Memory (LSTM) Recurrent Neural Networks, Convolutional Neural Networks (CNNs), and BERT (Bidirectional Encoder Representations from Transformers). The project also explores the use of different word embeddings, such as GloVe and FastText, to improve the performance of the models.

## Dataset

The dataset used in this project is the Large Movie Review Dataset, compiled by Stanford University's AI department. It consists of 50,000 movie reviews from IMDb, with equal numbers of positive and negative reviews. The reviews are labeled with the movie's score on a scale of 1-10, and the sentiment is classified as either positive (rating >= 7) or negative (rating <= 4).

## Methods

The project employs the following methods:

1. **Data Preprocessing**: Cleaning the review text by removing HTML tags, punctuation, stopwords, and converting to lowercase.
2. **Word Embeddings**: Utilizing GloVe and FastText embeddings to represent the textual data in a numerical format.
3. **Neural Networks**: Building and training LSTM and CNN models using the preprocessed data and word embeddings.
4. **BERT**: Implementing the BERT architecture for sentiment analysis.
5. **Evaluation**: Assessing the performance of the models using various metrics, such as accuracy, precision, recall, and F1-score.

## Results

The project report includes detailed results and discussions on the performance of the different models and techniques used. The BERT model achieved the highest accuracy and F1-score, followed by the LSTM models with GloVe embeddings.

## Files

- `code/`: Contains the Python code for data preprocessing, model implementation, and evaluation.
- `report/`: Contains the final project report in PDF format (`Final Project Report Gowtham Gopinathan.docx.pdf`).

## Requirements

- Python 3.x
- TensorFlow
- Keras
- NLTK
- pandas
- scikit-learn
- Transformers (for BERT)

## Usage

1. Clone the repository:
   `git clone https://github.com/gowtham-ng/sentiment-analysis-movie-reviews.git`
2. Install the required packages:
`pip install -r requirements.txt`
3. Run the Python scripts in the `code/` directory to preprocess the data, train the models, and evaluate the results.

## Acknowledgments

This project was completed as part of the CSE 572: Data Mining course at Arizona State University. Special thanks to Dr. Hannah Kerner for her guidance and support throughout the course.

## License

This project is licensed under the [MIT License](LICENSE).
