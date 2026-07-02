# IMDb Movie Review Sentiment Analysis Using Recurrent Neural Networks (RNN)

A deep learning project that classifies IMDb movie reviews as **positive** or **negative** using a Simple Recurrent Neural Network (RNN) built with TensorFlow/Keras.

## Overview

This project demonstrates the end-to-end workflow of applying an RNN to a sequential text classification task. It uses the built-in IMDb dataset from Keras, trains a Simple RNN model on padded review sequences, and evaluates performance using standard classification metrics and visualizations. The notebook also includes a custom function to predict sentiment on new, user-provided reviews.

## Dataset

- **Source:** `tensorflow.keras.datasets.imdb` (50,000 labeled movie reviews, pre-split into 25,000 training and 25,000 testing samples)
- **Vocabulary size:** 10,000 most frequent words
- **Sequence length:** Reviews are padded/truncated to 200 tokens (post-padding, post-truncating)

## Model Architecture

| Layer | Configuration |
|---|---|
| Embedding | `input_dim=10000`, `output_dim=128` |
| SimpleRNN | 64 units |
| Dropout | 0.5 |
| Dense (output) | 1 unit, sigmoid activation |

**Compilation:** Adam optimizer, binary cross-entropy loss, accuracy metric.

**Training:** 5 epochs, batch size 64, 20% validation split.

## Requirements

```
tensorflow
scikit-learn
seaborn
matplotlib
numpy
```

Install dependencies with:

```bash
pip install tensorflow scikit-learn seaborn matplotlib numpy
```

## Usage

1. Open `Recurrent_Neural_Networks.ipynb` in Jupyter Notebook / JupyterLab / Google Colab.
2. Run all cells in order to:
   - Load and preprocess the IMDb dataset
   - Build and train the RNN model
   - Evaluate performance on the test set
   - Visualize accuracy/loss curves and the confusion matrix
   - Predict sentiment on a custom review string

### Predicting a Custom Review

The notebook includes a helper function to classify any new review:

```python
review = "This movie was absolutely amazing and I loved every scene."
test = encode_review(review)
prediction = model.predict(test)
```

## Results & Evaluation

The notebook evaluates the model using:
- **Test loss and accuracy**
- **Accuracy and loss curves** (training vs. validation)
- **Classification report** (precision, recall, F1-score)
- **Confusion matrix** (visualized with a Seaborn heatmap)

**Key observations from the analysis:**
- Training accuracy improves steadily across epochs, while validation accuracy remains flat/erratic and lags behind — indicating **overfitting**. The model memorizes training data rather than generalizing well.
- The confusion matrix shows a skew toward predicting "Negative," with a notable number of actual positive reviews misclassified as negative.

## Potential Improvements

- Replace `SimpleRNN` with `LSTM` or `GRU` layers to better capture long-range dependencies and reduce vanishing gradient issues.
- Add early stopping or increase dropout/regularization to address overfitting.
- Use pre-trained word embeddings (e.g., GloVe, Word2Vec) instead of learning embeddings from scratch.
- Perform hyperparameter tuning (embedding size, RNN units, batch size, learning rate).

## Conclusion

This project demonstrates the effectiveness of Recurrent Neural Networks for sentiment analysis by learning sequential patterns in movie reviews. The trained model predicts whether an IMDb review expresses a positive or negative opinion, with performance assessed comprehensively via accuracy, precision, recall, F1-score, and confusion matrix analysis.

## License

This project is provided for educational purposes.
