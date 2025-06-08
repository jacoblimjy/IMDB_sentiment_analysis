# IMDB Movie Sentiment Analysis

## 📄 Project Documents
- [Final Report (Google Drive)](https://drive.google.com/file/d/1vbjaMyGt-8BUJHQhXrxPDrzuc7uVEPF6/view?usp=drive_link)
- [Presentation Slides (Google Drive)](https://drive.google.com/file/d/1tscwWYULW_X2NIYrUm-0_N8CCDpUG2z_/view?usp=drive_link)

## Project Overview
This project implements various machine learning and deep learning approaches to analyze sentiment in IMDB movie reviews. The goal is to classify movie reviews as either positive or negative using different models and compare their performance.

# Models Implemented
- Convolutional Neural Network (CNN)
- Long Short-Term Memory Network (LSTM)
- DistilBERT (Fine-tuned)
- Logistic Regression
- Support Vector Machine (SVM)
- Multinomial Naïve Bayes

## Project Structure
```
├── data/
│   ├── pos/                  # Positive review samples
│   ├── neg/                  # Negative review samples
│   ├── train_set.csv         # Training dataset
│   └── test_set.csv          # Test dataset
├── cnn_model.ipynb           # CNN implementation
├── lstm_model.ipynb          # LSTM implementation
├── fine_tuned_distilbert_model.ipynb # DistilBERT implementation
├── logistic_regression_model.ipynb # Logistic Regression implementation
├── svm_model.ipynb           # SVM implementation
├── multinomial_naïve_bayes_model.ipynb
├── data_preprocessing.ipynb  # Data preparation steps
└── requirements.txt          # Project dependencies
```

## Data Preprocessing
The dataset consists of IMDB movie reviews labeled with sentiment (positive/negative). The preprocessing steps include:
- Text cleaning and processing (removing HTML tags, stop words etc.)
- Train-test split

## Model Training
Each model is implemented in its own Jupyter notebook, allowing for independent experimentation and comparison. The models are trained on the preprocessed IMDB review dataset.

## Dependencies
Required Python packages are listed in `requirements.txt`. Install them using:
```bash
pip install -r requirements.txt
```

## Usage
1. Run `data_preprocessing.ipynb` first to prepare the dataset
2. Execute individual model notebooks to train and evaluate different approaches
3. Compare results across different models

## Performance Metrics
Models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-score

## Future Improvements
- Implement ensemble learning approaches
- Experiment with different preprocessing techniques
- Try other transformer-based models
- Optimize hyperparameters
