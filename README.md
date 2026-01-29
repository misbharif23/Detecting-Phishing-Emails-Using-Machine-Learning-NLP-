# Detecting-Phishing-Emails-Using-Machine-Learning-NLP-
### Overview

Phishing emails are fraudulent messages designed to trick users into revealing sensitive information such as passwords, banking details, or personal data. This project applies **Machine Learning (ML)** and **Natural Language Processing (NLP)** techniques to automatically detect phishing emails by analyzing their textual content.

The implementation is based on academic research and evaluates **six different ML algorithms** on real-world datasets to compare accuracy, precision, recall, and computational efficiency.

### Features

* Comprehensive text preprocessing pipeline
* TF-IDF–based feature extraction
* Evaluation of six machine learning algorithms
* Performance comparison using accuracy, precision, recall, F1-score, and ROC curves
* Confusion matrix visualization for each classifier

### Technologies Used

* Python
* Scikit-learn
* Pandas
* NumPy
* Natural Language Processing (NLP)
* TF-IDF Vectorization
* Jupyter Notebook

### Project Structure

```
├── dataset/
│   ├── processed_data.csv
├── notebook/
│   └── phishing_email_detection.ipynb
├── paper/
│   └── Detecting_Phishing_Emails_Using_ML_and_NLP.pdf
├── README.md
└── requirements.txt

Methodology
### 1. Data Preprocessing
- Handling missing subjects and messages
- Removing duplicate emails
- Cleaning text (removing stopwords, URLs, email addresses, and non-alphanumeric characters)

### 2. Feature Engineering
- Conversion of email text into numerical form using **TF-IDF vectorization**

### 3. Machine Learning Models
The following algorithms were evaluated:
- Logistic Regression (LR)
- K-Nearest Neighbors (KNN)
- Multinomial Naive Bayes (MNB)
- AdaBoost (AB)
- Gradient Boosting (GB)
- Random Forest (RF)

### 4. Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- ROC Curves
- Training Time

---

## ▶️ How to Run
1. Clone the repository:
```bash
git clone https://github.com/your-username/phishing-email-detection.git](https://github.com/misbharif23/Detecting-Phishing-Emails-Using-Machine-Learning-NLP-.git
````

2. Navigate to the project directory:

```bash
cd phishing-email-detection
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the notebook or script to train and test the model.

---

## Results

* **Random Forest** achieved the highest accuracy and recall, especially on larger datasets
* **KNN** and **Multinomial Naive Bayes** trained the fastest but showed lower accuracy
* Including the **email subject** significantly improved detection performance

The Random Forest model proved to be the most reliable and production-ready classifier for phishing email detection.


## 🔮 Future Improvements

* Integration of deep learning models (LSTM, BERT)
* Advanced feature selection techniques
* Testing on additional real-world datasets
* Deployment as a real-time email filtering system


