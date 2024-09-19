# Twitter-Sentiment-Analysis

 we try to implement an NLP Twitter sentiment analysis model that helps to overcome the challenges of sentiment classification of tweets. We will be classifying the tweets into positive or negative sentiments. The necessary details regarding the dataset involving the Twitter sentiment analysis project are:

The dataset provided is the Sentiment140 Dataset which consists of 1,600,000 tweets that have been extracted using the Twitter API. The various columns present in this Twitter data are:
* target: the polarity of the tweet (positive or negative)
* ids: Unique id of the tweet
* date: the date of the tweet
* flag: It refers to the query. If no such query exists, then it is NO QUERY.
* user: It refers to the name of the user that tweeted
* text: It refers to the text of the tweet

#  Twitter Sentiment Analysis Dataset: Project Pipeline
The various steps involved in the `Machine Learning Pipeline` are:

* Import Necessary Dependencies
* Read and Load the Dataset
* Exploratory Data Analysis
* Data Visualization of Target Variables
* Data Preprocessing
* Splitting our data into Train and Test sets.
* Transforming Dataset using TF-IDF Vectorizer
* Function for Model Evaluation
* Model Building
* Model Evaluation

#### Installation
To run this project, you need to have the following libraries installed:

bash
Copy code
*     pip install pandas numpy matplotlib seaborn scikit-learn
#### Data Exploration
We began the analysis with Exploratory Data Analysis (EDA) to understand the dataset. Key steps included:

* Data Cleaning: Removing duplicates, null values, and irrelevant information.
* Text Preprocessing: Tokenization, stemming, and lemmatization of tweets.
* Visualization: Word frequency graphs to identify the most common terms in positive and negative tweets.

### Model Comparison
Three different models were compared for their performance on the sentiment analysis task:

1. Logistic Regression
2. Support Vector Machine (SVM)
3. Bernoulli Naive Bayes

Each model was evaluated using:

* F1 Score: To measure the model's accuracy in classifying tweets.
* Accuracy: The proportion of true results (both true positives and true negatives) among the total number of cases examined.
* Confusion Matrix: To visualize the performance of the model.
* ROC-AUC Curve: To assess the models' ability to distinguish between classes.

###  ROC-AUC Curve
The ROC-AUC curves for all three models were plotted to visualize their performance.

Results
After comparing the models, Logistic Regression was found to be the best fit for the dataset, achieving the highest F1 score, accuracy, and AUC.

* Logistic Regression F1 Score: X.XX
* Logistic Regression Accuracy: X.XX
* Logistic Regression Confusion Matrix:

lua
Copy code
   *       [[True Negatives, False Positives],
           [False Negatives, True Positives]]

       
SVM F1 Score: X.XX

SVM Accuracy: X.XX

SVM Confusion Matrix:

lua
Copy code
*      [[True Negatives, False Positives],
       [False Negatives, True Positives]]

   
* Bernoulli Naive Bayes F1 Score: X.XX
* Bernoulli Naive Bayes Accuracy: X.XX
* Bernoulli Naive Bayes Confusion Matrix:

    lua
Copy code
*      [[True Negatives, False Positives],
      [False Negatives, True Positives]]


