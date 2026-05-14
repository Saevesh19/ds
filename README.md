# ds
# DSBDAL Complete Practical Codes (Python)

## 1. Data Wrangling I

```python
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('data.csv')

# Display first rows
print(df.head())

# Dataset information
print(df.info())

# Dimensions
print("Shape:", df.shape)

# Missing values
print(df.isnull().sum())

# Statistical summary
print(df.describe())

# Data types
print(df.dtypes)

# Fill missing numeric values with mean
numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].mean())

# Convert categorical variables into quantitative variables
categorical_cols = df.select_dtypes(include='object').columns

for col in categorical_cols:
    df = pd.get_dummies(df, columns=[col])

print(df.head())
```

---

# 2. Data Wrangling II

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('academic.csv')

print(df.head())

# Missing values
print(df.isnull().sum())

# Fill missing numeric values
numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].mean())

# Detect outliers using boxplot
for col in numeric_cols:
    plt.figure(figsize=(5,3))
    sns.boxplot(y=df[col])
    plt.title(col)
    plt.show()

# Remove outliers using IQR
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df = df[(df[col] >= lower) & (df[col] <= upper)]

# Normalization
for col in numeric_cols:
    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

print(df.head())
```

---

# 3. Descriptive Statistics I

```python
import pandas as pd

# Load dataset
df = pd.read_csv('data.csv')

# Grouped statistics
summary = df.groupby('Gender')['Income'].agg(
    ['mean', 'median', 'min', 'max', 'std']
)

print(summary)

# Create list for each category
income_list = df.groupby('Gender')['Income'].apply(list)

print(income_list)
```

---

# 4. Descriptive Statistics II (Iris Dataset)

```python
import pandas as pd

# Load dataset
df = pd.read_csv('iris.csv')

print(df.head())

# Group by species
grouped = df.groupby('species')

# Statistical details
print(grouped.describe())
```

---

# 5. Linear Regression - Boston Housing

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('BostonHousing.csv')

print(df.head())

# Heatmap
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Pairplot
sns.pairplot(df[['RM','LSTAT','PTRATIO','MEDV']])
plt.show()

# Features and target
X = df.drop('MEDV', axis=1)
y = df['MEDV']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('MSE:', mse)
print('R2 Score:', r2)
```

---

# 6. Logistic Regression - Social Network Ads

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Load dataset
df = pd.read_csv('Social_Network_Ads.csv')

# Features and target
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Logistic regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TP = cm[1][1]

accuracy = (TP + TN) / (TP + TN + FP + FN)
error_rate = (FP + FN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)

print('TP:', TP)
print('FP:', FP)
print('TN:', TN)
print('FN:', FN)
print('Accuracy:', accuracy)
print('Error Rate:', error_rate)
print('Precision:', precision)
print('Recall:', recall)
```

---

# 7. Naive Bayes - Iris Dataset

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

# Load dataset
df = pd.read_csv('iris.csv')

# Features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# Model
model = GaussianNB()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

TP = cm[0][0]
FP = cm[1][0] + cm[2][0]
FN = cm[0][1] + cm[0][2]
TN = cm.sum() - (TP + FP + FN)

accuracy = (TP + TN) / (TP + TN + FP + FN)
error_rate = (FP + FN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)

print('TP:', TP)
print('FP:', FP)
print('TN:', TN)
print('FN:', FN)
print('Accuracy:', accuracy)
print('Error Rate:', error_rate)
print('Precision:', precision)
print('Recall:', recall)
```

---

# 8. Text Analytics

```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer

# Downloads
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

text = "Natural Language Processing helps computers understand language"

# Tokenization
tokens = word_tokenize(text)
print(tokens)

# POS Tagging
print(pos_tag(tokens))

# Stop words removal
stop_words = set(stopwords.words('english'))
filtered = [word for word in tokens if word.lower() not in stop_words]
print(filtered)

# Stemming
ps = PorterStemmer()
stemmed = [ps.stem(word) for word in filtered]
print(stemmed)

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(word) for word in filtered]
print(lemmatized)

# TF-IDF
documents = [
    'Natural language processing is artificial intelligence',
    'Artificial intelligence helps computers understand language'
]

vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(documents)

print(tfidf.toarray())
print(vectorizer.get_feature_names_out())
```

---

# 9. Data Visualization I - Titanic Dataset

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('titanic.csv')

# Survival count
sns.countplot(x='survived', data=df)
plt.show()

# Survival based on gender
sns.countplot(x='sex', hue='survived', data=df)
plt.show()

# Passenger class vs survival
sns.countplot(x='pclass', hue='survived', data=df)
plt.show()

# Fare distribution
plt.figure(figsize=(10,6))
sns.histplot(df['fare'], bins=30, kde=True)
plt.title('Fare Distribution')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.show()
```

---

# 10. Data Visualization II - Titanic Boxplot

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('titanic.csv')

# Boxplot
plt.figure(figsize=(10,6))

sns.boxplot(
    x='sex',
    y='age',
    hue='survived',
    data=df
)

plt.title('Age Distribution by Gender and Survival')
plt.show()
```

---

# 11. Data Visualization III - Iris Dataset

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('iris.csv')

# Data types
print(df.dtypes)

# Histograms
df.hist(figsize=(12,10))
plt.show()

# Boxplots
plt.figure(figsize=(12,8))

for i, column in enumerate(df.columns[:-1], 1):

    plt.subplot(2,2,i)

    sns.boxplot(y=df[column])

    plt.title(column)

plt.tight_layout()
plt.show()
```

---

# 12. MapReduce Log File Processing (Python)

## mapper.py

```python
import sys

for line in sys.stdin:

    line = line.strip()

    words = line.split()

    if len(words) > 0:

        log_type = words[0]

        print(f'{log_type}\t1')
```

## reducer.py

```python
import sys

current_type = None
current_count = 0

for line in sys.stdin:

    line = line.strip()

    log_type, count = line.split('\t')

    count = int(count)

    if current_type == log_type:
        current_count += count

    else:
        if current_type:
            print(f'{current_type}\t{current_count}')

        current_type = log_type
        current_count = count

if current_type:
    print(f'{current_type}\t{current_count}')
```

## Run in terminal

```bash
cat log.txt | python mapper.py | sort | python reducer.py
```

---

# 13. Scala Spark Program

```scala
object HelloSpark {
  def main(args: Array[String]): Unit = {
    println("Apache Spark Scala Program")
  }
}
```
