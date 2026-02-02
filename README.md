<H3>ENTER YOUR NAME : Preethi S</H3>
<H3>ENTER YOUR REGISTER NO : 212223230157</H3>
<H3>EX. NO.1</H3>
<H1>Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:

```
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("Churn_Modelling.csv")
df

df.isnull().sum()

df.fillna(0)
df.isnull().sum()

df.duplicated()

df['EstimatedSalary'].describe()

scaler = StandardScaler()
inc_cols = ['CreditScore', 'Tenure', 'Balance', 'EstimatedSalary']
scaled_values = scaler.fit_transform(df[inc_cols])
df[inc_cols] = pd.DataFrame(scaled_values, columns = inc_cols, index = df.index)
df

x = df.iloc[:, :-1]
y = df.iloc[:, -1]

print("X Values")
x

print("Y Values")
y

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

print("X Training data")
x_train

print("X Testing data")
x_test

```
## OUTPUT:


# Read the dataset from drive


<img width="1260" height="481" alt="image" src="https://github.com/user-attachments/assets/464a1c8a-466f-4af8-b5bb-2aed82b7fd89" />



# Finding Missing Values


<img width="200" height="496" alt="image" src="https://github.com/user-attachments/assets/ac35d11c-9af9-4086-a1ca-315dffd8f21e" />



# Handling Missing values



<img width="1253" height="505" alt="image" src="https://github.com/user-attachments/assets/57ef9313-128d-466f-aae3-c8eef89c9121" />




# Check for Duplicates


<img width="211" height="531" alt="image" src="https://github.com/user-attachments/assets/9824a12e-19b3-4391-8216-b59234da8f7a" />


# Detect Outliers


<img width="260" height="378" alt="image" src="https://github.com/user-attachments/assets/142e1c75-ec46-4caf-bd91-7dccee0fe190" />




# Normalize the dataset




<img width="1260" height="502" alt="image" src="https://github.com/user-attachments/assets/46ddd822-2db0-4d1c-87d1-390b1927595e" />





# Split the dataset into input and output




<img width="208" height="516" alt="image" src="https://github.com/user-attachments/assets/8c35c2ca-c523-4fef-b178-dbd96bea6cbd" />


# Print the training data and testing data




<img width="897" height="514" alt="image" src="https://github.com/user-attachments/assets/21c9516d-8861-4116-a118-1214cfef5750" />


<img width="867" height="503" alt="image" src="https://github.com/user-attachments/assets/27cf431e-8fa4-4c94-9695-8c78d61db01a" />




## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


