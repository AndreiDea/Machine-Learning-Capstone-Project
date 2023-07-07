import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score

#   This is a comparison of two ML models whose purpose is to predict whether a person graduated from college or not, based on 
#   a selection of factors. The steps taken to complete this program are as follows:
#   
#   1. Read and prepare the CSV file for training and testing
#   2. Create the Logistic Regression and the K-Means classifier and train them on the dataset
#   3. Test the models on the dataset, plot statistics about the two models and compare them


# Step 1: Read and prepare the CSV file for training and testing:

df = pd.read_csv("~/Downloads/capstone_starter/profiles.csv")

#Create a function that gets the avg word length in a string
def get_str_avg_len(string):
    split_str = string.split(' ')
    lenghts = []
    for substr in split_str:
        lenghts.append(len(substr))
    return np.mean(lenghts)

#Create a column with the lenghts of all the combined essays
essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]
all_essays = df[essay_cols].replace(np.nan, '', regex=True)
all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)
df["essay_len"] = all_essays.apply(lambda x: len(x))

#Create a column with avg word lenghts
avg_word_lengths = all_essays.apply(lambda row: get_str_avg_len(str(row)))
df['avg_word_len'] = avg_word_lengths

#Create a column with number of lang spoken
df['num_lang_spoken'] = df['speaks'].apply(lambda row: len(str(row).split(', ')))

#Create code for smokes, drinks, drugs
features_of_interest = ['drinks', 'smokes', 'drugs', 'status']
df.dropna(subset=features_of_interest, inplace=True)

smokes_code = {'no': 0, 'trying to quit': 1, 'when drinking': 2, 'sometimes': 3, 'yes': 4}
drinks_code = {'not at all': 0, 'rarely': 1, 'socially': 2, 'often': 3, 'very often': 4, 'desperately': 5}
drugs_code = {'never': 0, 'sometimes': 1, 'often': 2}

df['smokes_code'] = df['smokes'].map(smokes_code)
df['drinks_code'] = df['drinks'].map(drinks_code)
df['drugs_code'] = df['drugs'].map(drugs_code)

#Create code for religion
df.dropna(subset=['religion'], inplace=True)
religions_in_dataset = ['christianity', 'catholicism', 'other', 'buddhism', 'judaism', 'hinduism', 'islam']
religion_codes = []

for row in df['religion']:
    for religion in religions_in_dataset:
        if str(row).split(' ')[0] == religion:
            religion_codes.append(2)
            break
    if str(row).split(' ')[0] == 'agnosticism':
        religion_codes.append(1)
    if str(row).split(' ')[0] == 'atheism':
        religion_codes.append(0)

df['religion_code'] = religion_codes

#Create labels
ambiguous_cols = ['working on space camp', 'graduated from space camp', 'dropped out of space camp', 'space camp']
df.drop(df[df['education'].apply(lambda row: row in ambiguous_cols)].index, inplace=True)
df.dropna(subset=['education'], inplace=True)
to_replace = ['college/university', 'two-year college', 'high school', 'masters program', 'law school', 'ph.d program',
              'med school']
with_ = ['graduated from college/university', 'graduated from two-year college', 'graduated from high school', 
         'graduated from masters program', 'graduated from law school', 'graduated from ph.d program',
         'graduated from med school']
for i in range(len(to_replace)):
    df['education'].replace(to_replace=to_replace[i], value=with_[i], inplace=True)

true_labels = ['graduated from college/university', 'graduated from masters program', 'graduated from ph.d program',
               'working on masters program', 'graduated from law school', 'working on ph.d program', 
               'dropped out of ph.d program', 'graduated from med school', 'working on med school',
               'working on law school', 'dropped out of masters program', 'dropped out of med school',
               'dropped out of law school']
false_labels = ['working on college/university', 'working on two-year college', 'graduated from high school',
                'dropped out of college/university', 'graduated from two-year college', 'dropped out of high school',
                'working on high school', 'dropped out of two-year college']

label_map = {}
for value in true_labels:
    label_map[value] = 1
for value in false_labels:
    label_map[value] = 0

df['labels_code'] = df['education'].map(label_map)


# Step 2: Create the Logistic Regression and the K-Means classifier and train them on the dataset

#Select features and prepare them for fitting
cols_to_use = ['essay_len', 'avg_word_len', 'num_lang_spoken', 'smokes_code', 'drinks_code', 'drugs_code', 'religion_code']


labels = df[['labels_code']]

#Create and fit the models
reg_accuracy = []
for i in range(len(cols_to_use)):
    new_features = []
    for j in range(i + 1):
        new_features.append(cols_to_use[j])
    
    X_train, X_test, y_train, y_test = train_test_split(df[new_features], labels, test_size=0.2, random_state=1)
    print(new_features)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    reg_accuracy.append(f1_score(y_test, y_pred))


# Step 3: Test the models on the dataset, plot statistics about the two models and compare them

#Caluclate accuracy (in this code, we plot the accuracy. To plot the F1 score, we simply need to change 
#accuracy_score to f1_score on line 137)
k_accuracy = []
for i in range(1, 30):
    model = KNeighborsClassifier(n_neighbors=i)
    X_train, X_test, y_train, y_test = train_test_split(df[cols_to_use], labels, test_size=0.2, random_state=1)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    k_accuracy.append(accuracy_score(y_test, y_pred))

#Plot statistics and display tables
fig, axs = plt.subplots(2)
fig.tight_layout()
fig.suptitle('Accuracy of the two models')

axs[0].plot(range(1, len(cols_to_use) + 1), reg_accuracy)
axs[0].set_xlabel('Logistic Regression: Number of features used')
axs[0].set_ylabel('Accuracy')

axs[1].plot(range(1, 30), k_accuracy)
axs[1].set_xlabel('K-Nearest Neighbors: Number of neighbors used')
axs[1].set_ylabel('Accuracy')

plt.show()

axs[0].plot(range(1, len(cols_to_use) + 1), reg_accuracy)
axs[0].set_xlabel('Logistic Regression: Number of features used')
axs[0].set_ylabel('F-1 Score')

axs[1].plot(range(1, 30), k_accuracy)
axs[1].set_xlabel('K-Nearest Neighbors: Number of neighbors used')
axs[1].set_ylabel('F-1 Score')

plt.show()
