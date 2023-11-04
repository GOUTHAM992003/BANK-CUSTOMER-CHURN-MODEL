# BANK-CUSTOMER-CHURN-MODEL

Bank Customer Churn Analysis
Churner is generally defined as a customer who stops using a product or service for a given period of time.

This notebook is to do the data analysis and predictions on the churn.csv file.

The first step in the Data Preprocessing is to import the libraries, load the data and do some Exploratory Data Analysis (EDA).

Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pysankey import sankey

# For the predictive models
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier as GBSklearn
from xgboost import XGBClassifier as XGB
import lightgbm as lgb

# Removing annoying warnings
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
C:\Users\soane\Anaconda3\lib\site-packages\pysankey\sankey.py:24: UserWarning: matplotlib.pyplot as already been imported, this call will have no effect.
  matplotlib.use('Agg')
Defining useful functions:

def count_by_group(data, feature, target):
    df = data.groupby([feature, target])[target].agg(['count'])
    temp = data.groupby([feature])[target].agg(['count'])
    df['pct'] = 100*df.div(temp, level = feature).reset_index()['count'].values
    return df.reset_index()

def classification_report_to_dataframe(true, predictions, predictions_proba, model_name, balanced = 'no'):
    a = classification_report(true, predictions, output_dict = True)
    zeros = pd.DataFrame(data = a['0'], index = [0]).iloc[:,0:3].add_suffix('_0')
    ones = pd.DataFrame(data = a['1'], index = [0]).iloc[:,0:3].add_suffix('_1')
    df = pd.concat([zeros, ones], axis = 1)
    temp = list(df)
    df['Model'] = model_name
    df['Balanced'] = balanced
    df['Accuracy'] = accuracy_score(true, predictions)
    df['Balanced_Accuracy'] = balanced_accuracy_score(true, predictions)
    df['AUC'] = roc_auc_score(true, predictions_proba, average = 'macro')
    df = df[['Model', 'Balanced', 'Accuracy', 'Balanced_Accuracy', 'AUC'] + temp]
    return df

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
Importing the dataset
dataset = pd.read_csv('churn.csv')
1. Exploratory Data Analysis
Printing the first rows of the dataset:

dataset.head()
RowNumber	CustomerId	Surname	CreditScore	Geography	Gender	Age	Tenure	Balance	NumOfProducts	HasCrCard	IsActiveMember	EstimatedSalary	Exited
0	1	15634602	Hargrave	619	France	Female	42	2	0.00	1	1	1	101348.88	1
1	2	15647311	Hill	608	Spain	Female	41	1	83807.86	1	0	1	112542.58	0
2	3	15619304	Onio	502	France	Female	42	8	159660.80	3	1	0	113931.57	1
3	4	15701354	Boni	699	France	Female	39	1	0.00	2	0	0	93826.63	0
4	5	15737888	Mitchell	850	Spain	Female	43	2	125510.82	1	1	1	79084.10	0
dataset.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10000 entries, 0 to 9999
Data columns (total 14 columns):
RowNumber          10000 non-null int64
CustomerId         10000 non-null int64
Surname            10000 non-null object
CreditScore        10000 non-null int64
Geography          10000 non-null object
Gender             10000 non-null object
Age                10000 non-null int64
Tenure             10000 non-null int64
Balance            10000 non-null float64
NumOfProducts      10000 non-null int64
HasCrCard          10000 non-null int64
IsActiveMember     10000 non-null int64
EstimatedSalary    10000 non-null float64
Exited             10000 non-null int64
dtypes: float64(2), int64(9), object(3)
memory usage: 1.1+ MB
Checking if there is any missing data in the dataset:

dataset.isna().sum()
RowNumber          0
CustomerId         0
Surname            0
CreditScore        0
Geography          0
Gender             0
Age                0
Tenure             0
Balance            0
NumOfProducts      0
HasCrCard          0
IsActiveMember     0
EstimatedSalary    0
Exited             0
dtype: int64
The dataset has no missing values.

round(dataset.describe(),3)
RowNumber	CustomerId	CreditScore	Age	Tenure	Balance	NumOfProducts	HasCrCard	IsActiveMember	EstimatedSalary	Exited
count	10000.000	1.000000e+04	10000.000	10000.000	10000.000	10000.000	10000.000	10000.000	10000.000	10000.000	10000.000
mean	5000.500	1.569094e+07	650.529	38.922	5.013	76485.889	1.530	0.706	0.515	100090.240	0.204
std	2886.896	7.193619e+04	96.653	10.488	2.892	62397.405	0.582	0.456	0.500	57510.493	0.403
min	1.000	1.556570e+07	350.000	18.000	0.000	0.000	1.000	0.000	0.000	11.580	0.000
25%	2500.750	1.562853e+07	584.000	32.000	3.000	0.000	1.000	0.000	0.000	51002.110	0.000
50%	5000.500	1.569074e+07	652.000	37.000	5.000	97198.540	1.000	1.000	1.000	100193.915	0.000
75%	7500.250	1.575323e+07	718.000	44.000	7.000	127644.240	2.000	1.000	1.000	149388.248	0.000
max	10000.000	1.581569e+07	850.000	92.000	10.000	250898.090	4.000	1.000	1.000	199992.480	1.000
Computing the number of exited and not exited clients:

exited = len(dataset[dataset['Exited'] == 1]['Exited'])
not_exited = len(dataset[dataset['Exited'] == 0]['Exited'])
exited_perc = round(exited/len(dataset)*100,1)
not_exited_perc = round(not_exited/len(dataset)*100,1)

print('Number of clients that have exited the program: {} ({}%)'.format(exited, exited_perc))
print('Number of clients that haven\'t exited the program: {} ({}%)'.format(not_exited, not_exited_perc))
Number of clients that have exited the program: 2037 (20.4%)
Number of clients that haven't exited the program: 7963 (79.6%)
So, around of 20% of the clients exited the bank, while around 80% stayed. As the goal here is to identify which of the customers are at higher risk to discontinue their services with the bank, we are dealing with a classification problem.

A important point to take into consideration here is that we are dealing with an imbalanced dataset.

country = list(dataset['Geography'].unique())
gender = list(dataset['Gender'].unique())

print(country)
print(gender)
['France', 'Spain', 'Germany']
['Female', 'Male']
# Create a Exited string variable to create the plots
dataset['Exited_str'] = dataset['Exited']
dataset['Exited_str'] = dataset['Exited_str'].map({1: 'Exited', 0: 'Stayed'})
gender_count = dataset['Gender'].value_counts()
gender_pct= gender_count / len(dataset.index)

gender = pd.concat([gender_count, round(gender_pct,2)], axis=1)\
        .set_axis(['count', 'pct'], axis=1, inplace=False)
gender
count	pct
Male	5457	0.55
Female	4543	0.45
geo_count = dataset['Geography'].value_counts()
geo_pct= geo_count / len(dataset.index)

geo = pd.concat([geo_count, round(geo_pct,2)], axis=1)\
        .set_axis(['count', 'pct'], axis=1, inplace=False)
geo
count	pct
France	5014	0.50
Germany	2509	0.25
Spain	2477	0.25
In the dataset, there are more men (55%) than women (45%), and it has only 3 different countries: France, Spain, and Germany. Where 50% of the customers are from France and 25% are from Germany, and the other group are from Spain.

Now, let's just check the relationship between the features and the outcome ('Exited').

def count_by_group(data, feature, target):
    df = data.groupby([feature, target])[target].agg(['count'])
    temp = data.groupby([feature])[target].agg(['count'])
    df['pct'] = 100*df.div(temp, level = feature).reset_index()['count'].values
    return df.reset_index()
count_by_group(dataset, feature = 'Gender', target = 'Exited')
Gender	Exited	count	pct
0	Female	0	3404	74.928461
1	Female	1	1139	25.071539
2	Male	0	4559	83.544072
3	Male	1	898	16.455928
pd.options.display.max_rows = 8
colorDict = {
    'Exited':'#f71b1b',
    'Stayed':'grey',
    'Female':'#FFD700',
    'Male':'#8E388E'
}
sankey(
    dataset['Gender'], dataset['Exited_str'], aspect=20, colorDict=colorDict,
    fontsize=12, figureName="Gender"
)

count_by_group(dataset, feature = 'Geography', target = 'Exited')
Geography	Exited	count	pct
0	France	0	4204	83.845233
1	France	1	810	16.154767
2	Germany	0	1695	67.556796
3	Germany	1	814	32.443204
4	Spain	0	2064	83.326605
5	Spain	1	413	16.673395
colorDict = {
    'Exited':'#f71b1b',
    'Stayed':'grey',
    'France':'#f3f71b',
    'Spain':'#12e23f',
    'Germany':'#f78c1b'
}
sankey(
    dataset['Geography'], dataset['Exited_str'], aspect=20, colorDict=colorDict,
    fontsize=12, figureName="geography"
)

HasCrCard_count = dataset['HasCrCard'].value_counts()
HasCrCard_pct= HasCrCard_count / len(dataset.index)

HasCrCard = pd.concat([HasCrCard_count, HasCrCard_pct], axis=1)\
        .set_axis(['count', 'pct'], axis=1, inplace=False)

HasCrCard
count	pct
1	7055	0.7055
0	2945	0.2945
count_by_group(dataset, feature = 'HasCrCard', target = 'Exited')
HasCrCard	Exited	count	pct
0	0	0	2332	79.185059
1	0	1	613	20.814941
2	1	0	5631	79.815734
3	1	1	1424	20.184266
# Create a HasCrCard string variable to create the plots
dataset['HasCrCard_str'] = dataset['HasCrCard'].map({1: 'Has Credit Card', 0: 'Does not have Credit Card'})

colorDict = {
    'Exited':'#f71b1b',
    'Stayed':'grey',
    'Has Credit Card':'#FFD700',
    'Does not have Credit Card':'#8E388E'
}
sankey(
    dataset['HasCrCard_str'], dataset['Exited_str'], aspect=20, colorDict=colorDict,
    fontsize=12, figureName="HasCrCard"
)

IsActiveMember_count = dataset['IsActiveMember'].value_counts()
IsActiveMember_pct= HasCrCard_count / len(dataset.index)

IsActiveMember = pd.concat([IsActiveMember_count, IsActiveMember_pct], axis=1)\
        .set_axis(['count', 'pct'], axis=1, inplace=False)

IsActiveMember
count	pct
1	5151	0.7055
0	4849	0.2945
count_by_group(dataset, feature = 'IsActiveMember', target = 'Exited')
IsActiveMember	Exited	count	pct
0	0	0	3547	73.149103
1	0	1	1302	26.850897
2	1	0	4416	85.730926
3	1	1	735	14.269074
# Create a IsActiveMember string variable to create the plots
dataset['IsActiveMember_str'] = dataset['IsActiveMember'].map({1: 'Is Active Member', 0: 'Is Not ActiveMember'})

colorDict = {
    'Exited':'#f71b1b',
    'Stayed':'grey',
    'Is Active Member':'#FFD700',
    'Is Not ActiveMember':'#8E388E'
}
sankey(
    dataset['IsActiveMember_str'], dataset['Exited_str'], aspect=20, colorDict=colorDict,
    fontsize=12, figureName="IsActiveMember_str"
)

NumOfProducts_count = dataset['NumOfProducts'].value_counts()
NumOfProducts_pct= NumOfProducts_count / len(dataset.index)

NumOfProducts = pd.concat([NumOfProducts_count, round(NumOfProducts_pct,2)], axis=1)\
        .set_axis(['count', 'pct'], axis=1, inplace=False)
NumOfProducts
count	pct
1	5084	0.51
2	4590	0.46
3	266	0.03
4	60	0.01
count_by_group(dataset, feature = 'NumOfProducts', target = 'Exited')
NumOfProducts	Exited	count	pct
0	1	0	3675	72.285602
1	1	1	1409	27.714398
2	2	0	4242	92.418301
3	2	1	348	7.581699
4	3	0	46	17.293233
5	3	1	220	82.706767
6	4	1	60	100.000000
# Create a IsActiveMember string variable to create the plots
dataset['NumOfProducts_str'] = dataset['NumOfProducts'].map({1: '1', 2: '2', 3: '3', 4: '4'})

colorDict = {
    'Exited':'#f71b1b',
    'Stayed':'grey',
    '1':'#f3f71b',
    '2':'#12e23f',
    '3':'#f78c1b',
    '4':'#8E388E'
}
sankey(
    dataset['NumOfProducts_str'], dataset['Exited_str'], aspect=20, colorDict=colorDict,
    fontsize=12, figureName="NumOfProducts"
)

#Stacked histogram: Age
figure = plt.figure(figsize=(15,8))
plt.hist([
        dataset[(dataset.Exited==0)]['Age'],
        dataset[(dataset.Exited==1)]['Age']
        ], 
         stacked=True, color = ['grey','r'],
         bins = 'auto',label = ['Stayed','Exited'],
         edgecolor='black', linewidth=1.2)
plt.xlabel('Age (years)')
plt.ylabel('Number of customers')
plt.legend()
<matplotlib.legend.Legend at 0x23d9c768908>

fig, axes = plt.subplots(nrows=2, ncols=2, figsize = (15,15))
fig.subplots_adjust(left=0.2, wspace=0.6)
ax0, ax1, ax2, ax3 = axes.flatten()

ax0.hist([
        dataset[(dataset.Exited==0)]['CreditScore'],
        dataset[(dataset.Exited==1)]['CreditScore']
        ], 
         stacked=True, color = ['grey','r'],
         bins = 'auto',label = ['Stayed','Exited'],
         edgecolor='black', linewidth=1.2)
ax0.legend()
ax0.set_title('Credit Score')

ax1.hist([
        dataset[(dataset.Exited==0)]['Tenure'],
        dataset[(dataset.Exited==1)]['Tenure']
        ], 
         stacked=True, color = ['grey','r'],
         bins = 'auto',label = ['Stayed','Exited'],
         edgecolor='black', linewidth=1.2)
ax1.legend()
ax1.set_title('Tenure')

ax2.hist([
        dataset[(dataset.Exited==0)]['Balance'],
        dataset[(dataset.Exited==1)]['Balance']
        ], 
         stacked=True, color = ['grey','r'],
         bins = 'auto',label = ['Stayed','Exited'],
         edgecolor='black', linewidth=1.2)
ax2.legend()
ax2.set_title('Balance')

ax3.hist([
        dataset[(dataset.Exited==0)]['EstimatedSalary'],
        dataset[(dataset.Exited==1)]['EstimatedSalary']
        ], 
         stacked=True, color = ['grey','r'],
         bins = 'auto',label = ['Stayed','Exited'],
         edgecolor='black', linewidth=1.2)
ax3.legend()
ax3.set_title('Estimated Salary')

fig.tight_layout()
plt.show()

From the tables and plots above, we can have some insights:

As for gender, women are lower in number than the men, but have a higher rate to close the account.
There is a higher rate of exited clients in Germany (32%, which is about 2x higher), and lower in Spain and France (around 16% each).
On age, customer bellow 40 and above 65 years old have a tendency to keep their account.
Has or not credit card does not impact on the decision to stay in the bank (both groups has 20% of exited customers)
Non active members tend to discontinue their services with a bank compared with the active clients (27% vs 14%).
The dataset has 96% of clients with 1 or 2 product, and customers with 1 product only have a higher rate to to close the account than those with 2 products (around 3x higher).
Estimated Salary does not seem to affect the churn rate
2. Predictive Models
Separating Dataset into X and y subsets
In this project we will test the following models and choose the best one based on the accuracy, balanced accuracy, and Exited Recall.

Models to be tested:

Logistic Regresstion (Package: Sklearn)
Multi Layers Perceptron - MLP (Package: Sklearn)
XGBoost (Package: XGBoost)
XGB: Gradient Boosting Classifier (Package: Sklearn)
Light GBM (Package: LightGBM)
As we have imbalanced dataset, we will test all the models defined above using two different strategies:

Complete training set (80% of the dataset)
Balanced training set, where we randomly select from the complete tranning set the same number of Stayed and Exited customers.
2.1 One-Hot encoding Categorical Attributes
# One-Hot encoding our categorical attributes
list_cat = ['Geography', 'Gender']
dataset = pd.get_dummies(dataset, columns = list_cat, prefix = list_cat)
dataset.head()
RowNumber	CustomerId	Surname	CreditScore	Age	Tenure	Balance	NumOfProducts	HasCrCard	IsActiveMember	...	Exited	Exited_str	HasCrCard_str	IsActiveMember_str	NumOfProducts_str	Geography_France	Geography_Germany	Geography_Spain	Gender_Female	Gender_Male
0	1	15634602	Hargrave	619	42	2	0.00	1	1	1	...	1	Exited	Has Credit Card	Is Active Member	1	1	0	0	1	0
1	2	15647311	Hill	608	41	1	83807.86	1	0	1	...	0	Stayed	Does not have Credit Card	Is Active Member	1	0	0	1	1	0
2	3	15619304	Onio	502	42	8	159660.80	3	1	0	...	1	Exited	Has Credit Card	Is Not ActiveMember	3	1	0	0	1	0
3	4	15701354	Boni	699	39	1	0.00	2	0	0	...	0	Stayed	Does not have Credit Card	Is Not ActiveMember	2	1	0	0	1	0
4	5	15737888	Mitchell	850	43	2	125510.82	1	1	1	...	0	Stayed	Has Credit Card	Is Active Member	1	0	0	1	1	0
5 rows × 21 columns

dataset = dataset.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited_str','HasCrCard_str', 'IsActiveMember_str','NumOfProducts_str'], axis = 1)
dataset.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10000 entries, 0 to 9999
Data columns (total 14 columns):
CreditScore          10000 non-null int64
Age                  10000 non-null int64
Tenure               10000 non-null int64
Balance              10000 non-null float64
NumOfProducts        10000 non-null int64
HasCrCard            10000 non-null int64
IsActiveMember       10000 non-null int64
EstimatedSalary      10000 non-null float64
Exited               10000 non-null int64
Geography_France     10000 non-null uint8
Geography_Germany    10000 non-null uint8
Geography_Spain      10000 non-null uint8
Gender_Female        10000 non-null uint8
Gender_Male          10000 non-null uint8
dtypes: float64(2), int64(7), uint8(5)
memory usage: 752.0 KB
features = list(dataset.drop('Exited', axis = 1))
target = 'Exited'
2.2 Splitting the dataset into the Training set and Test set
Now, let's split the data intro train and test sets (80% and 20%, respectively).

train, test = train_test_split(dataset, test_size = 0.2, random_state = 1)

print('Number of clients in the dataset: {}'.format(len(dataset)))
print('Number of clients in the train set: {}'.format(len(train)))
print('Number of clients in the test set: {}'.format(len(test)))
Number of clients in the dataset: 10000
Number of clients in the train set: 8000
Number of clients in the test set: 2000
exited_train = len(train[train['Exited'] == 1]['Exited'])
exited_train_perc = round(exited_train/len(train)*100,1)

exited_test = len(test[test['Exited'] == 1]['Exited'])
exited_test_perc = round(exited_test/len(test)*100,1)

print('Complete Train set - Number of clients that have exited the program: {} ({}%)'.format(exited_train, exited_train_perc))
print('Test set - Number of clients that haven\'t exited the program: {} ({}%)'.format(exited_test, exited_test_perc))
Complete Train set - Number of clients that have exited the program: 1622 (20.3%)
Test set - Number of clients that haven't exited the program: 415 (20.8%)
2.3 Feature Scaling
The data contain features in different order of magnitude. Tree based models can handle this kind of data without any normalization, but logistic regression and neural networks (MLP) require the normalization of the data for a better performance.

Here, I'm doing the StandardScaler normalization, and it is done according to the equation below:

 
where 
 is the a data element, 
 is the mean of the feature, 
 is the standard deviation, and 
 is the normalized element.

sc = StandardScaler()

# fit on training set
train[features] = sc.fit_transform(train[features])

# only transform on test set
test[features] = sc.transform(test[features])
2.4 Complete Trainning Set
2.4.1 Logistic Regression (Sklearn)
For the first prediction, let's use the Sklearn Logistic Regression searching for the best parameters using the GridSearchCV function:

parameters = {'C': [0.01, 0.1, 1, 10],
              'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
              'max_iter': [50, 100, 150]}
LR = LogisticRegression(penalty = 'l2')
model_LR = GridSearchCV(LR, parameters, cv = 5, n_jobs = 10, verbose = 1).fit(train[features], train[target])
pd.DataFrame(model_LR.cv_results_)
Fitting 5 folds for each of 60 candidates, totalling 300 fits
[Parallel(n_jobs=10)]: Using backend LokyBackend with 10 concurrent workers.
[Parallel(n_jobs=10)]: Done  30 tasks      | elapsed:    3.3s
[Parallel(n_jobs=10)]: Done 281 out of 300 | elapsed:    5.9s remaining:    0.3s
[Parallel(n_jobs=10)]: Done 300 out of 300 | elapsed:    6.1s finished
mean_fit_time	std_fit_time	mean_score_time	std_score_time	param_C	param_max_iter	param_solver	params	split0_test_score	split1_test_score	...	mean_test_score	std_test_score	rank_test_score	split0_train_score	split1_train_score	split2_train_score	split3_train_score	split4_train_score	mean_train_score	std_train_score
0	0.058843	0.012850	0.002393	0.000488	0.01	50	newton-cg	{'C': 0.01, 'max_iter': 50, 'solver': 'newton-...	0.813866	0.818239	...	0.811250	0.004373	26	0.811846	0.807939	0.815156	0.814560	0.813154	0.812531	0.002567
1	0.017952	0.001545	0.001397	0.000489	0.01	50	lbfgs	{'C': 0.01, 'max_iter': 50, 'solver': 'lbfgs'}	0.813866	0.818239	...	0.811250	0.004373	26	0.811846	0.807939	0.815156	0.814560	0.813154	0.812531	0.002567
2	0.033111	0.005965	0.001795	0.000399	0.01	50	liblinear	{'C': 0.01, 'max_iter': 50, 'solver': 'libline...	0.813866	0.819488	...	0.811125	0.005073	39	0.810595	0.807782	0.813438	0.814092	0.813310	0.811843	0.002358
3	0.108509	0.021717	0.001197	0.000399	0.01	50	sag	{'C': 0.01, 'max_iter': 50, 'solver': 'sag'}	0.813866	0.818239	...	0.811250	0.004373	26	0.811846	0.807939	0.815156	0.814560	0.813154	0.812531	0.002567
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
56	0.030518	0.011276	0.005983	0.009972	10	150	lbfgs	{'C': 10, 'max_iter': 150, 'solver': 'lbfgs'}	0.815740	0.819488	...	0.811000	0.006173	46	0.809970	0.809814	0.813750	0.814560	0.813779	0.812375	0.002048
57	0.052062	0.017955	0.001993	0.000629	10	150	liblinear	{'C': 10, 'max_iter': 150, 'solver': 'liblinear'}	0.815740	0.819488	...	0.811000	0.006173	46	0.809970	0.809814	0.813750	0.814560	0.813779	0.812375	0.002048
58	0.194080	0.063775	0.001397	0.000489	10	150	sag	{'C': 10, 'max_iter': 150, 'solver': 'sag'}	0.815740	0.819488	...	0.811000	0.006173	46	0.809970	0.809814	0.813750	0.814560	0.813779	0.812375	0.002048
59	0.099533	0.035409	0.001196	0.000400	10	150	saga	{'C': 10, 'max_iter': 150, 'solver': 'saga'}	0.815740	0.819488	...	0.811000	0.006173	46	0.809970	0.809814	0.813750	0.814560	0.813779	0.812375	0.002048
60 rows × 23 columns

print(model_LR.best_params_)
{'C': 0.1, 'max_iter': 50, 'solver': 'newton-cg'}
Now that we know the "best" parameters for the model, let's do a Recursive Feature Elimination to check the feature importance.

model = LogisticRegression(**model_LR.best_params_)
model.fit(train[features], train[target])

importances = abs(model.coef_[0])
importances = 100.0 * (importances / importances.max())
indices = np.argsort(importances)

plt.figure(figsize = (15, 8))
plt.title('Feature Importances: Complete Logistic Regression')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

Now, let's compute the predictions for the best set of parameters:

pred = model_LR.predict(test[features])
predp = model_LR.predict_proba(test[features])[:,1]

cm = confusion_matrix(test[target], pred)
plot_confusion_matrix(cm, target_names = ['Not Exited', 'Exited'], normalize = False)
plot_confusion_matrix(cm, target_names = ['Not Exited', 'Exited'], normalize = True, title = 'Confusion Matrix (Normalized)')

table_of_models = classification_report_to_dataframe(test[target], pred, predp, model_name = 'Logistic Regression')
table_of_models


Model	Balanced	Accuracy	Balanced_Accuracy	AUC	precision_0	recall_0	f1-score_0	precision_1	recall_1	f1-score_1
0	Logistic Regression	no	0.8115	0.59203	0.777608	0.825081	0.967192	0.890502	0.633803	0.216867	0.32316
2.4.2 MLP: Multi Layers Perceptron
Now, let's enter in the Neural Networks domain, by doing predictions using the Sklear Multi-Layer Perceptron (MLP) Classifier.

s = len(features)
parameters = {'hidden_layer_sizes': [(s,),
                                     (s,)*2,
                                     (s,)*4,
                                     (s,)*6],
              'solver': ['lbfgs', 'adam'],
              'alpha': [0, 0.01, 0.1, 1, 10]}
MLP = MLPClassifier()
model_MLP = GridSearchCV(MLP, parameters, cv = 5, n_jobs = 10, verbose = 1).fit(train[features], train[target])
pd.DataFrame(model_MLP.cv_results_)
Fitting 5 folds for each of 40 candidates, totalling 200 fits
[Parallel(n_jobs=10)]: Using backend LokyBackend with 10 concurrent workers.
[Parallel(n_jobs=10)]: Done  30 tasks      | elapsed:   16.5s
[Parallel(n_jobs=10)]: Done 180 tasks      | elapsed:  1.8min
[Parallel(n_jobs=10)]: Done 200 out of 200 | elapsed:  2.0min finished
mean_fit_time	std_fit_time	mean_score_time	std_score_time	param_alpha	param_hidden_layer_sizes	param_solver	params	split0_test_score	split1_test_score	...	mean_test_score	std_test_score	rank_test_score	split0_train_score	split1_train_score	split2_train_score	split3_train_score	split4_train_score	mean_train_score	std_train_score
0	1.331840	0.029704	0.012769	0.014581	0	(13,)	lbfgs	{'alpha': 0, 'hidden_layer_sizes': (13,), 'sol...	0.855715	0.858838	...	0.854375	0.003751	10	0.866854	0.867323	0.867344	0.870645	0.869864	0.868406	0.001539
1	5.326761	1.349720	0.001995	0.000001	0	(13,)	adam	{'alpha': 0, 'hidden_layer_sizes': (13,), 'sol...	0.858838	0.865084	...	0.853250	0.008333	14	0.860134	0.862166	0.865625	0.858616	0.861115	0.861531	0.002357
2	1.989682	0.523871	0.002392	0.000486	0	(13, 13)	lbfgs	{'alpha': 0, 'hidden_layer_sizes': (13, 13), '...	0.850718	0.851968	...	0.849375	0.004579	28	0.877325	0.873574	0.879844	0.876582	0.879394	0.877344	0.002247
3	6.960394	0.534453	0.002397	0.000492	0	(13, 13)	adam	{'alpha': 0, 'hidden_layer_sizes': (13, 13), '...	0.863210	0.858214	...	0.855625	0.006496	5	0.868729	0.868729	0.870938	0.861897	0.865021	0.867063	0.003207
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
36	5.595949	0.131108	0.005186	0.000977	10	(13, 13, 13, 13)	lbfgs	{'alpha': 10, 'hidden_layer_sizes': (13, 13, 1...	0.863210	0.855091	...	0.855625	0.008262	5	0.871074	0.874043	0.872188	0.874238	0.873770	0.873062	0.001230
37	9.071874	1.975966	0.004987	0.000631	10	(13, 13, 13, 13)	adam	{'alpha': 10, 'hidden_layer_sizes': (13, 13, 1...	0.797002	0.797002	...	0.797250	0.000208	39	0.797312	0.797312	0.797188	0.797219	0.797219	0.797250	0.000052
38	6.549893	0.140594	0.009175	0.003051	10	(13, 13, 13, 13, 13, 13)	lbfgs	{'alpha': 10, 'hidden_layer_sizes': (13, 13, 1...	0.863210	0.851968	...	0.855375	0.007234	7	0.866385	0.874355	0.870625	0.872520	0.873613	0.871500	0.002849
39	8.306718	1.301964	0.004188	0.001832	10	(13, 13, 13, 13, 13, 13)	adam	{'alpha': 10, 'hidden_layer_sizes': (13, 13, 1...	0.797002	0.797002	...	0.797250	0.000208	39	0.797312	0.797312	0.797188	0.797219	0.797219	0.797250	0.000052
40 rows × 23 columns

print(model_MLP.best_params_)
{'alpha': 10, 'hidden_layer_sizes': (13,), 'solver': 'lbfgs'}
pred = model_MLP.predict(test[features])
predp = model_MLP.predict_proba(test[features])[:,1]

cm = confusion_matrix(test[target], pred)
plot_confusion_matrix(cm, target_names = ['Not Exited', 'Exited'], normalize = False)
plot_confusion_matrix(cm, target_names = ['Not Exited', 'Exited'], normalize = True, title = 'Confusion Matrix (Normalized)')

temp = classification_report_to_dataframe(test[target], pred, predp, model_name = 'Multi-Layer Perceptron (MLP)')
temp


Model	Balanced	Accuracy	Balanced_Accuracy	AUC	precision_0	recall_0	f1-score_0	precision_1	recall_1	f1-score_1
0	Multi-Layer Perceptron (MLP)	no	0.864	0.725651	0.871359	0.87795	0.962145	0.918122	0.771863	0.489157	0.59882
table_of_models = table_of_models.append(temp)
table_of_models
Model	Balanced	Accuracy	Balanced_Accuracy	AUC	precision_0	recall_0	f1-score_0	precision_1	recall_1	f1-score_1
0	Logistic Regression	no	0.8115	0.592030	0.777608	0.825081	0.967192	0.890502	0.633803	0.216867	0.32316
0	Multi-Layer Perceptron (MLP)	no	0.8640	0.725651	0.871359	0.877950	0.962145	0.918122	0.771863	0.489157	0.59882
2.4.3 Gradient Boosting Classifier (Sklearn)
Now, let's use the Sklearn Gradient Boosting Classifier algorithm for the predictions.

parameters = {'max_depth': [2, 3, 4, 6, 10, 15],
              'n_estimators': [50, 100, 300, 500]}
GB = GBSklearn()
model_GB = GridSearchCV(GB, parameters, cv = 5, n_jobs = 10, verbose = 1).fit(train[features], train[target])
pd.DataFrame(model_GB.cv_results_)
Fitting 5 folds for each of 24 candidates, totalling 120 fits
[Parallel(n_jobs=10)]: Using backend LokyBackend with 10 concurrent workers.
[Parallel(n_jobs=10)]: Done  30 tasks      | elapsed:    7.0s
[Parallel(n_jobs=10)]: Done 120 out of 120 | elapsed:  2.8min finished
mean_fit_time	std_fit_time	mean_score_time	std_score_time	param_max_depth	param_n_estimators	params	split0_test_score	split1_test_score	split2_test_score	...	mean_test_score	std_test_score	rank_test_score	split0_train_score	split1_train_score	split2_train_score	split3_train_score	split4_train_score	mean_train_score	std_train_score
0	0.433641	0.034464	0.012967	0.013502	2	50	{'max_depth': 2, 'n_estimators': 50}	0.871330	0.852592	0.848750	...	0.856000	0.008035	13	0.859509	0.859822	0.861719	0.860178	0.860647	0.860375	0.000771
1	1.088889	0.105544	0.017753	0.011552	2	100	{'max_depth': 2, 'n_estimators': 100}	0.870706	0.859463	0.853750	...	0.861250	0.005585	7	0.864041	0.867167	0.868594	0.864709	0.866271	0.866156	0.001646
2	3.564870	0.042774	0.019348	0.007125	2	300	{'max_depth': 2, 'n_estimators': 300}	0.871330	0.864460	0.856250	...	0.862500	0.005158	2	0.874668	0.877012	0.876875	0.877988	0.874863	0.876281	0.001297
3	5.467584	0.082546	0.035904	0.015361	2	500	{'max_depth': 2, 'n_estimators': 500}	0.871330	0.861337	0.856875	...	0.860750	0.005720	9	0.882482	0.885920	0.884844	0.884081	0.882987	0.884063	0.001242
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
20	17.372581	0.527232	0.022540	0.002999	15	50	{'max_depth': 15, 'n_estimators': 50}	0.812617	0.840100	0.813750	...	0.825125	0.011432	24	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	0.000000
21	34.747278	0.774498	0.052260	0.006603	15	100	{'max_depth': 15, 'n_estimators': 100}	0.821986	0.848844	0.832500	...	0.837875	0.009547	21	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	0.000000
22	34.790757	1.127172	0.044481	0.002570	15	300	{'max_depth': 15, 'n_estimators': 300}	0.820112	0.848844	0.826250	...	0.836000	0.010912	23	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	0.000000
23	25.329547	0.701231	0.036097	0.008044	15	500	{'max_depth': 15, 'n_estimators': 500}	0.830731	0.850718	0.826875	...	0.837500	0.008789	22	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	0.000000
24 rows × 22 columns

print(model_GB.best_params_)
{'max_depth': 3, 'n_estimators': 100}
model = GBSklearn(**model_GB.best_params_)
model.fit(train[features], train[target])

importances = model.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize = (15, 8))
plt.title('Feature Importances: Complete Gradient Boosting (Sklearn)')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

pred = model_GB.predict(test[features])
predp = model_GB.predict_proba(test[features])[:,1]

cm = confusion_matrix(test[target], pred)
plot_confusion_matrix(cm, target_names = ['Not Exited', 'Exited'], normalize = False)
plot_confusion_matrix(cm, target_names = ['Not Exited', 'Exited'], normalize = True, title = 'Confusion Matrix (Normalized)')

temp = classification_report_to_dataframe(test[target], pred, predp, model_name = 'Gradient Boosting (Sklearn)')
temp


Model	Balanced	Accuracy	Balanced_Accuracy	AUC	precision_0	recall_0	f1-score_0	precision_1	recall_1	f1-score_1
0	Gradient Boosting (Sklearn)	no	0.86	0.715123	0.878344	0.873497	0.962776	0.915966	0.766798	0.46747	0.580838
table_of_models = table_of_models.append(temp)
table_of_models
Model	Balanced	Accuracy	Balanced_Accuracy	AUC	precision_0	recall_0	f1-score_0	precision_1	recall_1	f1-score_1
0	Logistic Regression	no	0.8115	0.592030	0.777608	0.825081	0.967192	0.890502	0.633803	0.216867	0.323160
0	Multi-Layer Perceptron (MLP)	no	0.8640	0.725651	0.871359	0.877950	0.962145	0.918122	0.771863	0.489157	0.598820
0	Gradient Boosting (Sklearn)	no	0.8600	0.715123	0.878344	0.873497	0.962776	0.915966	0.766798	0.467470	0.580838
2.4.4 Extreme Gradient Boosting (XGBoost)
Let's get out of the Scikit-learn for now and try the package XGBoost.

model_XGB = XGB(max_depth = 6,
            learning_rate = .1,
            n_estimators = 100,
            reg_lambda = 0.5,
            reg_alpha = 0,
            verbosity = 1,
            n_jobs = -1,
            tree_method = 'gpu_exact').fit(train[features], train[target])

pred = model_XGB.predict(test[features])
predp = model_XGB.predict_proba(test[features])[:,1]

importances = model_XGB.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize = (15, 8))
plt.title('Feature Importances: Complete Extreme Gradient Boosting (XGBoost)')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

cm = confusion_matrix(test[target], pred)
plot_confusion_matrix(cm, target_names = ['Not Exited', 'Exited'], normalize = False)
plot_confusion_matrix(cm, target_names = ['Not Exited', 'Exited'], normalize = True, title = 'Confusion Matrix (Normalized)')

temp = classification_report_to_dataframe(test[target], pred, predp, model_name = 'Gradient Boosting (XGBoost)')
temp


Model	Balanced	Accuracy	Balanced_Accuracy	AUC	precision_0	recall_0	f1-score_0	precision_1	recall_1	f1-score_1
0	Gradient Boosting (XGBoost)	no	0.8635	0.722667	0.875068	0.876579	0.963407	0.917944	0.775194	0.481928	0.594354
table_of_models = table_of_models.append(temp)
table_of_models
Model	Balanced	Accuracy	Balanced_Accuracy	AUC	precision_0	recall_0	f1-score_0	precision_1	recall_1	f1-score_1
0	Logistic Regression	no	0.8115	0.592030	0.777608	0.825081	0.967192	0.890502	0.633803	0.216867	0.323160
0	Multi-Layer Perceptron (MLP)	no	0.8640	0.725651	0.871359	0.877950	0.962145	0.918122	0.771863	0.489157	0.598820
0	Gradient Boosting (Sklearn)	no	0.8600	0.715123	0.878344	0.873497	0.962776	0.915966	0.766798	0.467470	0.580838
0	Gradient Boosting (XGBoost)	no	0.8635	0.722667	0.875068	0.876579	0.963407	0.917944	0.775194	0.481928	0.594354
2.4.5 Light Gradient Boosting Machine (LightGBM)
Now, another gradient boosting algorithm, from the package LightGBM.

train_data = lgb.Dataset(train[features],
                         label = train[target],
                         feature_name = features)
test_data = lgb.Dataset(test[features + [target]],
                              reference = train_data)
param = {'num_leaves': 21,
         'num_trees': 100,
         # 'max_depth': 20,
         'objective': 'binary',
         # 'boosting': 'dart',
         'lambda_l1': 1,
         'lambda_l2': 1,
         'learning_rate': 0.1,
         'metric': ['binary_logloss', 'binary_error'],
         'seed': 1}

num_round = 10
model_LGB = lgb.train(param, train_data, num_round, valid_sets=[test_data])
[1]	valid_0's binary_logloss: 0.227971	valid_0's binary_error: 0
[2]	valid_0's binary_logloss: 0.229883	valid_0's binary_error: 0
[3]	valid_0's binary_logloss: 0.232169	valid_0's binary_error: 0
[4]	valid_0's binary_logloss: 0.234652	valid_0's binary_error: 0
[5]	valid_0's binary_logloss: 0.237528	valid_0's binary_error: 0
[6]	valid_0's binary_logloss: 0.240373	valid_0's binary_error: 0.049
[7]	valid_0's binary_logloss: 0.243268	valid_0's binary_error: 0.0595
[8]	valid_0's binary_logloss: 0.246461	valid_0's binary_error: 0.072
[9]	valid_0's binary_logloss: 0.24951	valid_0's binary_error: 0.073
[10]	valid_0's binary_logloss: 0.252733	valid_0's binary_error: 0.0765
[11]	valid_0's binary_logloss: 0.255786	valid_0's binary_error: 0.082
[12]	valid_0's binary_logloss: 0.258817	valid_0's binary_error: 0.0885
[13]	valid_0's binary_logloss: 0.26197	valid_0's binary_error: 0.097
[14]	valid_0's binary_logloss: 0.265179	valid_0's binary_error: 0.1
[15]	valid_0's binary_logloss: 0.268195	valid_0's binary_error: 0.1055
[16]	valid_0's binary_logloss: 0.270178	valid_0's binary_error: 0.1105
[17]	valid_0's binary_logloss: 0.27254	valid_0's binary_error: 0.111
[18]	valid_0's binary_logloss: 0.274992	valid_0's binary_error: 0.1115
[19]	valid_0's binary_logloss: 0.277749	valid_0's binary_error: 0.1145
[20]	valid_0's binary_logloss: 0.280398	valid_0's binary_error: 0.1145
[21]	valid_0's binary_logloss: 0.28213	valid_0's binary_error: 0.1145
[22]	valid_0's binary_logloss: 0.284872	valid_0's binary_error: 0.1175
[23]	valid_0's binary_logloss: 0.287522	valid_0's binary_error: 0.117
[24]	valid_0's binary_logloss: 0.289096	valid_0's binary_error: 0.1195
[25]	valid_0's binary_logloss: 0.291027	valid_0's binary_error: 0.12
[26]	valid_0's binary_logloss: 0.29337	valid_0's binary_error: 0.1205
[27]	valid_0's binary_logloss: 0.294913	valid_0's binary_error: 0.12
[28]	valid_0's binary_logloss: 0.297251	valid_0's binary_error: 0.122
[29]	valid_0's binary_logloss: 0.298583	valid_0's binary_error: 0.1225
[30]	valid_0's binary_logloss: 0.300167	valid_0's binary_error: 0.1235
[31]	valid_0's binary_logloss: 0.302507	valid_0's binary_error: 0.123
[32]	valid_0's binary_logloss: 0.303701	valid_0's binary_error: 0.1235
[33]	valid_0's binary_logloss: 0.306011	valid_0's binary_error: 0.124
[34]	valid_0's binary_logloss: 0.3071	valid_0's binary_error: 0.1255
[35]	valid_0's binary_logloss: 0.308441	valid_0's binary_error: 0.1255
[36]	valid_0's binary_logloss: 0.310195	valid_0's binary_error: 0.126
[37]	valid_0's binary_logloss: 0.311274	valid_0's binary_error: 0.125
[38]	valid_0's binary_logloss: 0.313144	valid_0's binary_error: 0.126
[39]	valid_0's binary_logloss: 0.313318	valid_0's binary_error: 0.1265
[40]	valid_0's binary_logloss: 0.314013	valid_0's binary_error: 0.1265
[41]	valid_0's binary_logloss: 0.315214	valid_0's binary_error: 0.1265
[42]	valid_0's binary_logloss: 0.317	valid_0's binary_error: 0.127
[43]	valid_0's binary_logloss: 0.317327	valid_0's binary_error: 0.127
[44]	valid_0's binary_logloss: 0.318352	valid_0's binary_error: 0.127
[45]	valid_0's binary_logloss: 0.319424	valid_0's binary_error: 0.1285
[46]	valid_0's binary_logloss: 0.321291	valid_0's binary_error: 0.1295
[47]	valid_0's binary_logloss: 0.321829	valid_0's binary_error: 0.129
[48]	valid_0's binary_logloss: 0.322367	valid_0's binary_error: 0.128
[49]	valid_0's binary_logloss: 0.323694	valid_0's binary_error: 0.1285
[50]	valid_0's binary_logloss: 0.324826	valid_0's binary_error: 0.129
[51]	valid_0's binary_logloss: 0.326448	valid_0's binary_error: 0.1305
[52]	valid_0's binary_logloss: 0.326922	valid_0's binary_error: 0.1305
[53]	valid_0's binary_logloss: 0.327818	valid_0's binary_error: 0.1305
[54]	valid_0's binary_logloss: 0.328217	valid_0's binary_error: 0.131
[55]	valid_0's binary_logloss: 0.329908	valid_0's binary_error: 0.132
[56]	valid_0's binary_logloss: 0.330559	valid_0's binary_error: 0.1325
[57]	valid_0's binary_logloss: 0.331328	valid_0's binary_error: 0.1325
[58]	valid_0's binary_logloss: 0.331981	valid_0's binary_error: 0.133
[59]	valid_0's binary_logloss: 0.332286	valid_0's binary_error: 0.133
[60]	valid_0's binary_logloss: 0.333631	valid_0's binary_error: 0.134
[61]	valid_0's binary_logloss: 0.334273	valid_0's binary_error: 0.1335
[62]	valid_0's binary_logloss: 0.335357	valid_0's binary_error: 0.134
[63]	valid_0's binary_logloss: 0.335435	valid_0's binary_error: 0.134
[64]	valid_0's binary_logloss: 0.335571	valid_0's binary_error: 0.134
[65]	valid_0's binary_logloss: 0.336418	valid_0's binary_error: 0.134
[66]	valid_0's binary_logloss: 0.336855	valid_0's binary_error: 0.134
[67]	valid_0's binary_logloss: 0.337351	valid_0's binary_error: 0.134
[68]	valid_0's binary_logloss: 0.338173	valid_0's binary_error: 0.134
[69]	valid_0's binary_logloss: 0.338181	valid_0's binary_error: 0.1335
[70]	valid_0's binary_logloss: 0.33886	valid_0's binary_error: 0.134
[71]	valid_0's binary_logloss: 0.339127	valid_0's binary_error: 0.134
[72]	valid_0's binary_logloss: 0.339131	valid_0's binary_error: 0.134
[73]	valid_0's binary_logloss: 0.33967	valid_0's binary_error: 0.1335
[74]	valid_0's binary_logloss: 0.34006	valid_0's binary_error: 0.1335
[75]	valid_0's binary_logloss: 0.340925	valid_0's binary_error: 0.1335
[76]	valid_0's binary_logloss: 0.341895	valid_0's binary_error: 0.1335
[77]	valid_0's binary_logloss: 0.34214	valid_0's binary_error: 0.1345
[78]	valid_0's binary_logloss: 0.34254	valid_0's binary_error: 0.1335
[79]	valid_0's binary_logloss: 0.342197	valid_0's binary_error: 0.133
[80]	valid_0's binary_logloss: 0.341902	valid_0's binary_error: 0.1345
[81]	valid_0's binary_logloss: 0.342447	valid_0's binary_error: 0.1345
[82]	valid_0's binary_logloss: 0.34313	valid_0's binary_error: 0.134
[83]	valid_0's binary_logloss: 0.344059	valid_0's binary_error: 0.1345
[84]	valid_0's binary_logloss: 0.344631	valid_0's binary_error: 0.1345
[85]	valid_0's binary_logloss: 0.344571	valid_0's binary_error: 0.135
[86]	valid_0's binary_logloss: 0.34481	valid_0's binary_error: 0.135
[87]	valid_0's binary_logloss: 0.345851	valid_0's binary_error: 0.135
[88]	valid_0's binary_logloss: 0.346072	valid_0's binary_error: 0.1355
[89]	valid_0's binary_logloss: 0.34684	valid_0's binary_error: 0.1365
[90]	valid_0's binary_logloss: 0.347014	valid_0's binary_error: 0.136
[91]	valid_0's binary_logloss: 0.347793	valid_0's binary_error: 0.1365
[92]	valid_0's binary_logloss: 0.347596	valid_0's binary_error: 0.136
[93]	valid_0's binary_logloss: 0.34761	valid_0's binary_error: 0.1355
[94]	valid_0's binary_logloss: 0.347719	valid_0's binary_error: 0.1355
[95]	valid_0's binary_logloss: 0.347728	valid_0's binary_error: 0.135
[96]	valid_0's binary_logloss: 0.347801	valid_0's binary_error: 0.135
[97]	valid_0's binary_logloss: 0.348768	valid_0's binary_error: 0.135
[98]	valid_0's binary_logloss: 0.349353	valid_0's binary_error: 0.1345
[99]	valid_0's binary_logloss: 0.349113	valid_0's binary_error: 0.135
[100]	valid_0's binary_logloss: 0.348949	valid_0's binary_error: 0.1365
predp = model_LGB.predict(test[features])
pred = predp > 0.5

lgb.plot_importance(model_LGB,
                    figsize = (15,8),
                    height = 0.8,
                    title = 'Feature Importances: Complete Light Gradient Boosting Machine (LightGBM)',
                    ylabel = None,
                    grid = False)
<matplotlib.axes._subplots.AxesSubplot at 0x23d9fc1c8d0>

cm = confusion_matrix(test[target], pred)
plot_confusion_matrix(cm, target_names = ['Not Exited', 'Exited'], normalize = False)
plot_confusion_matrix(cm, target_names = ['Not Exited', 'Exited'], normalize = True, title = 'Confusion Matrix (Normalized)')

temp = classification_report_to_dataframe(test[target], pred, predp, model_name = 'Gradient Boosting (LightGBM)')
temp


Model	Balanced	Accuracy	Balanced_Accuracy	AUC	precision_0	recall_0	f1-score_0	precision_1	recall_1	f1-score_1
0	Gradient Boosting (LightGBM)	no	0.869	0.737699	0.881359	0.883034	0.962145	0.920894	0.78022	0.513253	0.619186
table_of_models = table_of_models.append(temp)
table_of_models
Model	Balanced	Accuracy	Balanced_Accuracy	AUC	precision_0	recall_0	f1-score_0	precision_1	recall_1	f1-score_1
0	Logistic Regression	no	0.8115	0.592030	0.777608	0.825081	0.967192	0.890502	0.633803	0.216867	0.323160
0	Multi-Layer Perceptron (MLP)	no	0.8640	0.725651	0.871359	0.877950	0.962145	0.918122	0.771863	0.489157	0.598820
0	Gradient Boosting (Sklearn)	no	0.8600	0.715123	0.878344	0.873497	0.962776	0.915966	0.766798	0.467470	0.580838
0	Gradient Boosting (XGBoost)	no	0.8635	0.722667	0.875068	0.876579	0.963407	0.917944	0.775194	0.481928	0.594354
0	Gradient Boosting (LightGBM)	no	0.8690	0.737699	0.881359	0.883034	0.962145	0.920894	0.780220	0.513253	0.619186
2.5 Balanced Tranning Set
The data provided is imbalanced, where 80% of the clients are not exited ones (labeled as 0), and 20% are exited ones (labeled as 1). We can reach 80% accuracy by only selecting a model that predict all the clients as 0. However, checking at the balanced accuracy, which is the average of the recall of each class.

One way to get around with this issue is to randomly remove data from the larger class until it matches the number of the smaller class, so 50% is 0 and 50% is 1. We do this balance only on the train set, so the trained model is not biased, but the test data is kept untouched.

The resample_data function will balance the data for us.

def resample_data(data, target):
    data_1 = data[data[target] == 1]
    data_0 = data[data[target] == 0]
    percentage = len(data_1)/len(data_0)
    temp = data_0.sample(frac = percentage, random_state = 1)

    data_new = data_1.append(temp)
    data_new.sort_index(inplace = True)
    return data_new
trainB = resample_data(train, target = target)
print('Number of clients in the dataset is : {}'.format(len(dataset)))
print('Number of clients in the balanced train set is : {}'.format(len(trainB)))
print('Number of clients in the test set is : {}'.format(len(test)))
Number of clients in the dataset is : 10000
Number of clients in the balanced train set is : 3244
Number of clients in the test set is : 2000
exited_trainB = len(trainB[trainB['Exited'] == 1]['Exited'])
exited_trainB_perc = round(exited_trainB/len(trainB)*100,1)

exited_test = len(test[test['Exited'] == 1]['Exited'])
exited_test_perc = round(exited_test/len(test)*100,1)

print('Balanced Train set - Number of clients that have exited the program: {} ({}%)'.format(exited_trainB, exited_trainB_perc))
print('Test set - Number of clients that haven\'t exited the program: {} ({}%)'.format(exited_test, exited_test_perc))
Balanced Train set - Number of clients that have exited the program: 1622 (50.0%)
Test set - Number of clients that haven't exited the program: 415 (20.8%)
2.5.1 Logistic Regresstion
parameters = {'C': [0.01, 0.1, 1, 10],
              'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
              'max_iter': [50, 100, 150]}
LR = LogisticRegression(penalty = 'l2')
model_LR = GridSearchCV(LR, parameters, cv = 5, n_jobs = 10, verbose = 1).fit(trainB[features], trainB[target])
pd.DataFrame(model_LR.cv_results_)
Fitting 5 folds for each of 60 candidates, totalling 300 fits
[Parallel(n_jobs=10)]: Using backend LokyBackend with 10 concurrent workers.
[Parallel(n_jobs=10)]: Done 100 tasks      | elapsed:    0.8s
[Parallel(n_jobs=10)]: Done 300 out of 300 | elapsed:    2.7s finished
mean_fit_time	std_fit_time	mean_score_time	std_score_time	param_C	param_max_iter	param_solver	params	split0_test_score	split1_test_score	...	mean_test_score	std_test_score	rank_test_score	split0_train_score	split1_train_score	split2_train_score	split3_train_score	split4_train_score	mean_train_score	std_train_score
0	0.020745	0.000747	0.001595	0.000489	0.01	50	newton-cg	{'C': 0.01, 'max_iter': 50, 'solver': 'newton-...	0.692308	0.700000	...	0.694513	0.017187	4	0.699692	0.698535	0.689522	0.703005	0.703775	0.698906	0.005086
1	0.009973	0.001092	0.001595	0.000488	0.01	50	lbfgs	{'C': 0.01, 'max_iter': 50, 'solver': 'lbfgs'}	0.692308	0.700000	...	0.694513	0.017187	4	0.699692	0.698535	0.689522	0.703005	0.703775	0.698906	0.005086
2	0.009175	0.000747	0.001396	0.000488	0.01	50	liblinear	{'C': 0.01, 'max_iter': 50, 'solver': 'libline...	0.696923	0.701538	...	0.697596	0.016011	1	0.699306	0.700463	0.687211	0.703005	0.699538	0.697904	0.005505
3	0.041888	0.001546	0.001795	0.000747	0.01	50	sag	{'C': 0.01, 'max_iter': 50, 'solver': 'sag'}	0.692308	0.700000	...	0.694513	0.017187	4	0.699692	0.698535	0.689522	0.703005	0.703005	0.698752	0.004945
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
56	0.017752	0.006949	0.001795	0.000399	10	150	lbfgs	{'C': 10, 'max_iter': 150, 'solver': 'lbfgs'}	0.684615	0.696923	...	0.690506	0.020837	22	0.699306	0.697379	0.683359	0.700693	0.700693	0.696286	0.006577
57	0.019548	0.010973	0.007979	0.012473	10	150	liblinear	{'C': 10, 'max_iter': 150, 'solver': 'liblinear'}	0.684615	0.696923	...	0.690506	0.020837	22	0.699306	0.697379	0.683359	0.700693	0.700693	0.696286	0.006577
58	0.162365	0.014839	0.003192	0.001163	10	150	sag	{'C': 10, 'max_iter': 150, 'solver': 'sag'}	0.684615	0.696923	...	0.690506	0.020837	22	0.699306	0.697379	0.683359	0.700693	0.700693	0.696286	0.006577
59	0.037101	0.008450	0.001404	0.000497	10	150	saga	{'C': 10, 'max_iter': 150, 'solver': 'saga'}	0.684615	0.696923	...	0.690506	0.020837	22	0.699306	0.697379	0.683359	0.700693	0.700693	0.696286	0.006577
60 rows × 23 columns

print(model_LR.best_params_)
{'C': 0.01, 'max_iter': 50, 'solver': 'liblinear'}
model = LogisticRegression(**model_LR.best_params_)
model.fit(trainB[features], trainB[target])

importances = abs(model.coef_[0])
importances = 100.0 * (importances / importances.max())
indices = np.argsort(importances)

plt.figure(figsize = (15, 8))
plt.title('Feature Importances: Balanced Logistic Regression')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

pred = model_LR.predict(test[features])
predp = model_LR.predict_proba(test[features])[:,1]

cm = confusion_matrix(test[target], pred)
plot_confusion_matrix(cm, target_names = ['Not Exited', 'Exited'], normalize = False)
plot_confusion_matrix(cm, target_names = ['Not Exited', 'Exited'], normalize = True, title = 'Confusion Matrix (Normalized)')

temp = classification_report_to_dataframe(test[target], pred, predp, model_name = 'Logistic Regression', balanced = 'yes')
temp


Model	Balanced	Accuracy	Balanced_Accuracy	AUC	precision_0	recall_0	f1-score_0	precision_1	recall_1	f1-score_1
0	Logistic Regression	yes	0.716	0.707871	0.779976	0.900079	0.721767	0.80112	0.395062	0.693976	0.503497
table_of_models = table_of_models.append(temp)
table_of_models
Model	Balanced	Accuracy	Balanced_Accuracy	AUC	precision_0	recall_0	f1-score_0	precision_1	recall_1	f1-score_1
0	Logistic Regression	no	0.8115	0.592030	0.777608	0.825081	0.967192	0.890502	0.633803	0.216867	0.323160
0	Multi-Layer Perceptron (MLP)	no	0.8640	0.725651	0.871359	0.877950	0.962145	0.918122	0.771863	0.489157	0.598820
0	Gradient Boosting (Sklearn)	no	0.8600	0.715123	0.878344	0.873497	0.962776	0.915966	0.766798	0.467470	0.580838
0	Gradient Boosting (XGBoost)	no	0.8635	0.722667	0.875068	0.876579	0.963407	0.917944	0.775194	0.481928	0.594354
0	Gradient Boosting (LightGBM)	no	0.8690	0.737699	0.881359	0.883034	0.962145	0.920894	0.780220	0.513253	0.619186
0	Logistic Regression	yes	0.7160	0.707871	0.779976	0.900079	0.721767	0.801120	0.395062	0.693976	0.503497
2.5.2 MLP: Multi Layers Perceptron
s = len(features)
parameters = {'hidden_layer_sizes': [(s,),
                                     (s,)*2,
                                     (s,)*4,
                                     (s,)*6],
              'solver': ['lbfgs', 'adam'],
              'alpha': [0, 0.01, 0.1, 1, 10]}
MLP = MLPClassifier()
model_MLP = GridSearchCV(MLP, parameters, cv = 5, n_jobs = 10, verbose = 1).fit(trainB[features], trainB[target])
pd.DataFrame(model_MLP.cv_results_)
Fitting 5 folds for each of 40 candidates, totalling 200 fits
[Parallel(n_jobs=10)]: Using backend LokyBackend with 10 concurrent workers.
[Parallel(n_jobs=10)]: Done  30 tasks      | elapsed:   12.1s
[Parallel(n_jobs=10)]: Done 180 tasks      | elapsed:  1.3min
[Parallel(n_jobs=10)]: Done 200 out of 200 | elapsed:  1.5min finished
mean_fit_time	std_fit_time	mean_score_time	std_score_time	param_alpha	param_hidden_layer_sizes	param_solver	params	split0_test_score	split1_test_score	...	mean_test_score	std_test_score	rank_test_score	split0_train_score	split1_train_score	split2_train_score	split3_train_score	split4_train_score	mean_train_score	std_train_score
0	0.971005	0.040249	0.002593	0.000488	0	(13,)	lbfgs	{'alpha': 0, 'hidden_layer_sizes': (13,), 'sol...	0.775385	0.758462	...	0.765105	0.008247	16	0.814958	0.812259	0.808552	0.808166	0.819723	0.812731	0.004298
1	3.530164	0.532055	0.007779	0.008680	0	(13,)	adam	{'alpha': 0, 'hidden_layer_sizes': (13,), 'sol...	0.764615	0.772308	...	0.769420	0.004349	8	0.786430	0.782961	0.786595	0.780046	0.782357	0.783678	0.002511
2	1.295337	0.224860	0.003989	0.001262	0	(13, 13)	lbfgs	{'alpha': 0, 'hidden_layer_sizes': (13, 13), '...	0.758462	0.733846	...	0.754007	0.012649	21	0.841172	0.839630	0.834361	0.834746	0.827427	0.835467	0.004822
3	5.174967	1.054086	0.003192	0.000747	0	(13, 13)	adam	{'alpha': 0, 'hidden_layer_sizes': (13, 13), '...	0.770769	0.766154	...	0.766954	0.007814	15	0.804934	0.808404	0.800077	0.806626	0.803544	0.804717	0.002835
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
36	2.436695	0.287705	0.004588	0.002722	10	(13, 13, 13, 13)	lbfgs	{'alpha': 10, 'hidden_layer_sizes': (13, 13, 1...	0.750769	0.780000	...	0.773120	0.011921	5	0.814187	0.813801	0.811633	0.812404	0.820493	0.814504	0.003134
37	6.320260	1.559395	0.003590	0.001197	10	(13, 13, 13, 13)	adam	{'alpha': 10, 'hidden_layer_sizes': (13, 13, 1...	0.758462	0.733846	...	0.694513	0.097738	39	0.760601	0.737857	0.500000	0.757704	0.738829	0.698998	0.099938
38	3.384064	1.423817	0.003989	0.000631	10	(13, 13, 13, 13, 13, 13)	lbfgs	{'alpha': 10, 'hidden_layer_sizes': (13, 13, 1...	0.778462	0.783077	...	0.720407	0.110250	38	0.798381	0.807247	0.826656	0.500000	0.809322	0.748321	0.124498
39	5.085484	0.475452	0.007180	0.006922	10	(13, 13, 13, 13, 13, 13)	adam	{'alpha': 10, 'hidden_layer_sizes': (13, 13, 1...	0.500000	0.500000	...	0.500000	0.000000	40	0.500000	0.500000	0.500000	0.500000	0.500000	0.500000	0.000000
40 rows × 23 columns

print(model_MLP.best_params_)
{'alpha': 10, 'hidden_layer_sizes': (13, 13), 'solver': 'lbfgs'}
pred = model_MLP.predict(test[features])
predp = model_MLP.predict_proba(test[features])[:,1]

cm = confusion_matrix(test[target], pred)
plot_confusion_matrix(cm, target_names = ['Not Exited', 'Exited'], normalize = False)
plot_confusion_matrix(cm, target_names = ['Not Exited', 'Exited'], normalize = True, title = 'Confusion Matrix (Normalized)')

temp = classification_report_to_dataframe(test[target], pred, predp, model_name = 'Multi-Layer Perceptron (MLP)', balanced = 'yes')
temp


Model	Balanced	Accuracy	Balanced_Accuracy	AUC	precision_0	recall_0	f1-score_0	precision_1	recall_1	f1-score_1
0	Multi-Layer Perceptron (MLP)	yes	0.785	0.772749	0.870068	0.92432	0.793691	0.854039	0.488263	0.751807	0.59203
table_of_models = table_of_models.append(temp)
table_of_models
Model	Balanced	Accuracy	Balanced_Accuracy	AUC	precision_0	recall_0	f1-score_0	precision_1	recall_1	f1-score_1
0	Logistic Regression	no	0.8115	0.592030	0.777608	0.825081	0.967192	0.890502	0.633803	0.216867	0.323160
0	Multi-Layer Perceptron (MLP)	no	0.8640	0.725651	0.871359	0.877950	0.962145	0.918122	0.771863	0.489157	0.598820
0	Gradient Boosting (Sklearn)	no	0.8600	0.715123	0.878344	0.873497	0.962776	0.915966	0.766798	0.467470	0.580838
0	Gradient Boosting (XGBoost)	no	0.8635	0.722667	0.875068	0.876579	0.963407	0.917944	0.775194	0.481928	0.594354
0	Gradient Boosting (LightGBM)	no	0.8690	0.737699	0.881359	0.883034	0.962145	0.920894	0.780220	0.513253	0.619186
0	Logistic Regression	yes	0.7160	0.707871	0.779976	0.900079	0.721767	0.801120	0.395062	0.693976	0.503497
0	Multi-Layer Perceptron (MLP)	yes	0.7850	0.772749	0.870068	0.924320	0.793691	0.854039	0.488263	0.751807	0.592030
2.5.3 Gradient Boosting Classifier (Sklearn)
parameters = {'max_depth': [2, 3, 4, 6, 10, 15],
              'n_estimators': [50, 100, 300, 500]}
GB = GBSklearn()
model_GB = GridSearchCV(GB,
                        parameters,
                        cv = 5,
                        n_jobs = 10,
                        verbose = 1).fit(trainB[features], trainB[target])
pd.DataFrame(model_GB.cv_results_)
Fitting 5 folds for each of 24 candidates, totalling 120 fits
[Parallel(n_jobs=10)]: Using backend LokyBackend with 10 concurrent workers.
[Parallel(n_jobs=10)]: Done  30 tasks      | elapsed:    3.9s
[Parallel(n_jobs=10)]: Done 120 out of 120 | elapsed:  1.1min finished
mean_fit_time	std_fit_time	mean_score_time	std_score_time	param_max_depth	param_n_estimators	params	split0_test_score	split1_test_score	split2_test_score	...	mean_test_score	std_test_score	rank_test_score	split0_train_score	split1_train_score	split2_train_score	split3_train_score	split4_train_score	mean_train_score	std_train_score
0	0.261102	0.076714	0.002992	0.000631	2	50	{'max_depth': 2, 'n_estimators': 50}	0.766154	0.772308	0.779321	...	0.778360	0.008271	6	0.790671	0.787201	0.783513	0.790062	0.787365	0.787762	0.002541
1	0.549332	0.097615	0.004188	0.000399	2	100	{'max_depth': 2, 'n_estimators': 100}	0.758462	0.783077	0.782407	...	0.780826	0.011714	3	0.803392	0.794140	0.797381	0.801233	0.796225	0.798474	0.003372
2	1.681704	0.305911	0.008577	0.002721	2	300	{'max_depth': 2, 'n_estimators': 300}	0.749231	0.783077	0.777778	...	0.777127	0.015098	7	0.832305	0.827294	0.830508	0.835901	0.830123	0.831226	0.002836
3	2.862948	0.291889	0.015757	0.006350	2	500	{'max_depth': 2, 'n_estimators': 500}	0.760000	0.778462	0.770062	...	0.771270	0.007111	10	0.856207	0.853123	0.855547	0.859014	0.849384	0.854655	0.003235
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
20	5.828220	0.426890	0.012168	0.006892	15	50	{'max_depth': 15, 'n_estimators': 50}	0.701538	0.718462	0.723765	...	0.709309	0.011657	23	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	0.000000
21	8.936313	0.405102	0.018351	0.008191	15	100	{'max_depth': 15, 'n_estimators': 100}	0.710769	0.713846	0.733025	...	0.715166	0.011667	21	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	0.000000
22	10.787963	0.578032	0.013962	0.000631	15	300	{'max_depth': 15, 'n_estimators': 300}	0.706154	0.713846	0.723765	...	0.709309	0.010085	23	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	0.000000
23	8.083791	0.328812	0.011370	0.001739	15	500	{'max_depth': 15, 'n_estimators': 500}	0.706154	0.713846	0.739198	...	0.715166	0.012312	21	1.000000	1.000000	1.000000	1.000000	1.000000	1.000000	0.000000
24 rows × 22 columns

print(model_GB.best_params_)
{'max_depth': 4, 'n_estimators': 50}
model = GBSklearn(**model_GB.best_params_)
model.fit(trainB[features], trainB[target])

importances = model.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize = (15, 8))
plt.title('Feature Importances: Balanced Gradient Boosting (Sklearn)')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

pred = model_GB.predict(test[features])
predp = model_GB.predict_proba(test[features])[:,1]

cm = confusion_matrix(test[target], pred)
plot_confusion_matrix(cm, target_names = ['Not Exited', 'Exited'], normalize = False)
plot_confusion_matrix(cm, target_names = ['Not Exited', 'Exited'], normalize = True, title = 'Confusion Matrix (Normalized)')

temp = classification_report_to_dataframe(test[target], pred, predp, model_name = 'Gradient Boosting (Sklearn)', balanced = 'yes')
temp


Model	Balanced	Accuracy	Balanced_Accuracy	AUC	precision_0	recall_0	f1-score_0	precision_1	recall_1	f1-score_1
0	Gradient Boosting (Sklearn)	yes	0.799	0.794033	0.877831	0.934607	0.802524	0.863544	0.510172	0.785542	0.618596
table_of_models = table_of_models.append(temp)
table_of_models
Model	Balanced	Accuracy	Balanced_Accuracy	AUC	precision_0	recall_0	f1-score_0	precision_1	recall_1	f1-score_1
0	Logistic Regression	no	0.8115	0.592030	0.777608	0.825081	0.967192	0.890502	0.633803	0.216867	0.323160
0	Multi-Layer Perceptron (MLP)	no	0.8640	0.725651	0.871359	0.877950	0.962145	0.918122	0.771863	0.489157	0.598820
0	Gradient Boosting (Sklearn)	no	0.8600	0.715123	0.878344	0.873497	0.962776	0.915966	0.766798	0.467470	0.580838
0	Gradient Boosting (XGBoost)	no	0.8635	0.722667	0.875068	0.876579	0.963407	0.917944	0.775194	0.481928	0.594354
0	Gradient Boosting (LightGBM)	no	0.8690	0.737699	0.881359	0.883034	0.962145	0.920894	0.780220	0.513253	0.619186
0	Logistic Regression	yes	0.7160	0.707871	0.779976	0.900079	0.721767	0.801120	0.395062	0.693976	0.503497
0	Multi-Layer Perceptron (MLP)	yes	0.7850	0.772749	0.870068	0.924320	0.793691	0.854039	0.488263	0.751807	0.592030
0	Gradient Boosting (Sklearn)	yes	0.7990	0.794033	0.877831	0.934607	0.802524	0.863544	0.510172	0.785542	0.618596
2.5.4 Extreme Gradient Boosting (XGBoost)
model_XGB = XGB(max_depth = 6,
            learning_rate = .1,
            n_estimators = 100,
            reg_lambda = 0.5,
            reg_alpha = 0,
            verbosity = 1,
            n_jobs = -1,
            tree_method = 'gpu_exact').fit(trainB[features], trainB[target])

pred = model_XGB.predict(test[features])
predp = model_XGB.predict_proba(test[features])[:,1]

importances = model_XGB.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize = (15, 8))
plt.title('Feature Importances: Balanced Extreme Gradient Boosting (XGBoost)')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

cm = confusion_matrix(test[target], pred)
plot_confusion_matrix(cm, target_names = ['Not Exited', 'Exited'], normalize = False)
plot_confusion_matrix(cm, target_names = ['Not Exited', 'Exited'], normalize = True, title = 'Confusion Matrix (Normalized)')

temp = classification_report_to_dataframe(test[target], pred, predp, model_name = 'Gradient Boosting (XGBoost)', balanced = 'yes')
temp


Model	Balanced	Accuracy	Balanced_Accuracy	AUC	precision_0	recall_0	f1-score_0	precision_1	recall_1	f1-score_1
0	Gradient Boosting (XGBoost)	yes	0.791	0.785428	0.869156	0.931264	0.794953	0.857726	0.497682	0.775904	0.606403
table_of_models = table_of_models.append(temp)
table_of_models
Model	Balanced	Accuracy	Balanced_Accuracy	AUC	precision_0	recall_0	f1-score_0	precision_1	recall_1	f1-score_1
0	Logistic Regression	no	0.8115	0.592030	0.777608	0.825081	0.967192	0.890502	0.633803	0.216867	0.323160
0	Multi-Layer Perceptron (MLP)	no	0.8640	0.725651	0.871359	0.877950	0.962145	0.918122	0.771863	0.489157	0.598820
0	Gradient Boosting (Sklearn)	no	0.8600	0.715123	0.878344	0.873497	0.962776	0.915966	0.766798	0.467470	0.580838
0	Gradient Boosting (XGBoost)	no	0.8635	0.722667	0.875068	0.876579	0.963407	0.917944	0.775194	0.481928	0.594354
...	...	...	...	...	...	...	...	...	...	...	...
0	Logistic Regression	yes	0.7160	0.707871	0.779976	0.900079	0.721767	0.801120	0.395062	0.693976	0.503497
0	Multi-Layer Perceptron (MLP)	yes	0.7850	0.772749	0.870068	0.924320	0.793691	0.854039	0.488263	0.751807	0.592030
0	Gradient Boosting (Sklearn)	yes	0.7990	0.794033	0.877831	0.934607	0.802524	0.863544	0.510172	0.785542	0.618596
0	Gradient Boosting (XGBoost)	yes	0.7910	0.785428	0.869156	0.931264	0.794953	0.857726	0.497682	0.775904	0.606403
9 rows × 11 columns

2.5.5 Light Gradient Boosting Machine (LightGBM)
train_data = lgb.Dataset(trainB[features],
                         label = trainB[target],
                         feature_name = features)
test_data = lgb.Dataset(test[features + [target]],
                              reference = train_data)
param = {'num_leaves': 21,
         'num_trees': 100,
         # 'max_depth': 20,
         'objective': 'binary',
         # 'boosting': 'dart',
         'lambda_l1': 1,
         'lambda_l2': 1,
         'learning_rate': 0.1,
         'metric': ['binary_logloss', 'binary_error'],
         'seed': 1}

num_round = 10
model_LGB = lgb.train(param, train_data, num_round, valid_sets=[test_data])
[1]	valid_0's binary_logloss: 0.674855	valid_0's binary_error: 0.3585
[2]	valid_0's binary_logloss: 0.661192	valid_0's binary_error: 0.371
[3]	valid_0's binary_logloss: 0.651515	valid_0's binary_error: 0.343
[4]	valid_0's binary_logloss: 0.643541	valid_0's binary_error: 0.34
[5]	valid_0's binary_logloss: 0.638305	valid_0's binary_error: 0.343
[6]	valid_0's binary_logloss: 0.634442	valid_0's binary_error: 0.3455
[7]	valid_0's binary_logloss: 0.63218	valid_0's binary_error: 0.346
[8]	valid_0's binary_logloss: 0.632156	valid_0's binary_error: 0.3475
[9]	valid_0's binary_logloss: 0.63273	valid_0's binary_error: 0.333
[10]	valid_0's binary_logloss: 0.633086	valid_0's binary_error: 0.3385
[11]	valid_0's binary_logloss: 0.633887	valid_0's binary_error: 0.3385
[12]	valid_0's binary_logloss: 0.634107	valid_0's binary_error: 0.3325
[13]	valid_0's binary_logloss: 0.635837	valid_0's binary_error: 0.332
[14]	valid_0's binary_logloss: 0.636673	valid_0's binary_error: 0.328
[15]	valid_0's binary_logloss: 0.637302	valid_0's binary_error: 0.3235
[16]	valid_0's binary_logloss: 0.640015	valid_0's binary_error: 0.325
[17]	valid_0's binary_logloss: 0.641787	valid_0's binary_error: 0.3265
[18]	valid_0's binary_logloss: 0.644237	valid_0's binary_error: 0.325
[19]	valid_0's binary_logloss: 0.647128	valid_0's binary_error: 0.3245
[20]	valid_0's binary_logloss: 0.649842	valid_0's binary_error: 0.323
[21]	valid_0's binary_logloss: 0.651744	valid_0's binary_error: 0.329
[22]	valid_0's binary_logloss: 0.653234	valid_0's binary_error: 0.322
[23]	valid_0's binary_logloss: 0.655283	valid_0's binary_error: 0.32
[24]	valid_0's binary_logloss: 0.657884	valid_0's binary_error: 0.3195
[25]	valid_0's binary_logloss: 0.661102	valid_0's binary_error: 0.3215
[26]	valid_0's binary_logloss: 0.662855	valid_0's binary_error: 0.317
[27]	valid_0's binary_logloss: 0.665524	valid_0's binary_error: 0.3165
[28]	valid_0's binary_logloss: 0.66844	valid_0's binary_error: 0.318
[29]	valid_0's binary_logloss: 0.67079	valid_0's binary_error: 0.317
[30]	valid_0's binary_logloss: 0.67292	valid_0's binary_error: 0.3145
[31]	valid_0's binary_logloss: 0.674317	valid_0's binary_error: 0.313
[32]	valid_0's binary_logloss: 0.678197	valid_0's binary_error: 0.311
[33]	valid_0's binary_logloss: 0.679331	valid_0's binary_error: 0.3095
[34]	valid_0's binary_logloss: 0.680649	valid_0's binary_error: 0.312
[35]	valid_0's binary_logloss: 0.682471	valid_0's binary_error: 0.3135
[36]	valid_0's binary_logloss: 0.68587	valid_0's binary_error: 0.313
[37]	valid_0's binary_logloss: 0.687543	valid_0's binary_error: 0.311
[38]	valid_0's binary_logloss: 0.689293	valid_0's binary_error: 0.312
[39]	valid_0's binary_logloss: 0.690593	valid_0's binary_error: 0.3115
[40]	valid_0's binary_logloss: 0.692848	valid_0's binary_error: 0.3105
[41]	valid_0's binary_logloss: 0.694873	valid_0's binary_error: 0.311
[42]	valid_0's binary_logloss: 0.695681	valid_0's binary_error: 0.31
[43]	valid_0's binary_logloss: 0.698311	valid_0's binary_error: 0.31
[44]	valid_0's binary_logloss: 0.699809	valid_0's binary_error: 0.3115
[45]	valid_0's binary_logloss: 0.700166	valid_0's binary_error: 0.3135
[46]	valid_0's binary_logloss: 0.702634	valid_0's binary_error: 0.3135
[47]	valid_0's binary_logloss: 0.705251	valid_0's binary_error: 0.3115
[48]	valid_0's binary_logloss: 0.705316	valid_0's binary_error: 0.311
[49]	valid_0's binary_logloss: 0.704961	valid_0's binary_error: 0.312
[50]	valid_0's binary_logloss: 0.706142	valid_0's binary_error: 0.3125
[51]	valid_0's binary_logloss: 0.707862	valid_0's binary_error: 0.3115
[52]	valid_0's binary_logloss: 0.709321	valid_0's binary_error: 0.3115
[53]	valid_0's binary_logloss: 0.70994	valid_0's binary_error: 0.3105
[54]	valid_0's binary_logloss: 0.711326	valid_0's binary_error: 0.3125
[55]	valid_0's binary_logloss: 0.71261	valid_0's binary_error: 0.3135
[56]	valid_0's binary_logloss: 0.715217	valid_0's binary_error: 0.3135
[57]	valid_0's binary_logloss: 0.716139	valid_0's binary_error: 0.312
[58]	valid_0's binary_logloss: 0.716627	valid_0's binary_error: 0.31
[59]	valid_0's binary_logloss: 0.718432	valid_0's binary_error: 0.312
[60]	valid_0's binary_logloss: 0.719909	valid_0's binary_error: 0.3135
[61]	valid_0's binary_logloss: 0.721709	valid_0's binary_error: 0.312
[62]	valid_0's binary_logloss: 0.723188	valid_0's binary_error: 0.313
[63]	valid_0's binary_logloss: 0.723379	valid_0's binary_error: 0.311
[64]	valid_0's binary_logloss: 0.725941	valid_0's binary_error: 0.31
[65]	valid_0's binary_logloss: 0.727532	valid_0's binary_error: 0.312
[66]	valid_0's binary_logloss: 0.730869	valid_0's binary_error: 0.314
[67]	valid_0's binary_logloss: 0.731359	valid_0's binary_error: 0.313
[68]	valid_0's binary_logloss: 0.73275	valid_0's binary_error: 0.3145
[69]	valid_0's binary_logloss: 0.733935	valid_0's binary_error: 0.314
[70]	valid_0's binary_logloss: 0.735136	valid_0's binary_error: 0.3155
[71]	valid_0's binary_logloss: 0.736222	valid_0's binary_error: 0.315
[72]	valid_0's binary_logloss: 0.736744	valid_0's binary_error: 0.315
[73]	valid_0's binary_logloss: 0.737535	valid_0's binary_error: 0.3155
[74]	valid_0's binary_logloss: 0.738675	valid_0's binary_error: 0.314
[75]	valid_0's binary_logloss: 0.738953	valid_0's binary_error: 0.316
[76]	valid_0's binary_logloss: 0.74102	valid_0's binary_error: 0.317
[77]	valid_0's binary_logloss: 0.743186	valid_0's binary_error: 0.318
[78]	valid_0's binary_logloss: 0.743276	valid_0's binary_error: 0.318
[79]	valid_0's binary_logloss: 0.743936	valid_0's binary_error: 0.317
[80]	valid_0's binary_logloss: 0.744597	valid_0's binary_error: 0.321
[81]	valid_0's binary_logloss: 0.745211	valid_0's binary_error: 0.3205
[82]	valid_0's binary_logloss: 0.747052	valid_0's binary_error: 0.3175
[83]	valid_0's binary_logloss: 0.747176	valid_0's binary_error: 0.319
[84]	valid_0's binary_logloss: 0.748573	valid_0's binary_error: 0.319
[85]	valid_0's binary_logloss: 0.748937	valid_0's binary_error: 0.319
[86]	valid_0's binary_logloss: 0.749683	valid_0's binary_error: 0.319
[87]	valid_0's binary_logloss: 0.751761	valid_0's binary_error: 0.3185
[88]	valid_0's binary_logloss: 0.751901	valid_0's binary_error: 0.3195
[89]	valid_0's binary_logloss: 0.753578	valid_0's binary_error: 0.3205
[90]	valid_0's binary_logloss: 0.753708	valid_0's binary_error: 0.3205
[91]	valid_0's binary_logloss: 0.754531	valid_0's binary_error: 0.3195
[92]	valid_0's binary_logloss: 0.75447	valid_0's binary_error: 0.319
[93]	valid_0's binary_logloss: 0.756143	valid_0's binary_error: 0.3185
[94]	valid_0's binary_logloss: 0.757028	valid_0's binary_error: 0.319
[95]	valid_0's binary_logloss: 0.757049	valid_0's binary_error: 0.3205
[96]	valid_0's binary_logloss: 0.757901	valid_0's binary_error: 0.3225
[97]	valid_0's binary_logloss: 0.75817	valid_0's binary_error: 0.324
[98]	valid_0's binary_logloss: 0.758516	valid_0's binary_error: 0.323
[99]	valid_0's binary_logloss: 0.758753	valid_0's binary_error: 0.3215
[100]	valid_0's binary_logloss: 0.758991	valid_0's binary_error: 0.323
predp = model_LGB.predict(test[features])
pred = predp > 0.5

lgb.plot_importance(model_LGB,
                    figsize = (15,8),
                    height = 0.8,
                    title = 'Feature Importances: Balanced Light Gradient Boosting Machine (LightGBM)',
                    ylabel = None,
                    grid = False)
<matplotlib.axes._subplots.AxesSubplot at 0x23db34bdd30>

cm = confusion_matrix(test[target], pred)
plot_confusion_matrix(cm, target_names = ['Not Exited', 'Exited'], normalize = False)
plot_confusion_matrix(cm, target_names = ['Not Exited', 'Exited'], normalize = True, title = 'Confusion Matrix (Normalized)')

temp = classification_report_to_dataframe(test[target], pred, predp, model_name = 'Gradient Boosting (LightGBM)', balanced = 'yes')
temp


Model	Balanced	Accuracy	Balanced_Accuracy	AUC	precision_0	recall_0	f1-score_0	precision_1	recall_1	f1-score_1
0	Gradient Boosting (LightGBM)	yes	0.7945	0.790304	0.871768	0.93353	0.797476	0.860157	0.503096	0.783133	0.61263
table_of_models = table_of_models.append(temp)
table_of_models
Model	Balanced	Accuracy	Balanced_Accuracy	AUC	precision_0	recall_0	f1-score_0	precision_1	recall_1	f1-score_1
0	Logistic Regression	no	0.8115	0.592030	0.777608	0.825081	0.967192	0.890502	0.633803	0.216867	0.323160
0	Multi-Layer Perceptron (MLP)	no	0.8640	0.725651	0.871359	0.877950	0.962145	0.918122	0.771863	0.489157	0.598820
0	Gradient Boosting (Sklearn)	no	0.8600	0.715123	0.878344	0.873497	0.962776	0.915966	0.766798	0.467470	0.580838
0	Gradient Boosting (XGBoost)	no	0.8635	0.722667	0.875068	0.876579	0.963407	0.917944	0.775194	0.481928	0.594354
...	...	...	...	...	...	...	...	...	...	...	...
0	Multi-Layer Perceptron (MLP)	yes	0.7850	0.772749	0.870068	0.924320	0.793691	0.854039	0.488263	0.751807	0.592030
0	Gradient Boosting (Sklearn)	yes	0.7990	0.794033	0.877831	0.934607	0.802524	0.863544	0.510172	0.785542	0.618596
0	Gradient Boosting (XGBoost)	yes	0.7910	0.785428	0.869156	0.931264	0.794953	0.857726	0.497682	0.775904	0.606403
0	Gradient Boosting (LightGBM)	yes	0.7945	0.790304	0.871768	0.933530	0.797476	0.860157	0.503096	0.783133	0.612630
10 rows × 11 columns

table_of_models.to_excel('table_of_models.xlsx', index = False)
Conclusions
On this notebook, I went through the bank custumer churn data. My focus was to process the data for modelling, and try different algorithms to evaluate their performance.

The first step was to analize the features, try to understand them, and have some insights.

Later, I started to prepare the data for the modelling. First, I applied a one-hot-encoding over the cathegorical features. After, I splitted the data into the train and test sets, standardazing the features on each set. Then, the modelling was done in two parts, one with the complete train data, and another with a balanced train data. For each part I test the same models and algorithms:

Logistic Regression
Multi-Layer Perceptron
Gradient Boosting (Scikit-Learn)
Extreme Gradient Boosting (XGBoost)
Light Gradient Boosting Machine (LightGBM)
Initially, we tested the models performance. They all, when looking to the pure accuracy score behaved reasonably well. Only the logistic regression was one step behind. However, the data is imbalanced. 80% of the data are clients that didn't exited (labeled as 0). So, if we create a model that predicts 0 for all the clients, its accuracy will be of 80%. A better score metric is the balanced accuracy, that weights the classes by occurence. For the balanced accuracy, the baseline, for a binary classification, is of 50%. Looking at the balanced accuracy for all the predictions, the logistic regression did a poor job, with 59% accuracy. The other models had a balanced accuracy from 72% to 74%, with the lightGBM being slightly better, when also looking at the recall of the exited clients (labaled as 1).

Trying to improve the predictions for the exited clients (label 1), I proposed to balance the train data, by simply ramdonly removing clients labeled as 0 until the number of exited and not exited were almost the same. When I did that, I was expecting that the accuracy and the balanced accuracy to have similar values for each model, and that was exactly what happened. For all the models, the balanced accuracy increased to up to 79% (LightGBM), and showed that the tree based classifier models worked better.

By looking at the score metrics and speed performance, the model I would chose is the Gradient Boosting Classifier from the LightGBM package. But the XGBoost is close behind.

However, I still believe I can improve the accuracy by applying feature engineering on the data, as well trying other models, even doing an ensemble model over all the tested models.

 
