#################### Business Problem ####################
# When the characteristics are specified, people with diabetes
# be able to predict whether they are sick or not.
# develop a machine learning model



#################### Dataset Story #######################
# The dataset is available at the National Institutes of Diabetes-Digestive-Kidney Diseases in the USA.
# It is part of the large data set.
# 21 living in Phoenix, the 5th largest city in the State of Arizona in the USA
# Diabetes study of Pima Indian women aged and older
# data used for.
# It consists of 768 observations and 8 numerical independent variables.
# The target variable is specified as "outcome";
# 1 diabetes test result being positive,
# 0 indicates negative.

#################### Task ################################
# Using data preprocessing and feature engineering techniques
# develop a diabet forecasting model.


#################### Variables ###########################

# Pregnancies – Number of pregnancies
# Glucose – Glucose
# 2-hour plasma glucose concentration in the oral glucose tolerance test
# SkinThickness – Skin Thickness
# Insulin – 2-hour serum insulin (mu U/ml)
# Blood Pressure (mm Hg)
# Variables
# DiabetesPedigreeFunction: – Function
# 2-hour plasma glucose concentration in the oral glucose tolerance test
# Age – Age (years)
# Outcome: Have the disease (1) or not (0)

#################### Importing Libraries ###################
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import RobustScaler, LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split, cross_validate

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)


#################### Loading the Dataset ####################
def load_diabets():
    data = pd.read_csv(r"C:\Users\WIN\OneDrive\Masaüstü\DSMLBC\datasets\diabetes.csv")
    return data
df = load_diabets()


#################### Data Overview ##########################
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df, head=10)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    # Returns the names of categorical, numeric and categorical but cardinal variables in the data set
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]

    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, cat_but_car = grab_col_names(df)


def cat_summary(dataframe, num_but_cat, plot=False):
    print(pd.DataFrame({num_but_cat: dataframe[num_but_cat].value_counts(),
                        "ratio": 100 * dataframe[num_but_cat].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[num_but_cat], data=dataframe)
        plt.show()


cat_summary(df, "Outcome")
cat_summary(df, "Outcome", plot=True)

#################### Outliers #################################

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    # Determination of the threshold value
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    # Outlier detection
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def grab_outliers(dataframe, col_name, index=False):
    # Access outliers.
    low, up = outlier_thresholds(dataframe, col_name)
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

def remove_outlier(dataframe, col_name):
    # Delete outliers
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers
check_df(df, head=10)



for col in num_cols:
    remove_outlier(df, col)


#################### Missing Values ###########################

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")



zero_columns = [col for col in df.columns if (df[col].min() == 0 ) and col not in ["Pregnancies", "Outcome"] ]

for col in zero_columns:
    df[col] = np.where(df[col] == 0, np.nan, df[col])


df.isnull().sum()

na_columns = missing_values_table(df, True)

missing_vs_target(df, "Outcome", na_columns)


# Filling in Missing Observations in Categorical Variable Breakdown
def median_target(col):
    temp = df[df[col].notnull()]
    temp = temp[[col, 'Outcome']].groupby(['Outcome'])[[col]].median().reset_index()
    return temp


for col in na_columns:
    df.loc[(df["Outcome"] == 0) & (df[col].isnull()), col] = median_target(col)[col][0]
    df.loc[(df["Outcome"] == 1) & (df[col].isnull()), col] = median_target(col)[col][1]



missing_values_table(df)

check_df(df)


#################### Feature Engineering #####################

def diabets_feature_engineering(dataframe):
    #age level
    dataframe.loc[(dataframe['Age'] < 18), 'NEW_AGE_CAT'] = 'young'
    dataframe.loc[(dataframe['Age'] >= 18) & (dataframe['Age'] < 56), 'NEW_AGE_CAT'] = 'mature'
    dataframe.loc[(dataframe['Age'] >= 56), 'NEW_AGE_CAT'] = 'senior'


    # BMI level
    dataframe.loc[(dataframe['BMI'] <= 20.0), 'NEW_BMI_CAT'] = 'Underweight'
    dataframe.loc[(dataframe['BMI'] > 20.0) & (dataframe['BMI'] <= 25.0), 'NEW_BMI_CAT'] = 'Healthy'
    dataframe.loc[(dataframe['BMI'] > 25.0) & (dataframe['BMI'] <= 30.0), 'NEW_BMI_CAT'] = 'Overweight'
    dataframe.loc[(dataframe['BMI'] > 30.0) & (dataframe['BMI'] <= 40.0), 'NEW_BMI_CAT'] = 'Obese'
    dataframe.loc[(dataframe['BMI'] > 40.0), 'NEW_BMI_CAT'] = 'Very Obese'

    #Insulin level
    dataframe.loc[(dataframe['Insulin'] <= 149), 'NEW_INSULIN_CAT'] = 'Excellent'
    dataframe.loc[(dataframe['Insulin'] > 150) & (dataframe['Insulin'] <= 180), 'NEW_INSULIN_CAT'] = 'Good'
    dataframe.loc[(dataframe['Insulin'] > 180), 'NEW_INSULIN_CAT'] = 'Poor'


    #Glucose level
    dataframe.loc[(dataframe['Glucose'] <= 60), 'NEW_GLUCOSE_CAT'] = 'Hypoglycemia'
    dataframe.loc[(dataframe['Glucose'] > 60) & (dataframe['Glucose'] <= 70), 'NEW_GLUCOSE_CAT'] = 'Earl Hypoglycemia'
    dataframe.loc[(dataframe['Glucose'] > 70) & (dataframe['Glucose'] <= 140), 'NEW_GLUCOSE_CAT'] = 'Normal'
    dataframe.loc[(dataframe['Glucose'] > 140) & (dataframe['Glucose'] <= 200), 'NEW_GLUCOSE_CAT'] = 'Early Diabetic'
    dataframe.loc[(dataframe['Glucose'] > 200), 'NEW_GLUCOSE_CAT'] = 'Diabetic'

    #Blood Pressure level
    dataframe.loc[(dataframe['BloodPressure'] <= 80), 'NEW_BLOODPRESSURE_CAT'] = 'Normal'
    dataframe.loc[(dataframe['BloodPressure'] > 80) & (dataframe['BloodPressure'] <= 89), 'NEW_BLOODPRESSURE_CAT'] = 'Hypertension Stage 1'
    dataframe.loc[(dataframe['BloodPressure'] > 89) & (dataframe['BloodPressure'] <= 120), 'NEW_BLOODPRESSURE_CAT'] = 'Hypertension Stage 2'
    dataframe.loc[(dataframe['BloodPressure'] > 120), 'NEW_BLOODPRESSURE_CAT'] = 'Seek emergency Case'

    return dataframe
df_prep = diabets_feature_engineering(df)
cat_cols, num_cols, cat_but_car = grab_col_names(df_prep)

#################### Encoding #################################


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

def rare_encoder(dataframe, rare_perc, num_but_cat):
    # 1'den fazla rare varsa düzeltme yap. durumu göz önünde bulunduruldu.
    # rare sınıf sorgusu 0.01'e göre yapıldıktan sonra gelen true'ların sum'ı alınıyor.
    # eğer 1'den büyük ise rare cols listesine alınıyor.
    rare_columns = [col for col in num_but_cat if (dataframe[col].value_counts() / len(dataframe) < 0.01).sum() > 1]

    for col in rare_columns:
        tmp = dataframe[col].value_counts() / len(dataframe)
        rare_labels = tmp[tmp < rare_perc].index
        dataframe[col] = np.where(dataframe[col].isin(rare_labels), 'Rare', dataframe[col])

    return dataframe

def rare_analyser(dataframe, target, num_but_cat):
    for col in num_but_cat:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

        def rare_encoder(dataframe, rare_perc, num_but_cat):

            rare_columns = [col for col in num_but_cat if
                            (dataframe[col].value_counts() / len(dataframe) < 0.01).sum() > 1]

            for col in rare_columns:
                tmp = dataframe[col].value_counts() / len(dataframe)
                rare_labels = tmp[tmp < rare_perc].index
                dataframe[col] = np.where(dataframe[col].isin(rare_labels), 'Rare', dataframe[col])

            return dataframe

# Label Encoding
binary_cols = [col for col in df_prep.columns if df_prep[col].dtype not in [int, float]
               and df_prep[col].nunique() == 2]

for col in binary_cols:
    df_prep = label_encoder(df_prep, col)

# Rare Encoding
df_prep = rare_encoder(df_prep, 0.01, cat_cols)

# One-Hot Encoding
ohe_cols = [col for col in df_prep.columns if 10 >= df_prep[col].nunique() > 2]
df_prep = one_hot_encoder(df_prep, ohe_cols)
# cat_cols, num_cols, cat_but_car = grab_col_names(df)

df.head()

#################### Scaling ##############################

scaler = StandardScaler()
df_prep[num_cols] = scaler.fit_transform(df_prep[num_cols])

check_df(df_prep, head=10)
#################### Model ###################################
y = df_prep["Outcome"]
X = df_prep.drop(["Outcome"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)


log_model = LogisticRegression().fit(X, y)

cv_results = cross_validate(log_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])


cv_results['test_accuracy'].mean()

cv_results['test_precision'].mean()

cv_results['test_recall'].mean()

cv_results['test_f1'].mean()

cv_results['test_roc_auc'].mean()





random_user = X.sample(1, random_state=44)

log_model.predict(random_user)


y_pred = log_model.predict(X)



# Confusion Matrix
def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

plot_confusion_matrix(y, y_pred)


# Başarı skorları:
print(classification_report(y, y_pred))


# ROC AUC
y_prob = log_model.predict_proba(X)[:, 1]# 1 is written because it is desired to focus on the 1 class
roc_auc_score(y, y_prob)
