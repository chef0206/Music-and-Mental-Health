
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('mxmh_survey_results.csv')
# df.head()

df.drop(['Timestamp', 'Permissions'], axis=1, inplace=True)

from sklearn.impute import KNNImputer,SimpleImputer
impute = KNNImputer(n_neighbors=5)
simple_impute = SimpleImputer(missing_values='NAN', strategy='mean')
df['Age'] = impute.fit_transform(df['Age'].values.reshape(-1,1))
df['BPM'] = impute.fit_transform(df['BPM'].values.reshape(-1,1))
df['Primary streaming service'] = df['Primary streaming service'].fillna(df['Primary streaming service'].mode()[0])
df['While working'] = df['While working'].fillna(df['While working'].mode()[0])
df['Instrumentalist'] = df['Instrumentalist'].fillna(df['Instrumentalist'].mode()[0])
df['Composer'] = df['Composer'].fillna(df['Composer'].mode()[0])
df['Foreign languages'] = df['Foreign languages'].fillna(df['Foreign languages'].mode()[0])
df['Music effects'] = df['Music effects'].fillna(df['Music effects'].mode()[0])

# df.isnull().sum()


q1 = np.percentile(df['Age'],25)

q3 = np.percentile(df['Age'],75)
iqr = q3 - q1
lb = q1 - (1.5 * iqr)
up = q3 + (1.5 * iqr)

def to_detect_outliers(X,features):
    final_outlier_index = []
    for col in features:
        q1 = np.percentile(X[col],25)
        q3 = np.percentile(X[col],75)
        IQR = (q3 - q1) * 1.5
        lower_limit = q1 - IQR
        upper_limit = q3 + IQR
        outlier_index = X[col][(X[col]<lower_limit)|(X[col]>upper_limit)].index.to_list()
        final_outlier_index.extend(outlier_index)
    out_index = list(set(final_outlier_index))
    out_index.sort()
    # print(out_index)
    # print("----------------------------------------------------------------------")
    # print(f'There are {len(out_index)} records which fall under outliers' )
    # print("----------------------------------------------------------------------")
    # print(f'the percentage of outliers in the dataset is : {round((len(out_index)/len(df)*100),2)}% ')

feature = [feature for feature in df.columns if df[feature].dtypes != 'O' ]
features = ['Age', 'Hours per day', 'BPM']
to_detect_outliers(df,features)

np.where(df['Age']>up,up,np.where(df['Age']<lb,lb,df['Age']))

def capping_outliers(df,feature):
    for col in feature:
        q1 = np.percentile(df[col],25)
        q3 = np.percentile(df[col],75)
        IQR = (q3-q1) * 1.5
        upper_bond = q3 + IQR
        lower_bond = q1 - IQR
        df[col] = np.where(df[col]>upper_bond,upper_bond,np.where(df[col]<lower_bond,lower_bond,df[col]))
        #name_of_col      if is true , replace it with upper bond , else (replace is lower bond if both 
        # condition are failed then keep remaining data as it  )

capping_outliers(df,features)


numerical_feature = [feature for feature in df.columns if df[feature].dtypes != 'O' ]
numerical_features = numerical_feature[:-1]
# numerical_features

categorical_feature = [feature for feature in df.columns if df[feature].dtypes == 'O' ]
index = [df.columns.get_loc(c) for c in numerical_feature]
# numerical_features


df['Music effects'] = df['Music effects'].map({'Improve':0,'No effect':1,'Worsen':2})


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import category_encoders as ce
from sklearn.ensemble import GradientBoostingClassifier

numerical_process = Pipeline(
    steps = [('scaler',StandardScaler())]
)

from sklearn import set_config
set_config(display="diagram")

# numerical_process

lr = LabelEncoder()

cateogtical_process_1 = Pipeline(
    steps = [('lr', ce.TargetEncoder()),
             ('scaler',StandardScaler())]
    
)

cateogtical_process_2 = Pipeline(
    steps = [('lr',OneHotEncoder(sparse=False)),
             ('scaler',StandardScaler())]
    
)

X= df.iloc[:,0:-1]
y = df.iloc[:,-1]

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)

processsor_1 = ColumnTransformer(
    [('Categorical_encoding',cateogtical_process_1,X.select_dtypes(include="object").columns),
    ('numerical_encoding',numerical_process,X.select_dtypes(exclude="object").columns)]

)


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()

from sklearn.pipeline import make_pipeline
pipe_1 = make_pipeline(processsor_1,RandomForestClassifier())
pipe_2 = make_pipeline(processsor_1,DecisionTreeClassifier())
pipe_3 = make_pipeline(processsor_1,SVC(kernel='rbf',decision_function_shape='ovo'))
pipe_4 = make_pipeline(processsor_1,SVC(kernel='poly',decision_function_shape='ovo'))
pipe_5 = make_pipeline(processsor_1,KNeighborsClassifier())
pipe_6 = make_pipeline(processsor_1, GradientBoostingClassifier())

pipelines = [pipe_1, pipe_2, pipe_3, pipe_4, pipe_5, pipe_6]

for pipe in pipelines:
    pipe.fit(x_train,y_train)



models = {0: 'Random Forest Classifier', 1: 'Decision Tree Classifier', 2: 'SVC kernel: rbf', 
        3:'SVC kernel: poly', 4: 'K Neighbors Classifier', 5: 'Gradient Boosting Classifier'}


for i,model in enumerate(pipelines):
    print("{} test accuracy : {}".format(models[i],model.score(x_train,y_train)))
    print("-----------------------------------------------------------------------")

for i,model in enumerate(pipelines):
    print("{} test accuracy : {}".format(models[i],model.predict(x_test)))
    print("-----------------------------------------------------------------------")

import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(pipe_4, f)
