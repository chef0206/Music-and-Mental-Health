import pickle
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore", category = FutureWarning)
warnings.filterwarnings("ignore", category = DeprecationWarning)
import plotly.express as px

# """## Load and Clean Data"""

data = pd.read_csv('mxmh_survey_results.csv')

# Set Streamlit page configuration settings
st.set_page_config(
    page_title="CS312 | Longevity Potential Analysis on World Bank Parameters",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define CSS to hide the footer element
hide_footer_css = """
<style>
footer {
    visibility: hidden;
}
</style>
"""
# Render the CSS using Streamlit's markdown function
st.markdown(hide_footer_css, unsafe_allow_html=True)


(rows, cols) = data.shape

# data.dropna(inplace = True, axis = 0)

data.rename(columns={
    'Frequency [Classical]': 'Classical',          
    'Frequency [Country]': 'Country',               
    'Frequency [EDM]': 'EDM',                  
    'Frequency [Folk]': 'Folk',                  
    'Frequency [Gospel]': 'Gospel',                
    'Frequency [Hip hop]': 'Hip hop',               
    'Frequency [Jazz]': 'Jazz',                  
    'Frequency [K pop]': 'K pop',                 
    'Frequency [Latin]': 'Latin',                 
    'Frequency [Lofi]': 'Lofi',                  
    'Frequency [Metal]': 'Metal',                 
    'Frequency [Pop]': 'Pop',                   
    'Frequency [R&B]': 'R&B',                  
    'Frequency [Rap]': 'Rap',               
    'Frequency [Rock]': 'Rock',                 
    'Frequency [Video game music]': 'Games music'      
}, inplace=True)

def cleaned_data():
    data.drop(['Timestamp', 'Permissions'], axis=1, inplace=True)
    data.rename(columns={
    'Frequency [Classical]': 'Classical',          
    'Frequency [Country]': 'Country',               
    'Frequency [EDM]': 'EDM',                  
    'Frequency [Folk]': 'Folk',                  
    'Frequency [Gospel]': 'Gospel',                
    'Frequency [Hip hop]': 'Hip hop',               
    'Frequency [Jazz]': 'Jazz',                  
    'Frequency [K pop]': 'K pop',                 
    'Frequency [Latin]': 'Latin',                 
    'Frequency [Lofi]': 'Lofi',                  
    'Frequency [Metal]': 'Metal',                 
    'Frequency [Pop]': 'Pop',                   
    'Frequency [R&B]': 'R&B',                  
    'Frequency [Rap]': 'Rap',               
    'Frequency [Rock]': 'Rock',                 
    'Frequency [Video game music]': 'Games music'      
    }, inplace=True)

    feature = [feature for feature in data.columns if data[feature].dtypes != 'O' ]
    features = ['Age', 'Hours per day', 'BPM']
    r = to_detect_outliers(data,features)

    data['BPM'].fillna(data['BPM'].mode()[0], inplace=True)
    data['Music effects'].fillna(data['Music effects'].mode()[0], inplace=True)
    data['Age'].fillna(data['Age'].mode()[0], inplace=True)
    data['Primary streaming service'].fillna(data['Primary streaming service'].mode()[0], inplace=True)
    data['While working'].fillna(data['While working'].mode()[0], inplace=True)
    data['Instrumentalist'].fillna(data['Instrumentalist'].mode()[0], inplace=True)
    data['Composer'].fillna(data['Composer'].mode()[0], inplace=True)
    data['Foreign languages'].fillna(data['Foreign languages'].mode()[0], inplace=True)

    def capping_outliers(df,feature):
        for col in feature:
            q1 = np.percentile(df[col],25)
            q3 = np.percentile(df[col],75)
            IQR = (q3-q1) * 1.5
            upper_bond = q3 + IQR
            lower_bond = q1 - IQR
            df[col] = np.where(df[col]>upper_bond,upper_bond,np.where(df[col]<lower_bond,lower_bond,df[col]))
    capping_outliers(data, ['Age', 'Hours per day', 'BPM'])

    bins= [0, 13, 30, 50]
    labels = ['Child', 'Teenagers', 'Adults']
    data['Age Group'] = pd.cut(data['Age'], bins=bins, labels=labels, right=False)

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
    return round((len(out_index)/len(data)*100),3)

# """As the Data is collected through form let's look at some respondents background"""
def capping_outliers(data,feature):
    for col in feature:
        q1 = np.percentile(data[col],25)
        q3 = np.percentile(data[col],75)
        IQR = (q3-q1) * 1.5
        upper_bond = q3 + IQR
        lower_bond = q1 - IQR
        data[col] = np.where(data[col]>upper_bond,upper_bond,np.where(data[col]<lower_bond,lower_bond,data[col]))

def homepage():
    # Add title and description
    st.title("Music and Mental Health")
    st.write("Music and mental health are interconnected in many ways. Listening to music can have a significant impact on our emotions, mood, and overall well-being. Numerous studies have shown that music can help reduce symptoms of depression, anxiety, and other mental health disorders.")

    st.write("Music therapy, which involves the use of music to promote healing and improve well-being, is a well-established form of therapy for people with mental health conditions. It has been shown to be effective in treating a wide range of mental health conditions, including depression, anxiety, and post-traumatic stress disorder (PTSD).")

    st.image("logos.png", use_column_width=True)
    # fig = px.pie(data, values=data['Primary streaming service'].value_counts(), names=data['Primary streaming service'].unique())
    # fig.update_layout(template = 'plotly_dark', title = dict(text = 'Primary streaming service', font=dict(size=25)), height = 600, width = 1200,)
    # st.plotly_chart(fig, use_container_width=True)

    # fig = px.histogram(x = data['Age'], y = data['Hours per day'])
    # fig.update_layout(template = 'plotly_dark', title = dict(text = 'Age and Sum of Hours Per Day', font=dict(size=25)), xaxis_title = 'Age',yaxis_title = 'Sum of Hours')
    # st.plotly_chart(fig, use_container_width=True)


def dataset():
    st.title("Dataset")
    st.markdown("""The music and mental health dataset consists of responses from 
            individuals about their music listening habits and mental health status. 
            The dataset includes information about the individuals' age, primary streaming service, 
            hours per day spent listening to music, whether they listen to music while working, whether they 
            play an instrument or compose music, their favorite genre, their exploratory 
            behavior in music listening, their knowledge of foreign languages, their frequency 
            of listening to various genres of music, and the effects of music on their mental health, 
            such as anxiety, depression, insomnia, and OCD. The dataset aims to explore the 
            relationship between music listening habits and mental health status.""")

    # Calculate the correlation matrix
    corr_matrix = data.corr()

    # Create a heatmap of the correlation matrix
    fig = px.imshow(corr_matrix, 
                    color_continuous_scale='RdBu_r',
                    title='Correlation Matrix of Music and Mental Health Dataset')
    
    fig.update_layout(template = 'plotly_dark', title = dict(text = 'Correlation Matrix of Music and Mental Health Dataset', font=dict(size=25)),
                height = 700,
                width = 1200,
                font=dict(
                family="Courier New, monospace",
                size=25,
                color="White"
    ))
    st.plotly_chart(fig, height = 700, use_container_width=True)

    st.write()

    st.markdown("""
    #### Here is the summary of indicators analyzed:

    * `Age` : The age of the respondent.
    
    * `Primary streaming service` : The main music streaming service used by the respondent. This column provides information on the music streaming habits of the respondents.
    
    * `Hours per day` : The number of hours per day the respondent spends listening to music. This column provides information on the music listening habits of the respondents.
    While working: Whether the respondent listens to music while working. This column provides information on the work habits of the respondents.

    * `Instrumentalist` : Whether the respondent plays a musical instrument. This column provides information on the musical skills and interests of the respondents.

    * `Composer` : Whether the respondent composes music. This column provides information on the musical skills and interests of the respondents.

    * `Fav genre` : The favorite music genre of the respondent. This column provides information on the musical preferences of the respondents.

    * `Exploratory` : Whether the respondent likes to explore new music genres. This column provides information on the musical preferences of the respondents.

    * `Foreign languages` : Whether the respondent listens to music in languages other than their primary language. This column provides information on the music listening habits of the respondents.

    * `BPM` : The preferred Beats Per Minute (BPM) of the respondent. This column provides information on the music listening habits of the respondents.

    * `Frequency`: Frequency of listening of each genre into four categories Never, Rarely, Sometimes, Very Frequently. The various genres listed are as follows:
        - Classical
        - Country
        - EDM 
        - Folk
        - Gospel
        - Hip Hop
        - Jazz
        - K pop
        - Latin
        - Lofi
        - Metal
        - Pop
        - R&B
        - Rap
        - Rock
        - Video game music
    
    * `Depression` : Depression level of respondend on a scale of one to ten.

    * `Anxiety` : Anxiety level of respondend on a scale of one to ten.

    * `Insomnia` : Insomnia level of respondend on a scale of one to ten.

    * `OCD` : OCD level of respondend on a scale of one to ten.

    * `Music Effects` : The last feature shows the effect of music people had on them. Some people showed positive result, some had no effect.

    """)

def datacleaning():
    st.title("Data Cleaning")
    st.markdown("""
    -  Number of Rows: 736
    - Number of Columns: 33
    """)

    st.markdown("""
    ### Description of Dataset
    """) 
    st.table(data.describe())
    st.markdown("""
    - Age: We have records of people ranging from 10 to 89 years of age in the survey

    - Hours per day: The maximum record in this column indicates a peculiar fact, 
    someone reported listening to music 24 hours a day.

    - BPM: Clearly we have the presence of outliers, the maximum value is 999999999

- may have resulted in research participants entering values they found correct or simply random values just to fill out the form, as an example of this the record of 999999999
    """)
    
    st.markdown("""
    ### Total Null Values
    """)
    st.table(pd.DataFrame({'Age': 1, 'Primary streaming service': 1, 'While working': 3, 'Instrumentalist': 4,
            'Composer': 1, 'BPM':107, 'Music effects': 8}, index=[0]))
    
    st.markdown("""
    ### Filling Null Values
    Since distribution of the data is skewed and there are outliers, 
    filling the null values with the mean or median may not be a good choice. 
    In this case, filling the null values with the mode can be a better choice as it is less affected by outliers.
    """)

    data.drop(['Timestamp', 'Permissions'], axis=1, inplace=True)
    data.rename(columns={
    'Frequency [Classical]': 'Classical',          
    'Frequency [Country]': 'Country',               
    'Frequency [EDM]': 'EDM',                  
    'Frequency [Folk]': 'Folk',                  
    'Frequency [Gospel]': 'Gospel',                
    'Frequency [Hip hop]': 'Hip hop',               
    'Frequency [Jazz]': 'Jazz',                  
    'Frequency [K pop]': 'K pop',                 
    'Frequency [Latin]': 'Latin',                 
    'Frequency [Lofi]': 'Lofi',                  
    'Frequency [Metal]': 'Metal',                 
    'Frequency [Pop]': 'Pop',                   
    'Frequency [R&B]': 'R&B',                  
    'Frequency [Rap]': 'Rap',               
    'Frequency [Rock]': 'Rock',                 
    'Frequency [Video game music]': 'Games music'      
    }, inplace=True)

    feature = [feature for feature in data.columns if data[feature].dtypes != 'O' ]
    features = ['Age', 'Hours per day', 'BPM']
    r = to_detect_outliers(data,features)

    data['BPM'].fillna(data['BPM'].mode()[0], inplace=True)
    data['Music effects'].fillna(data['Music effects'].mode()[0], inplace=True)
    data['Age'].fillna(data['Age'].mode()[0], inplace=True)
    data['Primary streaming service'].fillna(data['Primary streaming service'].mode()[0], inplace=True)
    data['While working'].fillna(data['While working'].mode()[0], inplace=True)
    data['Instrumentalist'].fillna(data['Instrumentalist'].mode()[0], inplace=True)
    data['Composer'].fillna(data['Composer'].mode()[0], inplace=True)
    data['Foreign languages'].fillna(data['Foreign languages'].mode()[0], inplace=True)

    st.markdown("""
    ### Detecting Outliers
    """)
    st.write(f"Total Number of Outliers in the Dataset: {r}%")

    fig = px.box(data['Age'], template='plotly_dark')

    fig.update_layout(template = 'plotly_dark', title = dict(text = 'Age', font=dict(size=25)),
                    xaxis_title = 'Age', yaxis_title = 'Values', 
                    font=dict(
                    family="Courier New, monospace",
                    size=15,
                    color="White"
        ))
    st.plotly_chart(fig, use_container_width=True)

    fig = px.box(data['Hours per day'], template='plotly_dark')
    fig.update_layout(template = 'plotly_dark', title = dict(text = 'Hours Per Day', font=dict(size=25)),
                    xaxis_title = 'Hours Per Day', yaxis_title = 'Values', 
                    font=dict(
                    family="Courier New, monospace",
                    size=15,
                    color="White"
        ))
    st.plotly_chart(fig, use_container_width=True)

    fig = px.box(data['BPM'], template='plotly_dark')
    fig.update_layout(template = 'plotly_dark', title = dict(text = 'BPM', font=dict(size=25)),
                    xaxis_title = 'BPM', yaxis_title = 'Values', 
                    font=dict(
                    family="Courier New, monospace",
                    size=15,
                    color="White"
        ))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('### Records that consist of more than 12 hours listening to Music:\n')
    st.write(f'{len(data[data["Hours per day"] > 12])} records, ', "{:.2f}% of data".format((len(data[data['Hours per day'] > 12]) / data.shape[0]) * 100))

    st.markdown("""### Records above 180 BPM:""")
    st.write(f'{len(data[data["BPM"] > 180])} records, ', "{:.2f}% of data".format((len(data[data['BPM'] > 180]) / data.shape[0]) * 100))

    index = data[data['BPM'] == 999999999.0].index
    data.drop(index, inplace=True)

    st.markdown("""## Capping Outliers""")
    st.markdown("""
    The Interquartile Range (IQR) method is commonly used for outlier 
    detection and treatment because it is robust to the presence of extreme values (i.e., outliers) in the data.
    Capping outliers using the IQR method can improve the robustness of statistical analysis, 
    reduce the impact of extreme values on model performance, 
    and prevent bias in the estimation of summary statistics. 
    """)
    def capping_outliers(df,feature):
        for col in feature:
            q1 = np.percentile(df[col],25)
            q3 = np.percentile(df[col],75)
            IQR = (q3-q1) * 1.5
            upper_bond = q3 + IQR
            lower_bond = q1 - IQR
            df[col] = np.where(df[col]>upper_bond,upper_bond,np.where(df[col]<lower_bond,lower_bond,df[col]))
    capping_outliers(data,features)

    fig = px.box(data['Age'], template='plotly_dark')
    fig.update_layout(template = 'plotly_dark', title = dict(text = 'Age', font=dict(size=25)),
                    xaxis_title = 'Age', yaxis_title = 'Values', 
                    font=dict(
                    family="Courier New, monospace",
                    size=15,
                    color="White"
        ))
    st.plotly_chart(fig, use_container_width=True)

    fig = px.box(data['Hours per day'], template='plotly_dark')
    fig.update_layout(template = 'plotly_dark', title = dict(text = 'Hours per day', font=dict(size=25)),
                    xaxis_title = 'Hours per day', yaxis_title = 'Values', 
                    font=dict(
                    family="Courier New, monospace",
                    size=15,
                    color="White"
        ))
    st.plotly_chart(fig, use_container_width=True)

    fig = px.box(data['BPM'], template='plotly_dark')
    fig.update_layout(template = 'plotly_dark', title = dict(text = 'BPM', font=dict(size=25)),
                    xaxis_title = 'BPM', yaxis_title = 'Values', 
                    font=dict(
                    family="Courier New, monospace",
                    size=15,
                    color="White"
        ))
    st.plotly_chart(fig, use_container_width=True)

def hours_per_day():

    st.title("Hours of Listening")
    cleaned_data()
    anx_mean_original = data.query('`Hours per day` < 7')['Anxiety'].mean()
    dep_mean_original = data.query('`Hours per day` < 7')['Depression'].mean()
    ins_mean_original = data.query('`Hours per day` < 7')['Insomnia'].mean()
    ocd_mean_original = data.query('`Hours per day` < 7')['OCD'].mean()

    # Averaging points assigned to mental conditions from outlier records:
    anx_mean_outlier = data.query('`Hours per day` >= 7')['Anxiety'].mean()
    dep_mean_outlier = data.query('`Hours per day` >= 7')['Depression'].mean()
    ins_mean_outlier = data.query('`Hours per day` >= 7')['Insomnia'].mean()
    ocd_mean_outlier = data.query('`Hours per day` >= 7')['OCD'].mean()

    over_12 = [anx_mean_outlier, dep_mean_outlier, ins_mean_outlier, ocd_mean_outlier]
    under_12 = [anx_mean_original, dep_mean_original, ins_mean_original, ocd_mean_original]
    condition = ['Anxiety', 'Depression', 'Insomnia', 'OCD']
    
        # Create the trace for the upper bar
    trace_over_12 = go.Scatter(
        x=over_12,
        y=condition,
        mode='markers',
        name='Over 7',
        marker=dict(
            color='#1f77b4',
            size=18,
            line=dict(width=1)
        )
    )

    # Create the trace for the lower bar
    trace_under_12 = go.Scatter(
        x=under_12,
        y=condition,
        mode='markers',
        name='Under 7',
        marker=dict(
            color='#ff7f0e',
            size=18,
            line=dict(width=1)
        )
    )

    # Create the lines that connect the dots for each condition
    lines = []
    for i in range(len(condition)):
        line = go.Scatter(
            x=[under_12[i], over_12[i]],
            y=[condition[i], condition[i]],
            mode='lines',
            line=dict(color='#bcbddc', width=1),
            showlegend=False
        )
        lines.append(line)

    # Create the layout for the chart
    layout = go.Layout(
        title = dict(text = 'Mental Health Conditions for Over 7 Hours of Listening and Under 7 Hours of Listening', font=dict(size=25)),
        xaxis=dict(
            title = dict(text = 'Value', font=dict(size=18)),
            range=[0, max(max(over_12), max(under_12)) + 1]
        ),
        yaxis=dict(
            title = dict(text = 'Condition', font=dict(size=18)),
            tickfont=dict(size=18)
        ),
        height=500,
        margin=dict(l=100, r=100, t=100, b=100),
        showlegend=True
    )

    # Put all the traces together and create the figure
    data_ = [trace_over_12, trace_under_12] + lines
    fig = go.Figure(data=data_, layout=layout)
    fig.update_layout(template = 'plotly_dark')
    # Show the figure
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
   - The average of Anxiety, Depression, Insomnia and OCD are higher in records that contain more than 7
    hours a day listening to music, with emphasis on Insomnia, which has the greatest variation of all conditions reported when compared to the original dataset. 
    This may indicate that people who have insomnia stay awake much longer and also spend more time listening to music.
    """)

    fig = px.histogram(x = data['Age'], y = data['Hours per day'])
    fig.update_layout(template = 'plotly_dark', title = dict(text = 'Age and Sum of Hours Per Day', font=dict(size=25)),
                    xaxis_title = 'Age', yaxis_title = 'Sum of Hours', 
                    font=dict(
                    family="Courier New, monospace",
                    size=15,
                    color="White"
        ))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
     - Younger people often have more free time and fewer responsibilities compared to older individuals. 
    This means that they may have more time to engage in leisure activities such as listening to music. 
    - Music plays an important role in youth culture and identity formation. 
    During the teenage years and early adulthood, individuals may be more likely to explore different music genres and artists, 
    and use music as a way to express themselves and connect with their peers. 
    - Research suggests that the brain's reward centers are more active in response to music during adolescence, 
    which may contribute to a greater desire to listen to music. This increased sensitivity to music may be related to the changes that occur in the brain during this period of development, 
    particularly in regions involved in emotion processing and social cognition.
    """)

def bpm():
    cleaned_data()
    st.title("Beats per minute(BPM)")
    fig = px.scatter(data['BPM'], color=data['Fav genre'])
    fig.update_traces(marker=dict(size=17,
                                line=dict(width=2,
                                            color='DarkSlateGrey')),
                    selector=dict(mode='markers'))

    fig.update_layout(template = 'plotly_dark', title = dict(text = 'BPM and Fav Genre', font=dict(size=25)),
                    xaxis_title = 'Index', yaxis_title = 'BPM Value', 
                    height = 500,
                    width = 1200,
                    font=dict(
                    family="Courier New, monospace",
                    size=15,
                    color="White"
        ))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    Some data that point to this are, which points out that the interviewee's most listened to genre is Jazz, 
    but with a BPM of 194, far above the average of 120 - 125 for the genre.
    Since we analysed that participants didn't have much information about the BPM and how to calculate it, so we have
    decided to remove it from our Mental Health Analyzation.
    """)

def streaming_services():
    cleaned_data()
    st.title("Streaming Services")
    fig = px.pie(data, values=data['Primary streaming service'].value_counts(), names=data['Primary streaming service'].unique())
    fig.update_layout(template = 'plotly_dark', title = dict(text = 'Primary streaming service', font=dict(size=25)), 
                    height = 600,
                    width = 1200,
                    font=dict(
                    # family="Courier New, monospace",
                    size=25,
                    color="White"
        ))
    st.plotly_chart(fig, use_container_width=True)

    streaming_services_popularity_adults = data[data['Age Group'] == 'Adults']
    fig = go.Figure(data=[go.Pie(labels=streaming_services_popularity_adults['Primary streaming service'].unique(),
            values = streaming_services_popularity_adults['Primary streaming service'].value_counts(), hole=.3)])
    fig.update_layout(template = 'plotly_dark', title = dict(text = 'Popular Streaming Service for Adults', font=dict(size=25)),
                    height = 600,
                    width = 1200,
                    font=dict(
                    family="Courier New, monospace",
                    size=15,
                    color="White"
        ))
    st.plotly_chart(fig, use_container_width=True)
    st.write("""
    #### Pandora is more common among adults compared to Spotify. 

    * Brand recognition: Pandora has been around since 2000, and many adults may be more familiar with the brand compared to Spotify, which was launched in 2008.

    * User interface: Pandora's interface is simpler and more user-friendly compared to Spotify, which has a wide range of features and customization options that may be overwhelming for some users, especially older adults.

    * Music selection: Pandora offers a more curated listening experience, where users can create custom radio stations based on their favorite artists, songs, or genres. This may be appealing to adults who prefer a more passive listening experience and don't want to spend time creating playlists or searching for specific songs.

    * Price: Pandora offers a free version with ads and a premium version without ads for a lower price compared to Spotify, which may be more appealing to budget-conscious adults.

    * Demographic targeting: Pandora has historically targeted an older audience, while Spotify has targeted a younger, more tech-savvy audience. This may have influenced the preferences and habits of each platform's user base.
    """)

    streaming_services_popularity_teenagers = data[data['Age Group'] == 'Teenagers']
    fig = go.Figure(data=[go.Pie(labels=streaming_services_popularity_teenagers['Primary streaming service'].unique(),
            values = streaming_services_popularity_teenagers['Primary streaming service'].value_counts(), hole=.3)])
    fig.update_layout(template = 'plotly_dark', title = dict(text = 'Popular Streaming Service for Teenagers', font=dict(size=25)),
                    height = 600,
                    width = 1200,
                    font=dict(
                    family="Courier New, monospace",
                    size=15,
                    color="White"
        ))
    st.plotly_chart(fig, use_container_width=True)
    st.write("#### Spotify is a popular streaming service among teenagers for several reasons:")
    st.write("* Wide music selection: Spotify has a vast library of music with millions of songs from various genres, making it a go-to platform for music lovers.")
    st.write("* Personalization: Spotify provides personalized music recommendations based on the listener's music taste, search history, and previously played songs, which helps users discover new music that they might like.")
    st.write("* Social Sharing: Spotify allows users to create and share playlists with friends and follow other users, making it a social platform for music sharing.")
    st.write("* Accessibility: Spotify is available on multiple devices such as smartphones, tablets, and laptops, which makes it easy for teenagers to access their favorite music from anywhere.")


    # streaming_services_popularity_child = data[data['Age Group'] == 'Child']
    # fig = go.Figure(data=[go.Pie(labels=streaming_services_popularity_child['Primary streaming service'].unique(),
    #         values = streaming_services_popularity_child['Primary streaming service'].value_counts(), hole=.3)])
    # fig.update_layout(template = 'plotly_dark', title = dict(text = 'Popular Streaming Service for Children', font=dict(size=25)),
    #                 height = 600,
    #                 width = 1200,
    #                 font=dict(
    #                 family="Courier New, monospace",
    #                 size=15,
    #                 color="White"
    #     ))
    # st.plotly_chart(fig, use_container_width=True)


def univariate_analysis():
    st.title('Univariate Analysis')
    cleaned_data()

    # Count the frequency of each favorite genre
    genre_counts = data["Fav genre"].value_counts()

    # Count the frequency of knowing a foreign language
    lang_counts = data["Foreign languages"].value_counts()

    # Create the bar chart for favorite genre
    genre_fig = go.Figure(
        data=go.Bar(
            x=genre_counts.index,
            y=genre_counts.values,
            marker=dict(color="#00CC96")
        )
    )
    genre_fig.update_layout(
        title="Favorite Music Genre",
        xaxis_title="Genre",
        yaxis_title="Count",
        yaxis_type='log',font=dict(size=25)
    )

    # Create the bar chart for foreign languages
    lang_fig = go.Figure(
        data=go.Bar(
            x=lang_counts.index,
            y=lang_counts.values,
            marker=dict(color="#AB63FA"),
        )
    )
    lang_fig.update_layout(
        title="Listening to a Foreign Language",
        xaxis_title="Foreign Language Musics",
        yaxis_title="Count",font=dict(size=25)
    )

    # Show the figures
    st.plotly_chart(lang_fig, use_container_width=True)

    hours_data = data['Hours per day']
    st.plotly_chart(genre_fig, use_container_width=True)

    st.markdown("""
    #### Populrity of Songs
    
    * Rock music can take on many different forms, from the heavy metal sounds of bands like Metallica to acoustic-driven sounds. This versatility allows rock music to appeal to a wide range of audiences.

    * Many of the lyrics in rock songs deal with universal themes like love, loss, and rebellion. These themes can resonate with people of all ages and backgrounds, making rock music a relatable form of expression.

    * Over the years, there has been a shift away from traditional religious beliefs and practices in many parts of the world. 

    * Lack of mainstream exposure: Unlike other genres of music, gospel music, folk music, latin music has not always received as much exposure in mainstream media outlets. 

    * Gospel music may not be as widely available as other genres of music. This could be due to a lack of distribution or promotion, which may make it more difficult for people to discover or access.



    """)

    # create a list of box traces for different streaming services
    box_traces = []
    for service in data['Primary streaming service'].unique():
        box_trace = go.Box(
            y=data[data['Primary streaming service']==service]['Hours per day'],
            name=service
        )
        box_traces.append(box_trace)

    # define layout with dark theme colors
    layout = go.Layout(
        title='Hours per day spent on streaming by service',
        xaxis=dict(title='Streaming Service'),
        yaxis=dict(title='Hours per day'),
        plot_bgcolor='rgb(17,17,17)',
        paper_bgcolor='rgb(17,17,17)',
        font=dict(color='white', )
    )

    # create figure with box traces and layout
    fig = go.Figure(data=box_traces, layout=layout)
    st.plotly_chart(fig, use_container_width=True)

    # Create violin plot
    fig = px.violin(data, y=['Anxiety', 'Depression', 'Insomnia', 'OCD'],
    box=True, points='all', color_discrete_sequence=['#636EFA', '#EF553B', '#00CC96', '#AB63FA'])

    # Add title and axis labels
    fig.update_layout(title='Distribution of Mental Health Conditions',
                   xaxis_title='Mental Health Conditions',
                   yaxis_title='Score', font=dict(size=25))

    st.plotly_chart(fig, use_container_width=True)

def bivariate_analysis():
    cleaned_data()
    st.title("Bivariate Analysis")

    data_while_work = data[data['While working']=='Yes']
    data_without_work = data[data['While working']=='No']

    data_while_work = data_while_work['Age Group'].value_counts()
    data_without_work = data_without_work['Age Group'].value_counts()

    fig = go.Figure(data=[
        go.Bar(x = data_while_work.index, y = data_while_work.values, name='While working'),
        go.Bar(x = data_without_work.index, y = data_without_work.values, name='While not working')
    ])

    fig.update_layout(template = 'plotly_dark', title = dict(text = 'Age Group vs While Working', font=dict(size=25)),
                xaxis_title = 'Age Group', yaxis_title = 'Value Counts', 
                height = 600,
                width = 1200,
                )
    st.plotly_chart(fig, use_container_width=True)

    fig = go.Figure(data=[
    go.Bar(x = data['Gospel'].value_counts().index, y = data['Gospel'].value_counts().values, name='Gospel'),
    go.Bar(x = data['K pop'].value_counts().index, y = data['K pop'].value_counts().values, name='K pop'),
    go.Bar(x = data['EDM'].value_counts().index, y = data['EDM'].value_counts().values, name='EDM'),
    go.Bar(x = data['Pop'].value_counts().index, y = data['Pop'].value_counts().values, name='Pop'),
    go.Bar(x = data['Games music'].value_counts().index, y = data['Games music'].value_counts().values, name='Games music'),
    go.Bar(x = data['Country'].value_counts().index, y = data['Country'].value_counts().values, name='Country'),
    go.Bar(x = data['R&B'].value_counts().index, y = data['R&B'].value_counts().values, name='R&B'),
    ])

    fig.update_layout(template = 'plotly_dark', title = dict(text = 'Frequency vs Value Counts', font=dict(size=25)),
            xaxis_title = 'Frequency', yaxis_title = 'Value Counts', 
            height = 700,
            width = 1200,
            font=dict(
            family="Courier New, monospace",
            size=15,
            color="White"
    ))
    st.plotly_chart(fig, use_container_width=True)

    fig = go.Figure(data=[
    go.Bar(x = data['Rap'].value_counts().index, y = data['Rap'].value_counts().values, name='Rap'),
    go.Bar(x = data['Rock'].value_counts().index, y = data['Rock'].value_counts().values, name='Rock'),
    go.Bar(x = data['Metal'].value_counts().index, y = data['Metal'].value_counts().values, name='Metal'),
    go.Bar(x = data['Lofi'].value_counts().index, y = data['Lofi'].value_counts().values, name='Lofi'),
    go.Bar(x = data['Latin'].value_counts().index, y = data['Latin'].value_counts().values, name='Latin'),
    go.Bar(x = data['Jazz'].value_counts().index, y = data['Jazz'].value_counts().values, name='Jazz'),
    go.Bar(x = data['Folk'].value_counts().index, y = data['Folk'].value_counts().values, name='Folk'),
    go.Bar(x = data['Classical'].value_counts().index, y = data['Classical'].value_counts().values, name='Classical'),
    ])

    fig.update_layout(template = 'plotly_dark', title = dict(text = 'Frequency vs Value Counts', font=dict(size=25)),
            xaxis_title = 'Frequency', yaxis_title = 'Value Counts', 
            height = 700,
            width = 1200,
            font=dict(
            family="Courier New, monospace",
            size=15,
            color="White"
    ))
    st.plotly_chart(fig, use_container_width=True)

def mental_health():
    cleaned_data()
    st.title("Mental Health")
    data_while_work = data[data['While working']=='Yes']
    data_without_work = data[data['While working']=='No']

    data_while_work = data_while_work['Age Group'].value_counts()
    data_without_work = data_without_work['Age Group'].value_counts()

    # fig = go.Figure(data=[
    #     go.Bar(x = data_while_work.index, y = data_while_work.values, name='While working'),
    #     go.Bar(x = data_without_work.index, y = data_without_work.values, name='While not working')
    # ])

    # fig.update_layout(template = 'plotly_dark', title = dict(text = 'Age Group vs While Working', font=dict(size=25)),
    #             xaxis_title = 'Age Group', yaxis_title = 'Value Counts', 
    #             height = 600,
    #             width = 1200,
    #             font=dict(
    #             family="Courier New, monospace",
    #             size=15,
    #             color="White"
    # ))
    # st.plotly_chart(fig, use_container_width=True)

    data_while_work = data[data['While working']=='Yes']
    data_without_work = data[data['While working']=='No']

    conditions = ['Anxiety', 'Depression', 'Insomnia', 'OCD']
    while_work = [data_while_work['Anxiety'].mean(), data_while_work['Depression'].mean(), data_while_work['Insomnia'].mean(), data_while_work['OCD'].mean()]
    without_work = [data_without_work['Anxiety'].mean(), data_without_work['Depression'].mean(), data_without_work['Insomnia'].mean(), data_without_work['OCD'].mean()]

    
    over_12 = [5.3125, 5.8125,  5.1875, 3.5]
    under_12 = [5.9, 4.9, 3.8, 2.6]
    condition = ['Anxiety', 'Depression', 'Insomnia', 'OCD']
    # Create the trace for the upper bar
    trace_while_work = go.Scatter(
        x=while_work,
        y=conditions,
        mode='markers',
        name='While Working',
        marker=dict(
            color='#1f77b4',
            size=12,
            line=dict(width=1)
        )
    )

    # Create the trace for the lower bar
    trace_without_work = go.Scatter(
        x=without_work,
        y=conditions,
        mode='markers',
        name='While Not Working',
        marker=dict(
            color='#ff7f0e',
            size=12,
            line=dict(width=1)
        )
    )

    # Create the lines that connect the dots for each condition
    lines = []
    for i in range(len(condition)):
        line = go.Scatter(
            x=[while_work[i], without_work[i]],
            y=[conditions[i], conditions[i]],
            mode='lines',
            line=dict(color='#bcbddc', width=1),
            showlegend=False
        )
        lines.append(line)

    # Create the layout for the chart
    layout = go.Layout(
        title='Mental Health Conditions for Listening Music While Work and While not Working',
        xaxis=dict(
            title='Value',
            range=[0, max(max(over_12), max(under_12)) + 1]
        ),
        yaxis=dict(
            title='Condition',
            tickfont=dict(size=14)
        ),
        height=500,
        margin=dict(l=100, r=100, t=100, b=100),
        showlegend=True
    )

    # Put all the traces together and create the figure
    data_ = [trace_while_work, trace_without_work] + lines
    fig = go.Figure(data=data_, layout=layout)
    fig.update_layout(template = 'plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(data, x="Hours per day", y="Fav genre", color="Fav genre", 
                   marginal="box", nbins=10, barmode="group")

    fig.update_layout(template = 'plotly_dark', title = dict(text = 'Fav Genre vs Sum of Hours Per Day', font=dict(size=25)),
                xaxis_title = 'Sum of Hours Per Day', yaxis_title = 'Fav Genre', 
                height = 700,
                width = 1200,
                font=dict(
                family="Courier New, monospace",
                size=15,
                color="White"
    ))
    st.plotly_chart(fig, use_container_width=True)

    fav_genre_among_teenagers = data[data['Age Group'] == 'Teenagers']
    fav_genre_among_adults = data[data['Age Group'] == 'Adults']
    fav_genre_among_senior_citizens = data[data['Age Group'] == 'Senior Citizens']
    # st.plotly_chart(fig, use_container_width=True)

    fig = go.Figure(data=[
        go.Bar(x = fav_genre_among_teenagers['Fav genre'].value_counts().index, y = fav_genre_among_teenagers['Fav genre'].value_counts().values, name='Teenagers'),
        go.Bar(x = fav_genre_among_adults['Fav genre'].value_counts().index, y = fav_genre_among_adults['Fav genre'].value_counts().values, name = 'Adults'),
        go.Bar(x = fav_genre_among_senior_citizens['Fav genre'].value_counts().index, y = fav_genre_among_senior_citizens['Fav genre'].value_counts().values, name = 'Senior Citizens'),
    ])

    fig.update_layout(template = 'plotly_dark', title = dict(text = 'Age Group vs Fav Genre Count', font=dict(size=25)), yaxis_type='log',
                xaxis_title = 'Fav Genre', yaxis_title = 'Count', 
                height = 700,
                width = 1200,
                font=dict(
                family="Courier New, monospace",
                size=15,
                color="White"
    ))
    st.plotly_chart(fig, use_container_width=True)

    music_effect_improve = data[data['Music effects'] == 'Improve']
    music_effect_no_effect  = data[data['Music effects'] == 'No effect']
    music_effect_worsen = data[data['Music effects'] == 'Worsen']

    fig = go.Figure(data=[
        go.Bar(x = music_effect_improve['Fav genre'].value_counts().index, y = music_effect_improve['Fav genre'].value_counts().values, name='Improve'),
        go.Bar(x = music_effect_no_effect['Fav genre'].value_counts().index, y = music_effect_no_effect['Fav genre'].value_counts().values, name = 'No effect'),
        go.Bar(x = music_effect_worsen['Fav genre'].value_counts().index, y = music_effect_worsen['Fav genre'].value_counts().values, name = 'Worsen'),
    ])

    fig.update_layout(template = 'plotly_dark', title = dict(text = 'Genre vs Mental Health Results', font=dict(size=25)),
                xaxis_title = 'Fav Genre', yaxis_title = 'Count', 
                height = 700,
                width = 1200,
                font=dict(
                family="Courier New, monospace",
                size=15,
                color="White"
        ))
    st.plotly_chart(fig, use_container_width=True)

    grouped_df = data.groupby('Age Group').agg({'Anxiety': 'mean', 'Depression': 'mean', 'Insomnia': 'mean', 'OCD': 'mean'}).reset_index()
    fig = go.Figure(data=[
    go.Line(x = ['Anxiety', 'Depression', 'Insomnia', 'OCD'], y = grouped_df.iloc[0, 1:], name = 'Child'),
    go.Line(x = ['Anxiety', 'Depression', 'Insomnia', 'OCD'], y = grouped_df.iloc[1, 1:], name = 'Teenagers'),
    go.Line(x = ['Anxiety', 'Depression', 'Insomnia', 'OCD'], y = grouped_df.iloc[2, 1:], name = 'Adults'),
])

    fig.update_layout(template = 'plotly_dark', title = dict(text = 'Age Group vs Mental Health Condition', font=dict(size=25)),
                    xaxis_title = 'Mental Health Condition Matrix', yaxis_title = 'Value', 
                    height = 700,
                    width = 1200,
                    font=dict(
                    family="Courier New, monospace",
                    size=15,
                    color="White"
        ))
    st.plotly_chart(fig, use_container_width=True)

    st.write("""
    
    ### Insights:

    - Anxiety level is exceptionally high in children in comparison to other mental health issue. This could be accounted due to many reasons.
        * Environmental factors: Factors such as family dynamics, stressful life events, and school-related stress can also contribute to anxiety levels in children.
        * Pressure to perform: Many children feel pressure to perform academically or socially, which can lead to anxiety and stress.
        * Lack of coping skills: Children may not have developed the coping skills needed to manage stress and anxiety, which can result in heightened levels of anxiety.
    
    - Higher values of insomnia can be seen in adults.

    - One common trend observed is that with increasing people are suffering more and more from these issues.
    """)

def conclusion():
    st.title("Conclusion")

    st.markdown("""
    * Mental health disorders can affect people of all ages and are not limited to a particular age group.
    
    * Although teenagers may have a higher prevalence of depression, anxiety, and insomnia, these disorders can occur in older age groups as well.
    
    * Teenagers often listen to music while working, which may be associated with higher rates of mental health disorders in this age group.
    
    * There is no clear evidence of a direct correlation between listening to music while working and mental health disorders. However, excessive listening of more than 7 hours a day may be associated with noticeable insomnia.

    * Certain types of music, such as rock, pop, and metal, have been shown to help improve mental health. However, other types of music, such as video game music, may worsen the situation rather than contribute to improvement.
    
    """
    )

    st.title("Future Work")
    st.markdown("""
    * Take a survey with more people to get better understanding of mental health and how it is dependent various features.

    * To discover more features which affect the mental health of a person and their dependency.

    * To Train the model on variety of features, so rather than just giving the music effect it gives genre of music for better health results.
    """)
    



def ml_model():
    st.title('Predicting Improvement of Mental Health by Music')
    from pathlib import Path
    # --- PATH SETTINGS ---
    current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
    css_file = current_dir / "main.css"
    with open(css_file) as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    age, streaming_service, hours, working = st.columns(4)
    age = age.text_input("Age", "Type Here  ")
    streaming_service = streaming_service.selectbox("Streaming service", ('Spotify', 'YouTube Music', 'Pandora', 'Apple Music', 'No Streaming Service', 'Others'))
    
    hours = hours.number_input('Hours of Listening')
    
    working = working.selectbox('While Working', ('Yes', 'No'))
        
    instrumentalist = st.selectbox('Do you listen to Instrumentalist?', ('Yes', 'No'))
    
    composer = st.selectbox('Do you listen to Composer?', ('Yes', 'No'))
    
    exploratory = st.selectbox("Do you listen to exploratory music?", ('Yes', 'No'))
    
    fav_genre = st.selectbox("Favourite Genre", ('Latin', 'Rock', 'Video game music', 'Jazz', 'R&B', 'K pop', 'Country', 'EDM', 'Hip hop', 'Pop', 'Rap', 'Classical', 'Metal', 'Folk', 'Lofi', 'Gospel'))

    foreign_language = st.selectbox("Do you listen Foreign Languages?", ('Yes', 'No'))
    
    BPM = st.number_input('Beats per minute(BPM)')

    classical = st.selectbox("Frequency of Classical Music", ('Rarely', 'Sometimes', 'Very frequently', 'Never'))
    country = st.selectbox("Frequency of Country Music", ('Rarely', 'Sometimes', 'Very frequently', 'Never'))
    EDM = st.selectbox("Frequency of EDM", ('Rarely', 'Sometimes', 'Very frequently', 'Never'))
    folk = st.selectbox("Frequency of Folk Music", ('Rarely', 'Sometimes', 'Very frequently', 'Never'))
    gospel = st.selectbox("Frequency of Gospel", ('Rarely', 'Sometimes', 'Very frequently', 'Never'))
    hiphop = st.selectbox("Frequency of Hip hop", ('Rarely', 'Sometimes', 'Very frequently', 'Never'))
    jazz = st.selectbox("Frequency of Jazz", ('Rarely', 'Sometimes', 'Very frequently', 'Never'))
    kpop = st.selectbox("Frequency of K pop", ('Rarely', 'Sometimes', 'Very frequently', 'Never'))
    latin = st.selectbox("Frequency of Latin Music", ('Rarely', 'Sometimes', 'Very frequently', 'Never'))
    lofi = st.selectbox("Frequency of Lofi", ('Rarely', 'Sometimes', 'Very frequently', 'Never'))
    metal = st.selectbox("Frequency of Metal", ('Rarely', 'Sometimes', 'Very frequently', 'Never'))
    pop = st.selectbox("Frequency of Pop", ('Rarely', 'Sometimes', 'Very frequently', 'Never'))
    rnb = st.selectbox("Frequency of R&B", ('Rarely', 'Sometimes', 'Very frequently', 'Never'))
    rap = st.selectbox("Frequency of Rap", ('Rarely', 'Sometimes', 'Very frequently', 'Never'))
    rock = st.selectbox("Frequency of Rock", ('Rarely', 'Sometimes', 'Very frequently', 'Never'))
    video_game = st.selectbox("Frequency of Video game Music", ('Rarely', 'Sometimes', 'Very frequently', 'Never'))

    anxiety = st.number_input("Anxiety level from a scale of one to ten", min_value=0, max_value=10)
    depression = st.number_input("Depression level from a scale of one to ten", min_value=0, max_value=10)
    insomnia = st.number_input("Insomnia level from a scale of one to ten", min_value=0, max_value=10)
    ocd = st.number_input("OCD level from a scale of one to ten", min_value=0, max_value=10)

    input = [age, streaming_service, hours, working, instrumentalist, composer, 
             fav_genre, exploratory, foreign_language, BPM, classical, country, 
             EDM, folk, gospel, hiphop, jazz, kpop, latin, lofi, metal, pop, rnb, 
             rap, rock, video_game, anxiety, depression, insomnia, ocd]

    input_ = {}



    import pandas as pd
    import numpy as np
    from sklearn.impute import KNNImputer,SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import LabelEncoder,OneHotEncoder
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    import category_encoders as ce
    from sklearn import set_config
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.ensemble import RandomForestClassifier
    df = pd.read_csv('mxmh_survey_results.csv')
    df.drop(['Timestamp', 'Permissions'], axis=1, inplace=True)
    impute = KNNImputer()
    simple_impute = SimpleImputer(missing_values='NAN', strategy='mean')
    df['Age'] = impute.fit_transform(df['Age'].values.reshape(-1,1))
    df['BPM'] = impute.fit_transform(df['BPM'].values.reshape(-1,1))
    df['Primary streaming service'] = df['Primary streaming service'].fillna(df['Primary streaming service'].mode()[0])
    df['While working'] = df['While working'].fillna(df['While working'].mode()[0])
    df['Instrumentalist'] = df['Instrumentalist'].fillna(df['Instrumentalist'].mode()[0])
    df['Composer'] = df['Composer'].fillna(df['Composer'].mode()[0])
    df['Foreign languages'] = df['Foreign languages'].fillna(df['Foreign languages'].mode()[0])
    df['Music effects'] = df['Music effects'].fillna(df['Music effects'].mode()[0])
    q1 = np.percentile(df['Age'],25)
    q3 = np.percentile(df['Age'],75)
    iqr = q3 - q1
    lb = q1 - (1.5 * iqr)
    up = q3 + (1.5 * iqr)

    feature = [feature for feature in df.columns if df[feature].dtypes != 'O' ]
    features = ['Age', 'Hours per day', 'BPM']


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
    categorical_feature = [feature for feature in df.columns if df[feature].dtypes == 'O' ]
    index = [df.columns.get_loc(c) for c in numerical_feature]
    df['Music effects'] = df['Music effects'].map({'Improve':0,'No effect':1,'Worsen':2})

    numerical_process = Pipeline(
        steps = [('scaler',StandardScaler())]
    )
    set_config(display="diagram")
    lr = LabelEncoder()

    cateogtical_process_1 = Pipeline(
        steps = [('lr', ce.TargetEncoder()),
            ('scaler',StandardScaler())]
        
    )

    X= df.iloc[:,0:-1]
    y = df.iloc[:,-1]
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=42)
    processsor_1 = ColumnTransformer(
        [('Categorical_encoding',cateogtical_process_1,X.select_dtypes(include="object").columns),
        ('numerical_encoding',numerical_process,X.select_dtypes(exclude="object").columns)])

    processsor_1.get_params()
    # st.write("askhff")
    # X.select_dtypes(include="object").columns

    rf = RandomForestClassifier()
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
    

    cols = list(df.columns)
    for i in range(len(cols)-1):
        input_[cols[i]] = input[i]
    
    valss = pd.DataFrame(input_, index=[0])
    result = ""
    if st.button("Mental Health"):
        result = pipe_4.predict(valss)
    success = ""
    if result == 0:
        st.success("Music is Improving the Mental Health")
    elif result == 1:
        st.success("Music has no effect on the Mental Health")
    elif result == 2:
        st.success("Music is worsening the Mental Health")


nav = st.sidebar.radio("Analysis", ["Home","About Dataset", "Data Cleaning", 
            "Hours Per Day", "BPM", "Streaming Services","Univariate Analysis", "Bivariate Analysis","Mental Health","Conclusion","ML Model"])
st.sidebar.image("developed.png", use_column_width=True)

# Show appropriate page based on selection
if nav == "Home":
    homepage()
elif nav == "About Dataset":
    dataset()
elif nav == "Data Cleaning":
    datacleaning()
elif nav == "Hours Per Day":
    hours_per_day()
elif nav == "BPM":
    bpm()
elif nav == "Streaming Services":
    streaming_services()
elif nav == "Univariate Analysis":
    univariate_analysis()
elif nav == "ML Model":
    ml_model()
elif nav == "Mental Health":
    mental_health()
elif nav == "Bivariate Analysis":
    bivariate_analysis()
elif nav == "Conclusion":
    conclusion()
else:
    pass
