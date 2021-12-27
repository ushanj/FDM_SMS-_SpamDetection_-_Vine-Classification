
gcount = 92
bcount = 2
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plot
from streamlit_lottie import st_lottie
import json
import matplotlib.pyplot as plt
import datetime
import streamlit as st
from PIL import Image
import pickle
import pandas as pd
import wordcloud
import matplotlib.pyplot as plt
import nltk 
import plotly_express as px
from nltk.corpus import stopwords
import requests
from requests import Request, Session
from sklearn.cluster import KMeans
import numpy as np
from pandas.plotting import andrews_curves
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
a_session = requests.Session()

def load_lottiefile(filepath: str):
        with open(filepath, "r") as f:
            return json.load(f)

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_hello = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_iqwxrsiq.json")
lottie_wine = load_lottieurl("https://assets7.lottiefiles.com/private_files/lf30_6BWvT3.json")

menu = ["Home","Classification","Clusterring"]
choices = st.sidebar.selectbox("Select Page",menu)

#########################################################################
#
#
#
#### Classification Model Page
#
#
#
#########################################################################

if choices =="Classification":

    pickle_in = open("models/spam_model.pkl","rb")
    classifier = pickle.load(pickle_in)

    pickle_in1 = open("models/tfidf_model.pk1","rb")
    tfidf_model = pickle.load(pickle_in1)

    image = Image.open('Images/logo.png')

    st.image(image, use_column_width = True)
    st.write("""

    # Spam Detection Model

    ## Brief introduction to this application,
    The classification technique (SMS Spam Dataset) used will help us to eliminate a 
    message received by predicting if it falls under the category of spam or ham. The model is ensured to have 
    high accuracy in order to prevent useful messages from being classified as 
    spam and ham which leads to an increase in commercial benefit.  
    ***
    """)

    #Loading the dataset

    df = pd.read_csv("spam.csv", encoding='latin-1')
    df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
    df.rename(columns = {'v1':'class_label','v2':'message'},inplace=True)

    # Plotting the historam based on Ham and Spam

    fig = px.histogram(df, x="class_label", color="class_label", color_discrete_sequence=["#871fff","#ffa78c"],height=500)
    st.write("The following chart shows the number of Normal messgaes and spam messgaes for a week")
    st.write(fig)
    st.write("As you can see from the above graph that a user receives spam message" +
     "rarely and from the below chart you can see the percentage of Spam message (13.4%) "  +  
     "because of that reason spams it hard to detect.")
    
    # Plotting pie chart and displat percentage of Ham and Spam

    fig = px.pie(df.class_label.value_counts(),labels='index', values='class_label', color="class_label", 
    color_discrete_sequence=["#871fff","#ffa78c"] )
    st.write(fig)
    st.write("""
    ***
    #### Enter your message below:
    """)
    get_user = st.text_area("")

    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Plotting the word cloud
    def show_wordcloud(msg):
        msg = [msg]
        dataset = {'msg': msg}
        data = pd.DataFrame(dataset)
        text = ' '.join(data['msg'].astype(str).tolist())
        stopwords = set(wordcloud.STOPWORDS)
        fig_wordcloud = wordcloud.WordCloud(stopwords=stopwords,background_color = "#ffa78c",colormap="hot",max_words=500,
        max_font_size=350, random_state=42)
        fig_wordcloud.generate(text)
        plt.figure(figsize=(0,0))
        fig, axes = plt.subplots(1,2, gridspec_kw={'width_ratios': [1, 0]})
        axes[0].imshow(fig_wordcloud)
        
        plt.axis('off')
        st.pyplot()

    # Displaying Word Cloud
    if st.button("Show Unique"):
        st.write(show_wordcloud(get_user))
       
    result = ""

#Prediction function
    def predicts(msg):

        msg = [msg]
        dataset = {'msg': msg}
        data = pd.DataFrame(dataset)
        nltk.download('stopwords')
        nltk.download('punkt')
        #####  Normalizing the dataset #####
        # Replacing Emails
        # Replacing Web Address
        # Replacing Mobile Symboles
        # Replacing Phone Numbers
        # Replacing Numbers
        # Replacing White Spaces
        #####

        data['msg'] = data['msg'].str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$','emailaddress')

        # Replace urls with 'webaddress'
        data['msg'] = data['msg'].str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$', 'webaddress')

        # Replace money symbol with 'money-symbol'
        data['msg'] = data['msg'].str.replace(r'£|\$', 'money-symbol')

        # Replace 10 digit phone number with 'phone-number'
        data['msg'] = data['msg'].str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$', 'phone-number')

        # Replace normal number with 'number'
        data['msg'] = data['msg'].str.replace(r'\d+(\.\d+)?', 'number')

        # remove punctuation
        data['msg'] = data['msg'].str.replace(r'[^\w\d\s]', ' ')

        # remove whitespace between terms with single space
        data['msg'] = data['msg'].str.replace(r'\s+', ' ')

        # remove leading and trailing whitespace
        data['msg'] = data['msg'].str.replace(r'^\s+|\s*?$', ' ')

        # change words to lower case
        data['msg'] = data['msg'].str.lower()

        stop_words = set(stopwords.words('english'))
        data['msg'] = data['msg'].apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))
        ss = nltk.SnowballStemmer("english")
        data["msg"] = data["msg"].apply(lambda x: ' '.join(ss.stem(term) for term in x.split()))
        tfids = tfidf_model.transform(data["msg"])
        tfidf_data = pd.DataFrame(tfids.toarray())
        prediction = classifier.predict(tfidf_data)

        return prediction
    
    st_lottie(lottie_hello,key="hello",height = 100,width = 100)
    if st.button("Predict"):
        me = predicts(get_user)
        if(me==0):
            result = "normal message"
        elif(me==1):
            result = "spam message"
        new_df = pd.DataFrame(zip(get_user,result), columns = ["Actual Msg","Output"])
    st.success('This is a {}'.format(result))

#########################################################################
#
#
#
#### Clusterring Model Page
#
#
#
#########################################################################

if choices =="Clusterring":

    image = Image.open('Images/wine.jpg')

    st.image(image, use_column_width = True)
    st.write("""

    # Wine Clusterring Model
    ## Brief introduction to this application,

    The data set that we are analyzing is a result of a chemical analysis of wines grown in a particular region in Italy but derived 
    from three different cultivars. The analysis determined the quantities of 13 constituents found in each of the three types of wines.  

    ***
    """)

    df = pd.read_csv("Wine.csv")

    df=df.dropna(axis = 0)
    data = df.drop(["Flavanoids"], axis=1)
    from scipy import stats
    z = np.abs(stats.zscore(data))
    threshold = 1.8
    f1 = data[(z < 1.8).all(axis=1)]
    x = f1.drop('Customer_Segment', axis=1)
    y = f1['Customer_Segment']
    scaler = StandardScaler()

    st.write("""
    ## Sample Dataset """)
    st.write(x)
    X = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

    kmeans = KMeans(n_clusters=3,init= 'k-means++',random_state = 42)
    y_means = kmeans.fit_predict(X) 

    pca = PCA(n_components =2)
    reduced_x = pd.DataFrame(pca.fit_transform(X), columns=["PC1" , "PC2"])
    reduced_x['cluster'] = y_means

    reduced_centers = pca.transform(kmeans.cluster_centers_)

    
    # Plotting the clusters
    
    st.write("""
    ## Visualized Clusters.
    """)
    st.write("The following plot you can see from 3 Regions Wine is extracted and the regions are")
    st.write("""
    - Cluster 1 - Region 1
    - Cluster 2 - Region 3
    - Cluster 3 - Region 2
    """)
    fig = plt.figure(figsize=(14,10))
    plt.scatter(reduced_x[reduced_x['cluster'] == 0].loc[:, 'PC1'] , reduced_x[reduced_x['cluster'] == 0].loc[:, 'PC2'] , c='green', label='Cluster 1')
    plt.scatter(reduced_x[reduced_x['cluster'] == 1].loc[:, 'PC1'] , reduced_x[reduced_x['cluster'] == 1].loc[:, 'PC2'] , c='blue', label='Cluster 2')
    plt.scatter(reduced_x[reduced_x['cluster'] == 2].loc[:, 'PC1'] , reduced_x[reduced_x['cluster'] == 2].loc[:, 'PC2'] , c='yellow', label='Cluster 3')
    plt.scatter(reduced_centers[:, 0] , reduced_centers[:, 1] , color='black' , s=100, label = "centroid")
    plt.legend()

    st.write(fig)
    st.title("Interactive K-Means Clustering")
    dataset = pd.DataFrame({'Original': y.values, 'Predicted': y_means}, columns=['Original', 'Predicted'])
    dataset.loc[dataset['Predicted'] == 0, 'Predicted'] = 3

    da = dataset.values
    gfcount = 0
    bfcount = 0
    for df in da:
        if df[0] == df[1]:
            gfcount = gcount + 1
        else:
            bfcount = bcount + 1

    # Pie chart, where the slices will be ordered and plotted counter-clockwise:

    labels = ["Correctly Predicted", "Wrongly Predicted"]
    data = [gcount, bcount]
    explode = (0, 0.5)  # only "explode" the 2nd slice (i.e. 'Wrongly Predicted')
    fig1 = plt.figure(figsize =(25, 10))
    fig1, ax1 = plt.subplots()
    ax1.pie(data, explode=explode, labels=labels, autopct='%1.2f%%',shadow=True, startangle=360)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.write(fig1)

    st.write("""
    ## Enter your inputs below for predictions
    ***
    """)
    col1, col2,col3,col4 = st.columns(4)
    
    Alcohol = col1.text_input("Enter Alcohol:")
    Malic_Acid = col2.text_input("Enter Malic_Acid:")
    Ash = col3.text_input("Enter Ash:")
    Ash_Alcanity = col4.text_input("Enter Ash_Alcanity:")
    Magnesium = col1.text_input("Enter Magnesium:")
    Total_Phenols = col2.text_input("Enter Total_Phenols:")
    Nonflavanoid_Phenols = col3.text_input("Enter Nonflavanoid_Phenols:")
    Proanthocyanins = col4.text_input("Enter Proanthocyanins:")
    Color_Intensity = col1.text_input("Enter Color_Intensity:")
    Hue = col2.text_input("Enter Hue:")
    OD280 = col3.text_input("Enter OD280:")
    Proline = col4.text_input("Enter Proline:")

    result = ""

    def predicts(Alcohol, Malic_Acid , Ash , Ash_Alcanity , Magnesium , Total_Phenols , Nonflavanoid_Phenols,Proanthocyanins,Color_Intensity,Hue,OD280,Proline):
        d = [{"Alcohol":Alcohol,"Malic_Acid": Malic_Acid, "Ash": Ash, "Ash_Alcanity": Ash_Alcanity, "Magnesium":Magnesium, "Total_Phenols":Total_Phenols,
        "Nonflavanoid_Phenols":Nonflavanoid_Phenols,"Proanthocyanins":Proanthocyanins,"Color_Intensity":Color_Intensity,"Hue":Hue,"OD280":OD280,"Proline":Proline }]
        
        data = pd.DataFrame(d)
        scaled_data = scaler.transform(data)
        pre = kmeans.predict(scaled_data)
        redx = pd.DataFrame(pca.transform(scaled_data), columns=["PC1" , "PC2"])
        redx['cluster'] = pre
        #if pre == 1:
            #clust = "Cluster 1"
        #elif pre == 2:
           # clust = "Cluster 2"
        #elif pre == 3:
           # clust = "Cluster 3"

        return pre,redx,data

    if st.button("Predict"):
        pre,redx,data= predicts(Alcohol,Malic_Acid,Ash,Ash_Alcanity,Magnesium,Total_Phenols,Nonflavanoid_Phenols,Proanthocyanins,Color_Intensity,Hue,OD280,Proline)
        
        if(pre==1):
            result = "First region"
            st.write("""
            ## Final prediction
            """)
            st.success('This wine belongs to {}'.format(result))
            st_lottie(lottie_wine,key="hello",height = 100,width = 100)
        elif(pre==2):
            st.write("""
            ## Final prediction
            """)
            result = "Second region"
            st.warning('This wine belongs to {}'.format(result))
            st_lottie(lottie_wine,key="hello",height = 100,width = 100)
        elif(pre==3):
            st.write("""
            ## Final prediction
            """)
            result = "Third region"
            st.error('This wine belongs to {}'.format(result))
            st_lottie(lottie_wine,key="hello",height = 100,width = 100)
            
        fig = plt.figure(figsize=(14,10))
        plt.scatter(reduced_x[reduced_x['cluster'] == 0].loc[:, 'PC1'] , reduced_x[reduced_x['cluster'] == 0].loc[:, 'PC2'] , c='green', label='Cluster 1')
        plt.scatter(reduced_x[reduced_x['cluster'] == 1].loc[:, 'PC1'] , reduced_x[reduced_x['cluster'] == 1].loc[:, 'PC2'] , c='blue', label='Cluster 2')
        plt.scatter(reduced_x[reduced_x['cluster'] == 2].loc[:, 'PC1'] , reduced_x[reduced_x['cluster'] == 2].loc[:, 'PC2'] , c='yellow', label='Cluster 3')
        plt.scatter(reduced_centers[:, 0] , reduced_centers[:, 1] , color='black' , s=100, label = "centroid")
       
        plt.scatter(redx[redx['cluster'] == pre].loc[:, 'PC1'] , redx[redx['cluster'] == pre].loc[:, 'PC2'] ,s=400, c='red', label = "Predicted{}".format(result))
        plt.legend()
        st.write(fig)
        
        st.write("""
        ## Final prediction with user values
        """)
        data['Predicted_Label'] = pre
        st.write(data)

#########################################################################
#
#
#
#### Home Page
#
#
#
#########################################################################


if choices == "Home":
    today = st.date_input("Today is",datetime.datetime.now())
    def load_lottiefile(filepath: str):
        with open(filepath, "r") as f:
            return json.load(f)

    def load_lottieurl(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    lottie_coding = load_lottiefile("ani/lap.json")
    
    st.markdown("""
    # Welcome to Data Science :sunglasses:
    ## Brief introduction to this application,

    - Our web application contains Spam classification model for detecting SMS spam messages.
    - This web application is also contains a clustering techniques regarding Wine to detect which region it belongs to.

    For clustering we have used K-means algorithm and for classification we have used LightGBM model which is a related to decision tree algorithm.

    # Our Business Goal,
    The goals of this projects are to increase sales for a potential wine business and to mitigate the 
    risks of ham and spam messages received by any business. we're hoping to address these 
    issues and derive a solution that would give long term results. 
    we hope to achieve this over a span of one month by the involvement of our whole team.
    """)

    st_lottie(
        lottie_coding,
        speed = 1,
        reverse=False,
        loop = True,
        quality = "low",
        renderer = "svg",
        height = None,
        width = None,
        key = None
    )
    
