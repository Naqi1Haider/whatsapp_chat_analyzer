import pandas as pd
import streamlit as st
import Preprocessing
import Helper
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

st.markdown(
    """
    <style>
    .big-font {
        font-size:100px !important;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True
)

# Using the custom class to style the title
st.markdown('<p class="big-font">Your Chat Insights</p>', unsafe_allow_html=True)


# Load the WhatsApp logo image
whatsapp_logo = Image.open('C:\\Users\\lenovo\\PycharmProjects\\Whatsapp-Chat-Analyzer\\whatsapp_logo.png')

# Display the logo in the sidebar
st.sidebar.image(whatsapp_logo, width=100)  # Adjust width as needed


st.sidebar.title("WhatsApp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = Preprocessing.preprocess(data)

    # Perform sentiment analysis using Hinglish stop words file
    stop_words_file = 'stop_hinglish.txt'
    df = Helper.analyze_sentiment(df, stop_words_file)

    # Fetch unique users
    user_list = df['user'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show analysis with respect to", user_list)

    if st.sidebar.button("Show Analysis"):

        #Stats Area
        if selected_user == "Overall":
            num_messages, words, num_media_messages, num_links = Helper.fetch_stats(selected_user, df)
        else:
            num_messages, words, num_media_messages, num_links = Helper.fetch_stats(selected_user, df[df['user'] == selected_user])
            # Custom CSS for increasing text size and changing color

        st.title("Top Statistics")
        cols = st.columns(4)

        with cols[0]:
            st.header("Total Messages")
            st.title(num_messages)
        with cols[1]:
            st.header("Total Words")
            st.title(words)
        with cols[2]:
            st.header("Media Shared")
            st.title(num_media_messages)
        with cols[3]:
            st.header("Links Shared")
            st.title(num_links)

        # Monthly timeline
        st.title("Monthly TimeLine")
        timeline = Helper.monthly_timeline(selected_user, df)
        fig,ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # daily timeline
        st.title("Daily Timeline")
        daily_timeline = Helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # activity map
        st.title('Activity Map')
        col1, col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = Helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = Helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = Helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)


        # finding the busiest user in the group (Group level)
        if selected_user == 'Overall':
            st.title('Most Busy Users')
            x, new_df = Helper.most_busy_users(df)
            fig, ax = plt.subplots()

            cols = st.columns(2)

            with cols[0]:
                ax.bar(x.index, x.values, color='red')
                plt.xticks(rotation = 'vertical')
                st.pyplot(fig)
            with cols[1]:
                st.dataframe(new_df)

        # WordCloud
        st.title("WordCloud")
        df_wc = Helper.create_wordcloud(selected_user,df)
        fig,ax = plt.subplots()
        plt.imshow(df_wc)
        st.pyplot(fig)

        #Most common words
        most_common_df = Helper.most_common_words(selected_user, df)
        fig,ax = plt.subplots()
        ax.barh(most_common_df[0], most_common_df[1])
        plt.xticks(rotation = 'vertical')
        st.title('Most Common Words')
        st.pyplot(fig)

        #emoji analysis
        emoji_df = Helper.emoji_helper(selected_user, df)
        st.title("Emoji Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(emoji_df)
        with col2:
            fig, ax = plt.subplots()
            ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct="%0.2f")
            st.pyplot(fig)



        # Sentiment Analysis
        st.title("Sentiment Analysis")
        sentiment_count = df['Sentiment'].value_counts()
        fig, ax = plt.subplots()
        ax.bar(sentiment_count.index, sentiment_count.values, color=['green', 'red', 'gray'])
        st.pyplot(fig)

        st.dataframe(df[['message', 'Sentiment']])