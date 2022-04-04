from model import Recommender
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

recommender = Recommender()
posters = recommender.posters
#Methods for button controlling
def counter_reset():
    st.session_state.count = 0
if 'count' not in st.session_state:
	st.session_state.count = 0
def increment_counter():
	st.session_state.count += 1

st.title("Recommender System Dashboard: Content-based using plot")
selected_userId =  st.number_input('Choose user', min_value=1, max_value=610, value=1, step=1,on_change=counter_reset)
recommendations = recommender.get_scaled_scores(selected_userId).head()
st.write('Recommendations for user: ',selected_userId)


if(st.button('Get next recommendation',on_click=increment_counter)):
    if (st.session_state.count >= 6):
        st.write('Choose next user')
    else:
        st.write('Here is your ',st.session_state.count,'th recommendation: ', recommendations.index[st.session_state.count-1])
        try:
            st.image(posters[recommendations.index[st.session_state.count-1]],width=100)
        except Exception:
            st.image("https://i.ibb.co/TqJsxKM/IMG-20190204-WA0001.jpg",width=100)
        st.write('Because you liked the following movies: ')
        recommendations.sort_values(by='score', ascending=False, inplace=True)
        images = []
        titles = []
        stars = []
        scores = recommendations.iloc[st.session_state.count-1].root_scores
        for title in recommendations.iloc[st.session_state.count-1].roots:
            try:
                images.append(posters[title])
            except Exception:
                images.append("https://i.ibb.co/TqJsxKM/IMG-20190204-WA0001.jpg")
            titles.append(title)

            #get star ratings
            ratings = recommender.get_user_ratings(selected_userId)
            n = int(ratings[title])
            star = ''
            for i in range(0,n):
                star += ' :star: '
            stars.append(star)
        columns = st.columns(len(titles))
        for i in range(0,len(titles)):

            columns[i].caption(titles[i][:15])
            columns[i].image(images[i],use_column_width=False,width=100)
            columns[i].markdown(stars[i])
        #plot the bar chart
        scores = np.array(scores, dtype='float32')
        fig = plt.figure(figsize=(18, 5))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.bar(titles, scores,color='#00876c')

        st.pyplot(fig)
