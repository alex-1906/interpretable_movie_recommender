import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

class Recommender:

    def __init__(self):
        '''Loading the required Datasets from the Data Folder'''
        metadata_df = pd.read_csv('../Data/movies_metadata.csv', low_memory=False)
        ratings_df = pd.read_csv('../Data/ratings.csv')
        movies_df = pd.read_csv('../Data/movies.csv')
        posters_df = pd.read_csv('../Data/movie_poster.csv', delimiter=';')
        links_df = pd.read_csv('../Data/links.csv')

        '''Preprocessing'''

        #drop the duplicates in movies_df and merge the imdbIds into it
        links_df.drop(columns='tmdbId', inplace=True)
        movies_df.drop_duplicates(inplace=True, subset='title', ignore_index=True)
        movies_df = pd.merge(movies_df, links_df, on='movieId')

        #get plots from the metadata set and merge it into movies_df
        liste = []
        for i in metadata_df.imdb_id.values:
            i = str(i)
            i = i[2:]
            if (i.find('n') != -1 or i == ''):
                i = 0
            i = int(i)
            liste.append(i)
        mapping_imdbId_plot = pd.Series(index=liste, data=metadata_df.overview.values)
        plots = []
        counter = 0
        for i in movies_df.imdbId.values:
            try:
                # changing all types to str, as there were some pandas.series mixed in between
                plots.append(str(mapping_imdbId_plot[i]))
            except Exception as e:
                plots.append('')
                counter += 1
        #Number of mismatches: 217
        movies_df['plot'] = plots

        #get dataframe for ratings for each user
        movies_with_genres = movies_df.copy(deep=True)
        ratings_df.drop('timestamp', axis=1, inplace=True)
        movie_ratings = pd.merge(ratings_df, movies_df, on='movieId')
        movie_ratings_user = movie_ratings.pivot_table(index='userId', columns='title', values='rating')

        posters_mapping = pd.Series(posters_df.poster.values, index=posters_df.title.values)

        '''Calculate the cosine_similarities'''
        tfidf = TfidfVectorizer(stop_words='english', min_df=2, max_df=0.7)
        tfidf_matrix = tfidf.fit_transform(movies_df['plot'])
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


        self.metadata = metadata_df
        self.ratings = ratings_df
        self.movies = movies_df
        self.posters = posters_mapping
        self.movie_ratings_user = movie_ratings_user
        self.cosine_sim = cosine_sim
        self.indices = pd.Series(index=movies_df.title.values,data=movies_df.index)

    def get_user_ratings(self,userId):
        user_ratings = self.movie_ratings_user.iloc[userId]
        user_ratings.dropna(inplace=True)
        return user_ratings

    def get_recommendations(self,movieId, already_seen=[]):
        movies_df = self.movies
        movie_sim = pd.DataFrame(index=movies_df.title, data=self.cosine_sim[movieId, :], columns=['similarity'])
        movie_sim.drop(index=already_seen, inplace=True)
        movie_sim.sort_values(inplace=True, by='similarity', ascending=False)
        #   print(movie_sim.shape)
        return movie_sim[1:6]

    def get_votes(self,userId):
        top_ratings = self.get_user_ratings(userId).sort_values(ascending=False)
        already_seen = top_ratings.index.values
        #   print(already_seen.shape)
        liste = []
        for r in top_ratings.index[:5]:
            recommendations = self.get_recommendations(self.indices[r], already_seen)
            recommendations['root'] = r
            liste.append(recommendations)
        df = pd.concat(liste)

        df_out = pd.DataFrame(index=df.index.unique())
        df_out['score'] = 0
        df_out['roots'] = ''

        for row in df.iterrows():
            df_out.loc[row[0], 'score'] += 1
            df_out.loc[row[0], 'roots'] += row[1].root + '|'

        df_out['roots'] = df_out.roots.str.split('|')

        for row in df_out.iterrows():
            row[1].roots.pop()

        df_out.sort_values(by='score', ascending=False, inplace=True)
        return df_out

    def get_weighted_votes(self,userId):
        top_ratings = self.get_user_ratings(userId).sort_values(ascending=False)
        top_ratings = top_ratings[top_ratings >= 4]
        already_seen = top_ratings.index.values
        liste = []
        for r in top_ratings.index:
            recommendations = self.get_recommendations(self.indices[r],already_seen)
            recommendations['root'] = r
            liste.append(recommendations)
        df = pd.concat(liste)
        print('recommended movies: ',df.index.shape)
        print('unique recommended movies: ',df.index.unique().shape)
        df_out = pd.DataFrame(index=df.index.unique())
        df_out['score'] = 0
        df_out['roots'] = ''

        for row in df.iterrows():
            rating = top_ratings[row[1].root]
            similarity = row[1].similarity
            score = similarity * rating
            df_out.loc[row[0],'score'] += score
            df_out.loc[row[0],'roots'] += row[1].root+'|'

        df_out['roots'] = df_out.roots.str.split('|')

        for row in df_out.iterrows():
            row[1].roots.pop()
        df_out.sort_values(by='score',ascending=False,inplace=True)
        return df_out

    def get_scaled_scores(self,userId):
        top_ratings = self.get_user_ratings(userId).sort_values(ascending=False)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_ratings = scaler.fit_transform(top_ratings.values.reshape(-1, 1))
        top_ratings_scaled = pd.DataFrame(data=scaled_ratings, index=top_ratings.index, columns=['scaled_ratings'])

        already_seen = top_ratings_scaled.index.values
        liste = []
        for r in top_ratings.index:
            recommendations = self.get_recommendations(self.indices[r], already_seen)
            recommendations['root'] = r
            liste.append(recommendations)
        top_ratings_scaled
        df = pd.concat(liste)


        df_out = pd.DataFrame(index=df.index.unique())
        df_out['score'] = 0
        df_out['roots'] = ''
        df_out['root_scores'] = ''

        for row in df.iterrows():
            scaled_rating = top_ratings_scaled.loc[row[1].root].scaled_ratings
            similarity = row[1].similarity
            scaled_score = similarity * scaled_rating
            df_out.loc[row[0], 'score'] += scaled_score
            df_out.loc[row[0], 'roots'] += row[1].root + '|'
            df_out.loc[row[0], 'root_scores'] += str(round(scaled_score, 2)) + '|'

        df_out['roots'] = df_out.roots.str.split('|')
        df_out['root_scores'] = df_out.root_scores.str.split('|')
        # remove the first empty string '' in the lists
        for row in df_out.iterrows():
            row[1].roots.pop()
            row[1].root_scores.pop()
        df_out.sort_values(by='score', ascending=False, inplace=True)
        return df_out

