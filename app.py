
# Turn into streamlit:
import pandas as pd
from sklearn.neighbors import NearestNeighbors as KNN
import streamlit as st
from difflib import get_close_matches


fpath = f'data/movies.csv'
# load data
data = pd.read_csv(fpath)

data_tmp = data.copy()
data_tmp['year'] = data_tmp['title'].str.extract(r'\((\d{4})\)')
data_tmp = data_tmp.dropna(subset = ['year'])
data_tmp['title'] = data_tmp['title'].str.replace(r'\s*\(\d{4}\)$', '', regex=True)


# Split genres into new columns
data_tmp['genres'] = data_tmp['genres'].str.split('|')
genres = set(genre for sublist in data_tmp['genres'] for genre in sublist)
genres.remove('(no genres listed)')


# Convert the list of subgenres into a format suitable for get_dummies
data_tmp['subgenres_str'] = data_tmp['genres'].apply(lambda x: ', '.join(x))
# Create binary columns for each genre
genre_dummies = data_tmp['subgenres_str'].str.get_dummies(sep=', ')
# Merge the dummy columns with the original DataFrame
data_tmp = pd.concat([data_tmp, genre_dummies], axis=1)



# using k nearest neigbor, find 4 of the most similar movies by genre by inputting a name 
# Extract features (binary genre columns)
features = data_tmp.drop(['title', 'movieId', 'genres', 'year', 'subgenres_str', '(no genres listed)'], axis=1)
features.head()
# Fit the k-NN model
knn = KNN(n_neighbors=4, metric='euclidean')  # Choose 'hamming' or 'cosine' if desired
knn.fit(features)



# create a search feature that has the user type in a movie and then checks and finds the closest match and approves it with the user
movies = data_tmp['title'].tolist()
st.title("Movie Search Interface")

search_query = st.text_input("Please enter a movie you like, and I will give you 4 movies that I think you will like", "")  

found_match = False
closest_match = []

if search_query:
    # Find closest matches
    closest_matches = get_close_matches(search_query, movies, n=1, cutoff=0.8)
    
    if closest_matches:
        closest_match = closest_matches[0]
        st.write(f"Closest match: **{closest_match}**")
        
        # Yes or no Buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes"):
                st.success(f"Great! You found your movie: **{closest_match}**")
                found_match = True                
                closest_index = movies.index(closest_match)
        with col2:
            if st.button("No"): 
                st.error(f"Sorry about that. Try searching again")
                found_match = False
    
    else:
        st.write("No close matches found. Please try a different search.")
        

if found_match == True:
    # Query for similar movies (use the same feature set)
    movie_index = closest_index
    distances, indices = knn.kneighbors([features.iloc[movie_index]])

    # Get results
    print(f"Similar movies to '{data_tmp.iloc[movie_index]['title']}':")
    for idx, dist in zip(indices[0], distances[0]):
        print(f"- {data_tmp.iloc[idx]['title']} (Distance: {dist:.2f})")
    
    
if found_match:
    # Query for similar movies
    movie_index = closest_index
    distances, indices = knn.kneighbors([features.iloc[movie_index]])

    # Display the title of the queried movie
    st.markdown(f"### Similar movies to: **{data_tmp.loc[movie_index]['title']}**")

    # Alternatively, display results as a bulleted list
    st.markdown("#### Recommended Movies:")
    for idx, dist in zip(indices[0], distances[0]):
        st.markdown(f"- **{data_tmp.iloc[idx]['title']}** (Distance: {dist:.2f})")

