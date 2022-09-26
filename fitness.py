import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# loading the data from the csv file to apandas dataframe
workout_data = pd.read_csv('/content/Workout.csv')

workout_data.shape

# selecting the relevant features for recommendation
selected_features = ['Bodyarea','Exercise Name']

# replacing the null valuess with null string

for feature in selected_features:
  workout_data[feature] = workout_data[feature].fillna('')

# combining all the selected features
combined_features = workout_data['Bodyarea']+' '+workout_data['Exercise Name']

# converting the text data to feature vectors
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# getting the similarity scores using cosine similarity
similarity = cosine_similarity(feature_vectors)

# getting the weight of the user
print("You can Workout for the following")
print("Chest")
print("Lower")
print("Shoulder")
print("Back")
print("Legs")
print("                                        ")

body_part = input('Enter the body part you want to exercise for : ')

# creating a list with all the Workouts given in the dataset
list_of_all_workouts = workout_data['Bodyarea'].tolist()

# finding the close match for the weight given by the user
find_close_match = difflib.get_close_matches(body_part, list_of_all_workouts)
close_match = find_close_match[0]
print(close_match)

# finding the body area

index_of_workout = workout_data[workout_data.Bodyarea == close_match]['index'].values[0]
print(index_of_workout)

# getting a list of similar workouts
similarity_score = list(enumerate(similarity[index_of_workout]))

# sorting the exercise based on the similarity score

sorted_similar_workout= sorted(similarity_score, key = lambda x:x[1], reverse = True) 


# print the name of similar movies based on the index

print('Exercises suggested for you : \n')

i = 1
print('Exercise                 |   Reps   | Average Weight(in lbs) | Sets')

for w in sorted_similar_workout:
  index = w[0]
  workout_from_index = workout_data[workout_data.index==index]['Exercise Name'].values[0]
  if (i<30):
    print(i, '.',workout_from_index,'    | ',workout_data.Reps[i],' |    ',workout_data.AverageWeight[i],'      |    ',workout_data.Set[i])
    i+=1
