import os
import pandas
import numpy

os.chdir('Recommendation-System')

# Restaurant ratings [UserID | RestaurantID | Overall | Food | Service] (Scale 0-2)
frame = pandas.read_csv('data/popularity/rating_final.csv')
# IDs each restaurant [RestaurantID | TypeOfFood] 
cuisine = pandas.read_csv('data/popularity/chefmozcuisine.csv')
print(frame.head(),'\n')

# Creates a new DataFrame by grouping data by placeID and taking the count of rating values
rating_count = pandas.DataFrame(frame.groupby('placeID')['rating'].count())
# Sorts data
sorted_rating_count = rating_count.sort_values('rating', ascending=False)
print(sorted_rating_count.head(),'\n')

most_rated_places = pandas.DataFrame([135085,132825,135032,135052,132834], index=numpy.arange(5), columns=['placeID'])

summary = pandas.merge(most_rated_places, cuisine, on='placeID') 
print(summary,'\n')

print(cuisine['Rcuisine'].describe(),'\n')