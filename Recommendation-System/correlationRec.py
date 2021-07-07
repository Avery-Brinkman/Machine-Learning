import numpy
import pandas
import os 

os.chdir('Recommendation-System')


# [userID | placeID | rating (0-2) | food_rating (0-2) | service_rating (0-2)]
frame = pandas.read_csv('data/correlation/rating_final.csv')
# [placeID | Rcuisine]
cuisine = pandas.read_csv('data/correlation/chefmozcuisine.csv')
# [placeID | lat | long | the_geom_meter | name | address | city | state | country 
#   | fax | zip | alcohol | smoking_area | dress_code | accessibility | price 
#   | url | Rambience | franchise | area | other_services]
geodata = pandas.read_csv('data/correlation/geoplaces2.csv', encoding = 'ISO-8859-1')

# print(frame.head(), '\n')
# print(cuisine.head(), '\n')
# print(geodata.head(), '\n')


# Subset of geodata
places = geodata[['placeID', 'name']]
# print(places.head(), '\n')


# Create a new dataframe from frame called rating. Groups each placeID from frame
#   and looks at the rating value, and takes the mean.
rating = pandas.DataFrame(frame.groupby('placeID')['rating'].mean())

# Adds a column to rating called rating_count, and sets it equal to a dataframe
#   that groups frame's placeIDs and takes the count of they're ratings.
rating['rating_count'] = pandas.DataFrame(frame.groupby('placeID')['rating'].count())
# print(rating.head(), '\n')

rating = rating.sort_values('rating_count', ascending=False)
# print(rating.head(), '\n')

# Print the value from places when in places' placeID is 135085 (top rated restaurant)
# print(places[places['placeID']==135085])
# Print the value from cuisine when cuisine's placeID is 135085
# print(cuisine[cuisine['placeID']==135085], '\n')

places_crosstab = pandas.pivot_table(data=frame, values='rating', index='userID', columns='placeID')
# print(places_crosstab.head(), '\n')

# Tortas_ratings is a pandas series equal to the values from the column titled 135085 in places_crosstab,
#   which is each individual rating it received (includes userID of rating).
Tortas_ratings = places_crosstab[135085]
# Sets Tortas_ratings equal to Tortas_ratings where the value of Tortas_ratings is greater than 0 (not null).
Tortas_ratings = Tortas_ratings[Tortas_ratings>=0]
# print(Tortas_ratings)

# Sets similar_to_Tortas equal to a matrix 
similar_to_Tortas = places_crosstab.corrwith(Tortas_ratings)
# Sets corr_Tortas to a dataframe 
corr_Tortas = pandas.DataFrame(similar_to_Tortas, columns=['PearsonR'])
# Filters out NaN values
# corr_Tortas is a table [placeID | PearsonR] where PearsonR is an R value representing how similar
#   the restaurant is to Tortas.
corr_Tortas.dropna(inplace=True)
# print(corr_Tortas.head(), '\n')


# Sets Tortas_corr_summary equal to corr_Tortas, but with rating's rating_count column added on.
Tortas_corr_summary = corr_Tortas.join(rating['rating_count'])
# Filters out anything with less than 10 ratings, and sorts them by descending R value.
Tortas_corr_summary = Tortas_corr_summary[Tortas_corr_summary['rating_count']>=10].sort_values('PearsonR', ascending=False)
# print(Tortas_corr_summary.head(10), '\n')
# This shows lots of places with pearson r value of 1. These values aren't meaningful. These are 
#   here because there was only one user who gave a reveiw to both places, and that user gave 
#   both places the same score. Correlations need to have more than one reveiwer in common.

places_corr_Tortas = pandas.DataFrame([135085, 132754, 135045, 135062, 135028, 135042, 135046], index=numpy.arange(7), columns=['placeID'])
summary = pandas.merge(places_corr_Tortas, cuisine, on='placeID')
print(summary.head(), '\n')

# Everything we've done shows that Restaurante El Reyecito would be a good recommendation for people who liked 
#   Tortas Locas Hipocampo. This is because a statistically significant number of people rated both Tortas 
#   and Reyecito well. Reyecito, like Tortas, ranks highly in restuarant ratings and serves the same kind of food.
print(places[places['placeID'] == 135046])