import pdb #byebug for python --> pdb.set_trace()
from numpy import *

num_movies = 10

num_users = 5

ratings = random.randint(11, size = (num_movies, num_users))

did_rate = (ratings != 0) * 1

sheraz_ratings = zeros((num_movies, 1))

sheraz_ratings[0] = 8
sheraz_ratings[4] = 7
sheraz_ratings[7] = 3

ratings = append(sheraz_ratings, ratings, axis = 1)
did_rate = append(((sheraz_ratings != 0) * 1), did_rate, axis = 1)

pdb.set_trace()

a = [10, 20, 30]
aSum = sum(a)


aMean = aSum / 3


aMean = mean(a)

a = [10 - aMean, 20 - aMean, 30 - aMean]


def normalize_ratings(ratings, did_rate):
    num_movies = ratings.shape[0]
    
    ratings_mean = zeros(shape = (num_movies, 1))
    ratings_norm = zeros(shape = ratings.shape)
    
    for i in range(num_movies): 
        # Get all the indexes where there is a 1
        idx = where(did_rate[i] == 1)[0]
        #  Calculate mean rating of ith movie only from user's that gave a rating
        ratings_mean[i] = mean(ratings[i, idx])
        ratings_norm[i, idx] = ratings[i, idx] - ratings_mean[i]
    
    return ratings_norm, ratings_mean

ratings, ratings_mean = normalize_ratings(ratings, did_rate)
print ratings_mean
print ratings