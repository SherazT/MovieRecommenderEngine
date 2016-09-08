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

def unroll_params(X_and_theta, num_users, num_movies, num_features):
	# Retrieve the X and theta matrixes from X_and_theta, based on their dimensions (num_features, num_movies, num_movies)
	# Get the first 30 (10 * 3) rows in the 48 X 1 column vector
	first_30 = X_and_theta[:num_movies * num_features]
	# Reshape this column vector into a 10 X 3 matrix
	X = first_30.reshape((num_features, num_movies)).transpose()
	# Get the rest of the 18 the numbers, after the first 30
	last_18 = X_and_theta[num_movies * num_features:]
	# Reshape this column vector into a 6 X 3 matrix
	theta = last_18.reshape(num_features, num_users ).transpose()
	return X, theta


def calculate_gradient(X_and_theta, ratings, did_rate, num_users, num_movies, num_features, reg_param):
	X, theta = unroll_params(X_and_theta, num_users, num_movies, num_features)
	
	# we multiply by did_rate because we only want to consider observations for which a rating was given
	difference = X.dot( theta.T ) * did_rate - ratings
	X_grad = difference.dot( theta ) + reg_param * X
	theta_grad = difference.T.dot( X ) + reg_param * theta
	
	# wrap the gradients back into a column vector 
	return r_[X_grad.T.flatten(), theta_grad.T.flatten()]

def calculate_cost(X_and_theta, ratings, did_rate, num_users, num_movies, num_features, reg_param):
	X, theta = unroll_params(X_and_theta, num_users, num_movies, num_features)
	
	# we multiply (element-wise) by did_rate because we only want to consider observations for which a rating was given
	cost = sum( (X.dot( theta.T ) * did_rate - ratings) ** 2 ) / 2
	# '**' means an element-wise power
	regularization = (reg_param / 2) * (sum( theta**2 ) + sum(X**2))
	return cost + regularization
