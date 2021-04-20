#######################################################################################
#######################################################################################
#                   Movielens Project R Scripts                                       #
#                        Arezou Tufekci                                               #
#                       April 20, 2021                                                #
# Train a Machine Learning algorithm using the inputs in one subset to predict movie  #            
# ratings in the validation set. Generate predicted movie rating and use RMSE         # 
# as the measure of accuracy to evaluate how close our predictions are to the true    #
# values in the validation set. .                                                     #
#######################################################################################

# We used the following code given in the instruction to generate the datasets. We will create
# edx and Validation datasets in the code below.
# Later, we develop our algorithm using the edx set to create the train and test set from edx set. 

# First install necessary packages and then libraries for our work
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

##########################################################################################
# Create edx set and validation set (final hold-out test set)                            #
##########################################################################################

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

##########################################################################################
#                           Data Exploration: Movie Ratings                              #
#    The most rated movie has 31,362 Movie Ratings.126 single time rated movies. Large   #
#                           variety of ratings on each movie.                            #
##########################################################################################

# Most rated movies
edx  %>%
  group_by(title)   %>%
  summarize(number_ratings = n())   %>%
  arrange(desc(number_ratings))

# Number of single rated movies
edx %>% 
  group_by(title) %>%
  summarize(number_ratings = n()) %>%
  filter(number_ratings==1) %>%
  count() %>% 
  pull() 

####################################################
# Create Training & Test Datasets from edx dataset
####################################################
set.seed(1, sample.kind="Rounding")

# The train set will be 90% of the edx data and the test set will be the remaining 10%
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-test_index,]
temp <-  edx[test_index,]

# Make sure 'movieId' and 'userId' which exist in the test set are also in then train set
test_set <- temp %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Add back the rows removed from test set into train set
removed <-  anti_join(temp, test_set)
train_set <- rbind(train_set, removed)
rm(test_index, temp, removed)
 
# Reduce columns down to title, rating, movie id, user id. less complexity, less resource usage
train_set <- train_set  %>% 
  select(userId, movieId, rating, title)
test_set  <- test_set  %>% 
  select(userId, movieId, rating, title)

################################################################################################
# Examine Three Main Methods                                                                   #
################################################################################################


######################################################
# Method 1: Random Prediction
######################################################
# Plot a symetric distribution of Movies and ratings
edx %>%   group_by(movieId) %>%
  summarize(n = n()) %>%
  ggplot(aes(n)) +
  geom_histogram(color = "pink") +
  scale_x_log10()  +
  xlab("# of Ratings") +
  ylab("# of Movies") 


##################################################################
# Use Monte Carlos Simulation to guess probability of each rating
##################################################################
set.seed(1, sample.kind = "Rounding")

# Build the probability for each movie rating
pr <- function(x, y) 
              mean(y == x)
rating <- seq(0.5,5,0.5)

# Guess the probability of each rating utilizing Monte Carlo simulation
B <- 10000
Mont <- replicate(B, {
  samp <- sample(train_set$rating, 100, replace = TRUE)
  sapply(rating, pr, y= samp)
})
prob <- sapply(1:nrow(Mont), function(x)
                                     mean(Mont[x,]))

# Predict random ratings
y_hat_random <- sample(rating,  size = nrow(test_set),  replace = TRUE,  prob = prob)

#Random Prediction Method. Use test set and then calculate RMSE
RMSE(test_set$rating, y_hat_random)

######################################################
# Method 2: Linear Models
######################################################

# Linear Model - Get Mean and calculate RMSE
mu <- mean(edx$rating)
RMSE(test_set$rating, mu)

# Linear Model - Mean + Movie bias and calculate RMSE
bi <- train_set     %>%
      group_by(movieId)     %>%
      summarize(bi = mean(rating - mu))

# All (unknown) movie ratings prediction using mu and bi
predicted_ratings <- test_set     %>%
                     left_join(bi, by='movieId')    %>%
                     mutate(predict = mu + bi)      %>%
                     pull(predict)

# Now calculate RMSE of movie affect
RMSE(test_set$rating, predicted_ratings)

# Now we add user bias term = bu
bu <- train_set  %>%
      left_join(bi, by='movieId')   %>%
      group_by(userId)    %>%
      summarize(bu = mean(rating - mu - bi))

# predicting new movie ratings bringing in movie and user bias
predicted_ratings <- test_set    %>%
                     left_join(bi, by='movieId')   %>%
                     left_join(bu, by='userId')    %>%
                     mutate(predict = mu + bi + bu)   %>%
                     pull(predict)

# calculate RMSE of movie + user bias effect
RMSE(predicted_ratings, test_set$rating)


######################################################
# Method 3: Regularization
######################################################

# Write a regularization function to fine tune (need lambda)
regularization <- function(lambda, train, test) {
  
  # trainset mean
  mu <- mean(train$rating)
  # Movie bias/effect, b_i
  b_i <- train    %>% 
    group_by(movieId)   %>%
    summarize(b_i = sum(rating - mu)/(n()+lambda))
  # User bias/effect, b_u  
  b_u <- train   %>% 
         left_join(b_i, by = "movieId")   %>%
         filter(!is.na(b_i))    %>%
         group_by(userId)    %>%
         summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))
  # Rating prediction using mean + b_i + b_u (remove na's) 
  predicted_ratings <- test   %>% 
                       left_join(b_i, by = "movieId")   %>%
                       left_join(b_u, by = "userId")    %>%
                       filter(!is.na(b_i), !is.na(b_u))     %>%
                       mutate(pred = mu + b_i + b_u)    %>%
                       pull(pred)
  #use test data and calculate RMSE
  return(RMSE(predicted_ratings, test$rating))
}

# An effective method to choose lambda that minimizes the RMSE is to run
# simulations with several values of lambda
# Define a set (lambdas)
lambdas <- seq(0, 10, 0.25)

# Tuning by using sapply
RMSEs <- sapply(lambdas, regularization, train = train_set, test = test_set)

# Plot the lambda vs RMSE to find out the minimum RMSE
# useful to use tibble on big datasets
tibble(Lambda = lambdas, RMSE = RMSEs) %>%
       ggplot(aes(x = Lambda, y = RMSE)) +
       geom_point()

# Pick lambda that returns the minimum RMSE using which function
lambda <- lambdas[which.min(RMSEs)]
lambda

# Now we have Lambda that returns min RMSE, so we plug it in
mu <- mean(train_set$rating)

# bring in Movie effect, bi into the new model
b_i <- train_set    %>% 
       group_by(movieId)      %>%
       summarize(b_i = sum(rating - mu)/(n()+lambda))

# bring in user effect, bu into the new model
b_u <- train_set   %>% 
       left_join(b_i, by = "movieId")  %>%
       group_by(userId)   %>%
       summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))

# Prediction now that we have fine tuned using lambda
y_hat_reg <- test_set %>% 
             left_join(b_i, by = "movieId") %>%
             left_join(b_u, by = "userId") %>%
             mutate(pred = mu + b_i + b_u) %>%
             pull(pred)

# Use Test set and calculate RMSE
RMSE(test_set$rating, y_hat_reg)

##################################################################
# Final Validation with the Validation Set
##################################################################

mu_edx <- mean(edx$rating)

# Movie effect (bi)
b_i_edx <- edx %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu_edx)/(n()+lambda))

# User effect (bu)
b_u_edx <- edx %>% 
           left_join(b_i_edx, by="movieId") %>%
           group_by(userId) %>%
           summarize(b_u = sum(rating - b_i - mu_edx)/(n()+lambda))

# Prediction
y_hat_edx <- validation %>% 
             left_join(b_i_edx, by = "movieId") %>%
             left_join(b_u_edx, by = "userId") %>%
             mutate(pred = mu_edx + b_i + b_u) %>%
             pull(pred)

RMSE(validation$rating, y_hat_edx)


