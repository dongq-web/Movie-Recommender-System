# Movie Recommender System
### Summary

Recommender System is a subclass of information filtering system that provide sugguestions for items that are most pertinent to a particular user. One type of recommender system, collaborative filtering, recommend items based on how similar users behave.

<img src=/Images/recommender_system.png width=50% height=50%>

In most cases, the model would takes in a sparse matrix where the rows are different users and the columns are different movies. Since most users only watched a few movies, most of the entries would be empty.

<img src=/Images/sparse_matrix.png width=50% height=50%>

Throughout this project, three recommender systems was developped leveraging the MovieLens Dataset.

1. Popularity Model
    - Calculates the average ratings for each movie. No personalization for each user
    - Pro: easy to build
    - Con: low accuracy compared to other models

2. Spark's Alternating Least Squares (ALS) Model
    - A Spark collaborative filtering model that predicts missing entries by calculating latent factors
    - Pro: built-in library in Spark with built-in evaluation metrics
    - Con: lower accuracy and higher computation time compared to LightFM model

3. LightFM Model
    - A single-machine implementation of a recommender system
    - Pro: highest accuracy among the three
    - Con: only has a few built-in evaluation metrics, need to calculate some of them by hand

### Dataset
In this project, we utilized two versions of the MovieLens dataset.
- MovieLens-Small
    - 100,000 ratings and 3,600 tag applications across 9,000 movies by 600 users
- MovieLens-Large
    - 33M ratings and 2M tag applications applied to 86,000 movies by 330,975 users

<img src=/Images/dataset.png width=50% height=50%>

In machine learning, the usual approach involves partitioning data into train, validation, and test sets for cross-validation. However, this methodology doesn't translate well to recommender systems. These systems face a limitation: a model can't recommend items to a user it hasn't encountered in the training set. Consequently, it's crucial to ensure that users in the validation and test sets also exist in the training set. Moreover, it's essential to avoid overlap between users in the validation and test sets since both need to remain distinct, ensuring that the validation set generalizes effectively across all users without overlap with the test set.

**Data Partition Code Utilizing SQL and Spark**
<img src=/Images/spark_sql.png width=80% height=80%>

### Prediction Results

**Popularity Model Results**

<img src=/Images/popularity_result.png width=50% height=50%>

Popularity model simply recommends the same movies to every user based on their average ratings.

**ALS Model Results**

<img src=/Images/als_result.png width=30% height=30%>

ALS Model gives a predicted rating to every user based on other similar users.

**LightFM Model Results**

Because the model input of LightFM model is different than the other two, the output of LightFM would look like the interaction matrix sample that was provided at the beginning, with each entries filled with the predicted **rank** (not rating) of each movie

### Performance Evaluation
The evaluation metrics are Precision at k and Mean Average Precision(MAP).

<img src=/Images/metric_example.png width=40% height=40%>

Imagine this is a list of movie recommended to a user, ordering descendingly by the rank of each movie. The green check mark means that the user watched the movie, whereas the red cross mark means the user didn't watch the movie. 

- Precision at 1 = 1/1 = 1
- Precision at 2 = 1/2 = 0.5
- Precision at 3 = 1/3 = 0.33
- Average Precision = Average of the all the Precision at k for k=1 to 100
- Mean Average Precision = Mean of Average Precision for all the users

**Python Code Calculating MAP using Numpy and Pandas**
<img src=/Images/lightfm_map.png width=60% height=60%>

**Evaluation Results Comparing Popularity and ALS Using Precition at k and MAP**
<img src=/Images/baseline_als.png width=40% height=40%>

**Evaluation Results Comparing LightFM and ALS Using Precision at k and MAP**
<img src=/Images/als_lightfm.png width=40% height=40%>

Something to note is that this recommender system is evaluated based on implicit feedback: the number of movies that a specific user watched from the list of movies that were recommended to them. Since it is impossible for a user to watch all the movies that were recommended to them, it is impossible to reach a high precision.







