# Movie Recommender System
### Summary
Throughout this project, multiple recommender systems was developped leveraging the MovieLens Dataset. These included the popularity model, Spark ALS, and LightFM. Our assessment involved a comprehensive comparison of these models, evaluating their ranking performance. This enabled us to tailor recommendations based on specific use cases. The popularity model stands out for those seeking a straightforward, non-personalized recommender system. Spark ALS proves advantageous for users interested in leveraging Spark for dataset construction. Meanwhile, LightFM emerges as the recommendation for those prioritizing heightened overall accuracy while minimizing computation and fitting time.

In this project, we utilized two versions of the MovieLens dataset: a smaller set comprising 100,000 ratings and 3,600 tag applications across 9,000 movies by 600 users, and a larger dataset comprising 33,000,000 ratings and 2,000,000 tag applications applied to 86,000 movies by 330,975 users. Our evaluation metric of choice was Mean Average Precision (MAP).

### Data Partition
In machine learning, the usual approach involves partitioning data into train, validation, and test sets for cross-validation. However, this methodology doesn't translate well to recommender systems. These systems face a limitation: a model can't recommend items to a user it hasn't encountered in the training set. Consequently, it's crucial to ensure that users in the validation and test sets also exist in the training set. Moreover, it's essential to avoid overlap between users in the validation and test sets since both need to remain distinct, ensuring that the validation set generalizes effectively across all users without overlap with the test set.

**Data Partition Code Utilizing SQL and Spark**
<img src=/Images/spark_sql.png width=80% height=80%>


### Baseline Popularity Model
A recommender system is an algorithm designed to forecast the score or ranking a user might assign to an item. A basic example is the popularity model, which calculates the average rank for each item. However, this model predicts an identical ranking for all users, resulting in a lack of personalization.

### Alternating Least Suqare (ALS) Model
A more advanced recommendation model utilizes Spark's alternating least squares (ALS) method to acquire latent factor representations for both users and items.

**Evaluation Results Comparing Baseline and ALS Using Precition at k and MAP**
<img src=/Images/baseline_als.png width=60% height=60%>

### LightFM Model
LightFM operates as a single-machine implementation of a recommender system, equipped with its own Python library, eliminating the necessity for Spark usage.

**Python Code Calculating MAP using Numpy and Pandas**
<img src=/Images/lightfm_map.png width=80% height=80%>

**Evaluation Results Comparing Baseline and ALS Using Precition at k and MAP**
<img src=/Images/als_lightfm.png width=60% height=60%>









