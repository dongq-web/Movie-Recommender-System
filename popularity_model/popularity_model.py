# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 21:04:08 2022

@author: 14376
"""

import sys
from pyspark.sql import SparkSession

def main(spark, train_path, test_path, movie_path, name):
    #read in train, test, movie data
    train = spark.read.parquet(train_path, schema = "userId INT, movieId INT, rating FLOAT, timestamp LONG", header=True)
    test = spark.read.parquet(test_path, schema = "userId INT, movieId INT, rating FLOAT, timestamp LONG", header=True)
    train.createOrReplaceTempView('train')
    test.createOrReplaceTempView('test')
    movies = spark.read.csv(movie_path, schema = "movieId INT, title STRING, genres STRING")
    movies.createOrReplaceTempView("movies")

    #create predicted rating for every movie
    predict_rating = spark.sql("select t.movieId, first(m.title) as movie_title, avg(t.rating) as predict_rating \
    from train t join movies m on t.movieId = m.movieId group by t.movieId order by avg(t.rating) DESC")
    predict_rating.createOrReplaceTempView('predict_rating')
    
    #create prediction for test data
    prediction = spark.sql("select t.userId, t.movieId, t.rating, t.timestamp, p.predict_rating from test t \
    join predict_rating p on t.movieId = p.movieId order by userId")
    prediction.write.csv("prediction_"+name+'.csv', header=True)
    
    #create top 100 recommendation for every user
    top100 = spark.sql("select * from predict_rating order by predict_rating desc")
    top100.write.csv("top100_"+name+'.csv', header=True)

if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('popularity').getOrCreate()

    # Get file_path for dataset to analyze
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    movie_path = sys.argv[3]
    name = sys.argv[4]

    main(spark, train_path, test_path, movie_path, name)