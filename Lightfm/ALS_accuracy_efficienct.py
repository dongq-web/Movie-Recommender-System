# -*- coding: utf-8 -*-
"""
Created on Sun May 15 16:47:29 2022

@author: 14376
"""

#This code was adapted form the ALS section

from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import col, row_number, collect_list, explode
import pyspark.sql.functions as f
from pyspark.ml.evaluation import RegressionEvaluator, RankingEvaluator
from pyspark.ml.recommendation import ALS
import time

def main(spark):
    # Load data
    train_small = spark.read.csv('hdfs:/user/qd2046/train_test_small.csv', schema='userId INT, movieId INT, rating FLOAT, timestamp LONG', header=True)
    test_small = spark.read.csv('hdfs:/user/qd2046/test_modified_small.csv', schema='userId INT, movieId INT, rating FLOAT, timestamp LONG', header=True)
    train_large = spark.read.csv('hdfs:/user/qd2046/train_test_large.csv', schema='userId INT, movieId INT, rating FLOAT', header=True)
    test_large = spark.read.csv('hdfs:/user/qd2046/test_modified_large.csv', schema='userId INT, movieId INT, rating FLOAT', header=True)

    # Build ALS Model
    als = ALS(rank=50, maxIter=20, regParam=0.02, userCol="userId", itemCol="movieId", ratingCol="rating")
    start = time.time()
    model = als.fit(train_small)
    end = time.time()
    print("time to fit the small dataset is", end-start)
    
    # Get Prediction
    users = test_small.select(test_small.userId).distinct()
    userRecs = model.recommendForUserSubset(users, 100)
    
    # Evaluation
    predictions = userRecs.withColumn("recommendations", explode(userRecs.recommendations)).select('userId', col("recommendations.movieId"), col("recommendations.rating").alias('prediction')) 
    w = Window.partitionBy('userId').orderBy(col("prediction").desc())
    score = predictions.withColumn("movieId",predictions.movieId.cast('double')).withColumn('score', f.collect_list('movieId').over(w)).groupBy('userId').agg(f.max('score').alias('score'))
    w = Window.partitionBy('userId').orderBy(col("rating").desc())
    label = test_small.withColumn("movieId",test_small.movieId.cast('double')).withColumn('label', f.collect_list('movieId').over(w)).groupBy('userId').agg(f.max('label').alias('label'))
    scoreAndLabels = score.join(label, 'userId')
    
    evaluator = RankingEvaluator(predictionCol='score', labelCol='label', metricName='meanAveragePrecision', k=100)
    print("MAP(implicit) for small dataset= " + str(evaluator.evaluate(scoreAndLabels)))
    
    evaluator = RankingEvaluator(predictionCol='score', labelCol='label', metricName='precisionAtK', k=100)
    print("precision at 100(implicit) for small dataset= " + str(evaluator.evaluate(scoreAndLabels)))
    
    # Build ALS Model
    als = ALS(rank=100, maxIter=20, regParam=0.02, userCol="userId", itemCol="movieId", ratingCol="rating")
    start = time.time()
    model = als.fit(train_large)
    end = time.time()
    print("time to fit the large dataset is", end-start)

    # Get Prediction
    users = test_large.select(test_large.userId).distinct()
    userRecs = model.recommendForUserSubset(users, 100)
    
    # Evaluation
    predictions = userRecs.withColumn("recommendations", explode(userRecs.recommendations)).select('userId', col("recommendations.movieId"), col("recommendations.rating").alias('prediction')) 
    w = Window.partitionBy('userId').orderBy(col("prediction").desc())
    score = predictions.withColumn("movieId",predictions.movieId.cast('double')).withColumn('score', f.collect_list('movieId').over(w)).groupBy('userId').agg(f.max('score').alias('score'))
    w = Window.partitionBy('userId').orderBy(col("rating").desc())
    label = test_large.withColumn("movieId",test_large.movieId.cast('double')).withColumn('label', f.collect_list('movieId').over(w)).groupBy('userId').agg(f.max('label').alias('label'))
    scoreAndLabels = score.join(label, 'userId')
    
    evaluator = RankingEvaluator(predictionCol='score', labelCol='label', metricName='meanAveragePrecision', k=100)
    print("MAP(implicit) for large dataset= " + str(evaluator.evaluate(scoreAndLabels)))
    
    evaluator = RankingEvaluator(predictionCol='score', labelCol='label', metricName='precisionAtK', k=100)
    print("precision at 100(implicit) for large dataset= " + str(evaluator.evaluate(scoreAndLabels)))
    
    
if __name__ == '__main__':
    spark = SparkSession.builder.appName('latent').getOrCreate()
    main(spark)