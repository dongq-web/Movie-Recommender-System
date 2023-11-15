# -*- coding: utf-8 -*-
"""
Created on Mon May 16 19:21:14 2022

@author: zoeyu
"""

from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import col, row_number, collect_list, explode
import pyspark.sql.functions as f
from pyspark.ml.evaluation import RegressionEvaluator, RankingEvaluator
from pyspark.ml.recommendation import ALS

def main(spark):
    # Load data
    train_large = spark.read.csv('hdfs:/user/yt2336/train_large.csv', schema='userId INT, movieId INT, rating FLOAT, timestamp LONG', header=True)
    val_large = spark.read.csv('hdfs:/user/yt2336/val_large.csv', schema='userId INT, movieId INT, rating FLOAT, timestamp LONG', header=True)
    test_large = spark.read.csv('hdfs:/user/yt2336/test_large.csv', schema='userId INT, movieId INT, rating FLOAT, timestamp LONG', header=True)

    # Build ALS Model
    als = ALS(rank=100, maxIter=20, regParam=0.02, userCol="userId", itemCol="movieId", ratingCol="rating")
    model = als.fit(train_large)

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
    
    evaluator = RankingEvaluator(predictionCol='score', labelCol='label', metricName='ndcgAtK', k=100)
    print("ndcg at 100(implicit) for large dataset= " + str(evaluator.evaluate(scoreAndLabels)))
    
    table = predictions.join(test_large.select(test_large.userId, test_large.movieId, test_large.rating),['userId','movieId'])
    table = table.withColumn("movieId",table.movieId.cast('double'))
    w = Window.partitionBy('userId').orderBy(col("rating").desc())
    label = table.withColumn('label', f.collect_list('movieId').over(w)).groupBy('userId').agg(f.max('label').alias('label'))
    w = Window.partitionBy('userId').orderBy(col("prediction").desc())
    score = table.filter(table.rating > 2.5).withColumn('score', f.collect_list('movieId').over(w)).groupBy('userId').agg(f.max('score').alias('score'))
    scoreAndLabels_e = score.join(label, 'userId')
    
    evaluator = RankingEvaluator(predictionCol='score', labelCol='label', metricName='meanAveragePrecision', k=100)
    print("MAP(explicit) for large dataset= " + str(evaluator.evaluate(scoreAndLabels_e)))
    
    evaluator = RankingEvaluator(predictionCol='score', labelCol='label', metricName='precisionAtK', k=100)
    print("precision at 100(explicit) for large dataset= " + str(evaluator.evaluate(scoreAndLabels_e)))
    
    evaluator = RankingEvaluator(predictionCol='score', labelCol='label', metricName='ndcgAtK', k=100)
    print("ndcg at 100(explicit) for large dataset= " + str(evaluator.evaluate(scoreAndLabels_e)))
    
if __name__ == '__main__':
    spark = SparkSession.builder.appName('latent').getOrCreate()
    main(spark)