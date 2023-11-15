# -*- coding: utf-8 -*-
"""
Created on Sun May 15 16:59:15 2022

@author: zoeyu
"""

from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RankingEvaluator
from pyspark.sql.window import Window
from pyspark.sql.functions import col, row_number, collect_list, explode
import pyspark.sql.functions as f
import numpy as np

def main(spark):
    # Load data
    test_small = spark.read.csv('hdfs:/user/yt2336/test_small.csv', schema='userId INT, movieId INT, rating FLOAT, timestamp LONG', header=True)
    test_large = spark.read.csv('hdfs:/user/yt2336/test_large.csv', schema='userId INT, movieId INT, rating FLOAT, timestamp LONG', header=True)
    baseline_small = spark.read.csv('hdfs:/user/yt2336/top100_small.csv', schema='movieId INT, movie_title STRING, predict_rating DOUBLE', header=True)
    baseline_large = spark.read.csv('hdfs:/user/yt2336/top100_large.csv', schema='movieId INT, movie_title STRING, predict_rating DOUBLE', header=True)

    # baseline evaluation for small dataset
    w = Window.partitionBy('userId').orderBy(col("rating").desc())
    label = test_small.withColumn("movieId",test_small.movieId.cast('double')).withColumn('label', f.collect_list('movieId').over(w)).groupBy('userId').agg(f.max('label').alias('label'))
    scoreAndLabels = label.select(label.label).withColumn("score", f.array(*map(f.lit, np.array(baseline_small.select(baseline_small.movieId).withColumn("movieId",baseline_small.movieId.cast('double')).collect()).flatten())))

    evaluator = RankingEvaluator(predictionCol='score', labelCol='label', metricName='meanAveragePrecision', k=100)
    print("MAP(implicit) for small dataset= " + str(evaluator.evaluate(scoreAndLabels)))

    evaluator = RankingEvaluator(predictionCol='score', labelCol='label', metricName='precisionAtK', k=100)
    print("precision at 100(implicit) for small dataset= " + str(evaluator.evaluate(scoreAndLabels)))

    evaluator = RankingEvaluator(predictionCol='score', labelCol='label', metricName='ndcgAtK', k=100)
    print("ndcg at 100(implicit) for small dataset= " + str(evaluator.evaluate(scoreAndLabels)))

    table = test_small.select(test_small.userId).distinct().withColumn("movieId", f.array(*map(f.lit, np.array(baseline_small.select(baseline_small.movieId).withColumn("movieId",baseline_small.movieId.cast('double')).collect()).flatten())))
    table = table.withColumn("movieId", explode(table.movieId)).withColumn("row_number",row_number().over(Window.partitionBy('userId').orderBy('userId'))).join(test_small.select(test_small.userId, test_small.movieId, test_small.rating),['userId','movieId'])
    w = Window.partitionBy('userId').orderBy(col("rating").desc())
    label = table.withColumn('label', f.collect_list('movieId').over(w)).groupBy('userId').agg(f.max('label').alias('label'))
    w = Window.partitionBy('userId').orderBy(col("row_number"))
    score = table.filter(table.rating > 2.5).withColumn('score', f.collect_list('movieId').over(w)).groupBy('userId').agg(f.max('score').alias('score'))
    scoreAndLabels_e = score.join(label, 'userId')

    evaluator = RankingEvaluator(predictionCol='score', labelCol='label', metricName='meanAveragePrecision', k=100)
    print("MAP(explicit) for small dataset= " + str(evaluator.evaluate(scoreAndLabels_e)))

    evaluator = RankingEvaluator(predictionCol='score', labelCol='label', metricName='precisionAtK', k=100)
    print("precision at 100(explicit) for small dataset= " + str(evaluator.evaluate(scoreAndLabels_e)))

    evaluator = RankingEvaluator(predictionCol='score', labelCol='label', metricName='ndcgAtK', k=100)
    print("ndcg at 100(explicit) for small dataset= " + str(evaluator.evaluate(scoreAndLabels_e)))

    # baseline evaluation for large dataset
    w = Window.partitionBy('userId').orderBy(col("rating").desc())
    label = test_large.withColumn("movieId",test_large.movieId.cast('double')).withColumn('label', f.collect_list('movieId').over(w)).groupBy('userId').agg(f.max('label').alias('label'))
    scoreAndLabels = label.select(label.label).withColumn("score", f.array(*map(f.lit, np.array(baseline_large.select(baseline_large.movieId).withColumn("movieId",baseline_large.movieId.cast('double')).collect()).flatten())))

    evaluator = RankingEvaluator(predictionCol='score', labelCol='label', metricName='meanAveragePrecision', k=100)
    print("MAP(implicit) for large dataset= " + str(evaluator.evaluate(scoreAndLabels)))

    evaluator = RankingEvaluator(predictionCol='score', labelCol='label', metricName='precisionAtK', k=100)
    print("precision at 100(implicit) for large dataset= " + str(evaluator.evaluate(scoreAndLabels)))

    evaluator = RankingEvaluator(predictionCol='score', labelCol='label', metricName='ndcgAtK', k=100)
    print("ndcg at 100(implicit) for large dataset= " + str(evaluator.evaluate(scoreAndLabels)))

    table = test_large.select(test_large.userId).distinct().withColumn("movieId", f.array(*map(f.lit, np.array(baseline_large.select(baseline_large.movieId).withColumn("movieId",baseline_large.movieId.cast('double')).collect()).flatten())))
    table = table.withColumn("movieId", explode(table.movieId)).withColumn("row_number",row_number().over(Window.partitionBy('userId').orderBy('userId'))).join(test_large.select(test_large.userId, test_large.movieId, test_large.rating),['userId','movieId'])
    w = Window.partitionBy('userId').orderBy(col("rating").desc())
    label = table.withColumn('label', f.collect_list('movieId').over(w)).groupBy('userId').agg(f.max('label').alias('label'))
    w = Window.partitionBy('userId').orderBy(col("row_number"))
    score = table.filter(table.rating > 2.5).withColumn('score', f.collect_list('movieId').over(w)).groupBy('userId').agg(f.max('score').alias('score'))
    scoreAndLabels_e = score.join(label, 'userId')

    evaluator = RankingEvaluator(predictionCol='score', labelCol='label', metricName='meanAveragePrecision', k=100)
    print("MAP(explicit) for large dataset= " + str(evaluator.evaluate(scoreAndLabels_e)))

    evaluator = RankingEvaluator(predictionCol='score', labelCol='label', metricName='precisionAtK', k=100)
    print("precision at 100(explicit) for large dataset= " + str(evaluator.evaluate(scoreAndLabels_e)))

    evaluator = RankingEvaluator(predictionCol='score', labelCol='label', metricName='ndcgAtK', k=100)
    print("ndcg at 100(explicit) for large dataset= " + str(evaluator.evaluate(scoreAndLabels_e)))
    
if __name__ == '__main__':
    spark = SparkSession.builder.appName('baseline').getOrCreate()
    main(spark)