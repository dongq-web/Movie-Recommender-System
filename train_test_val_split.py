# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 16:39:43 2022

@author: 14376
"""
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit

def main(spark, filepath, name):
    #read in filepath
    data = spark.read.csv(filepath, header=True, schema = "userId INT, movieId INT, rating FLOAT, timestamp LONG")
    data.createOrReplaceTempView('data')
    
    #split users into users for test and users for train
    test_user = spark.sql("select * from data where userId%2=0")
    val_user = spark.sql("select * from data where userId%2=1")
    
    #create fraction dict in order to use sampleBy
    test_fraction = test_user.select("userId").distinct().withColumn("fraction", lit(0.6)).rdd.collectAsMap()
    val_fraction = val_user.select("userId").distinct().withColumn("fraction", lit(0.6)).rdd.collectAsMap()
    
    #sampleBy to take 60% of reviews as train
    train1 = test_user.sampleBy('userId', test_fraction, 101)
    train2 = val_user.sampleBy('userId', val_fraction, 101)
    
    #exceptAll to remove train data
    test = test_user.exceptAll(train1)
    val = val_user.exceptAll(train2)
    train = train1.union(train2)
    
    #write the data as parquet
    #train.write.parquet(name+'train.parquet')
    #test.write.parquet(name+'test.parquet')
    #val.write.parquet(name+'val.parquet')
    train.write.csv('train_'+name+'.csv', header=True)
    test.write.csv('test_'+name+'.csv', header=True)
    val.write.csv('val_'+name+'.csv', header=True)

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('split').getOrCreate()

    # Get file_path for dataset to analyze
    file_path = sys.argv[1]
    name = sys.argv[2]

    main(spark, file_path, name)