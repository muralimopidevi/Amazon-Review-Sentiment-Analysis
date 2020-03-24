# Databricks notebook source
# MAGIC %md ## Amazon Review Cell Phone and Accessories

# COMMAND ----------

# MAGIC %md ## Dataset Info , Check the below link.
# MAGIC 
# MAGIC selected dataset is --> Cell Phone and Accessories.
# MAGIC https://nijianmo.github.io/amazon/index.html

# COMMAND ----------

# MAGIC %md ## Databricks File System(DBFS) Connection.

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/Cell_Phones_and_Accessories_5.json"
file_type = "json"

# Json options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for json files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

#diaplaying dataframe 'df'
display(df)

# COMMAND ----------

# MAGIC %md ## Printing the Schema of a Dataframe(df)

# COMMAND ----------

# Displaying datatypes in a table. 
df.printSchema()

# COMMAND ----------

# MAGIC %md ## Importing the Libraries

# COMMAND ----------

# Import the libraries 
from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import re 
import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %md ## Creating a Temporary view

# COMMAND ----------

# Creating a view or table for a dataframe df.
temp_table_name = "temptable"
df.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

# MAGIC %md ## Using SQL to select all columns in temp view(table) as 'temptable'

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from temptable

# COMMAND ----------

# MAGIC %md ## Top 10 users who Rated Most 

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(reviewerID), reviewerID from temptable Group By reviewerID ORDER BY COUNT(reviewerID) DESC LIMIT 10

# COMMAND ----------

# MAGIC %md ## count of in overall

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(overall), overall from temptable Group By overall ORDER BY COUNT(overall) DESC

# COMMAND ----------

# MAGIC %md ## Reviewed as 'Good' with count of overall(ratings) 

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT count(overall), overall FROM temptable WHERE reviewText LIKE '%good%' GROUP BY overall ORDER BY overall

# COMMAND ----------

# MAGIC %md ## Reviewed as 'Good' with count of overall(ratings) 

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT count(overall), overall FROM temptable WHERE summary LIKE '%good%' GROUP BY overall ORDER BY overall

# COMMAND ----------

# MAGIC %md ## Reviewed as 'bad' with count of overall(ratings) 

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT count(overall), overall FROM temptable WHERE reviewText LIKE '%bad%' GROUP BY overall ORDER BY overall

# COMMAND ----------

# MAGIC %md ## Reviewed as 'bad' with count of overall(ratings) 

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT count(overall), overall FROM temptable WHERE summary LIKE '%bad%' GROUP BY overall ORDER BY overall

# COMMAND ----------

from pyspark.ml.feature import RegexTokenizer

tokenizer = (RegexTokenizer()
            .setInputCol("reviewText")
            .setOutputCol("tokens")
            .setPattern("\\W+"))

tokenizedDF = tokenizer.transform(df)
display(tokenizedDF.limit(5))

# COMMAND ----------

from pyspark.ml.feature import StopWordsRemover

remover = (StopWordsRemover()
          .setInputCol("tokens")
          .setOutputCol("stopWordFree"))

removedStopWordsDF = remover.transform(tokenizedDF)
display(removedStopWordsDF.limit(5))

# COMMAND ----------

from pyspark.ml.feature import CountVectorizer

counts = (CountVectorizer()
          .setInputCol("stopWordFree")
          .setOutputCol("features")
          .setVocabSize(1000))

countModel = counts.fit(removedStopWordsDF)

# COMMAND ----------

from pyspark.ml.feature import Binarizer

binarizer = Binarizer()  \
  .setInputCol("overall") \
  .setOutputCol("label") \
  .setThreshold(3.5)

print("Binarizer output with Threshold = %f" % binarizer.getThreshold())

# COMMAND ----------

(testDF, trainingDF) = df.select("overall","reviewText").na.drop().randomSplit((0.20, 0.80), seed=123)
testDF.cache()
trainingDF.cache()

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

lr = LogisticRegression()

p = Pipeline().setStages([tokenizer, remover, counts, binarizer, lr])
model = p.fit(trainingDF)
model.stages[-1].summary.areaUnderROC

result = model.transform(testDF)


# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator()
print ("AUC: %(result)s" % {"result": evaluator.evaluate(result)})

# COMMAND ----------


