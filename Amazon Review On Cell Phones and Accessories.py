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

# File location and type.
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

# Displaying dataframe 'df'.
display(df)

# COMMAND ----------

# MAGIC %md ## Count of DataFrame.

# COMMAND ----------

# Counting number of rows in dataframe 'df'.
original_count = df.count()
original_count

# Printing Count
print("Total Rows = %d" % original_count)

# COMMAND ----------

# MAGIC %md ## Printing the Schema of a Dataframe(df)

# COMMAND ----------

# Displaying datatypes in a dataframe 'df'. 
df.printSchema()

# COMMAND ----------

# MAGIC %md ## Removing NA, NULL, NaN Values and Dropping Unwanted columns.

# COMMAND ----------

import pyspark.sql.functions as F

# Dropping Unwanted Columns and saving in new dataframe called dfmodel.
step1 = df.drop('asin','helpful','reviewTime','reviewerID','reviewerName','unixReviewTime')

# Selecting "?","NULL", "NA", "NaN" values.
step1 = [F.when(~F.col(x).isin("?","NULL", "NA", "NaN"), F.col(x)).alias(x)  for x in step1.columns] 

# Droping "?","NULL", "NA", "NaN" values.
dfmodel = df.select(*step1).dropna(how='any')

# COMMAND ----------

# MAGIC %md ## Count of Deleted Rows.

# COMMAND ----------

after_count = dfmodel.count()
tot_del = original_count - after_count

# Printing Count
print("Deleted Rows = %d" % tot_del)

# COMMAND ----------

# MAGIC %md ## Removing Unwanted symbols.

# COMMAND ----------

import re

# Creating a different types of varibales
REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

# Creating userdefined function.
def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    
    return reviews
  
  # Calling function.
    reviews_train_clean = preprocess_reviews(dfmodel)

# COMMAND ----------

# MAGIC %md ## Converting sentence into Tokens.

# COMMAND ----------

from pyspark.ml.feature import RegexTokenizer

# Tokenizer
tokenizer = (RegexTokenizer()
            .setInputCol("reviewText")
            .setOutputCol("tokens")
            .setPattern("\\W+"))

tokenizedDF = tokenizer.transform(dfmodel)

# Displaying Dataframe
display(tokenizedDF.limit(5))

# COMMAND ----------

# MAGIC %md ## Removing StopWords from Tokens

# COMMAND ----------

from pyspark.ml.feature import StopWordsRemover

#StopwordsRemover
remover = (StopWordsRemover()
          .setInputCol("tokens")
          .setOutputCol("stopWordFree"))

removedStopWordsDF = remover.transform(tokenizedDF)

# Displaying Dataframe
display(removedStopWordsDF.limit(5))

# COMMAND ----------

# MAGIC %md ## Converitng Stopwordfree words into Vector for applying machine learning models.

# COMMAND ----------

from pyspark.ml.feature import CountVectorizer

counts = (CountVectorizer()
          .setInputCol("stopWordFree")
          .setOutputCol("features")
          .setVocabSize(2000))

cModel = counts.fit(removedStopWordsDF)
countModel = cModel.transform(removedStopWordsDF)

# Displaying Dataframe
display(countModel.limit(5))

# COMMAND ----------

# MAGIC %md ## Creating a Userdefind function for reducting values of Independent Variable

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import *

# Clearly identifying the job categories
def overall(rating):
  if(rating == 1.0 or rating == 2.0 or rating == 3.0):
    return 0.0
  if(rating == 4.0 or rating == 5.0):
    return 1.0
  else:
    return(rating)
  
  #CALLING USER DEFINED FUNCTIONS
etype_udf = udf(overall,DoubleType())
datamodel = countModel.withColumn("label", etype_udf("overall"))

# Displaying Dataframe
display(datamodel.limit(5))

# COMMAND ----------

# MAGIC %md ## Selecting Columns for Applying Machine Learning Models
# MAGIC ## and Spliting dataset into train and test sets.

# COMMAND ----------

# selecting and spliting 
(trainDF, testDF) = datamodel.select("label","features").randomSplit((0.80, 0.20), seed=1234)

#caching the dataframe.
trainDF.cache()
testDF.cache()

# COMMAND ----------

# MAGIC %md ### MACHINE LEARNING MODELS APPLYING.

# COMMAND ----------

# MAGIC %md ## 1.1) Logistic Regression (Single Run)

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

# Creating  a varible for LogisticRegression()
lr = LogisticRegression()

# Fit the model to trainDf
lrModel = lr.fit(trainDF)

# Print the coefficients and intercept for Logistic Regression
print("Coefficients: " + str(lrModel.coefficients))
print("Intercept: " + str(lrModel.intercept))
lrModel.summary.accuracy

#testDf
result = lrModel.transform(testDF)
result.select("prediction", "label", "features").show(5)

# COMMAND ----------

# MAGIC %md ## ROC Curve:-

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

#Binary Classification.
evaluator = BinaryClassificationEvaluator()

#Dispaying ROC CURVE
display(lrModel, trainDF, "ROC")


# COMMAND ----------

# MAGIC %md ## Evaluator Result

# COMMAND ----------

#Printing  the values of AUC
print("ACC: %(result)s" % {"result": evaluator.evaluate(result)})

# COMMAND ----------

# MAGIC %md ## Displaying LogisticRegression Model and TraningModel.

# COMMAND ----------

display(lrModel, trainDF)

# COMMAND ----------

# MAGIC %md ## Binary Classification Evaluator

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Let's use the run-of-the-mill evaluator
evaluator = BinaryClassificationEvaluator(labelCol='label')

# We have only two choices: area under ROC and PR curves :-(
auroc = evaluator.evaluate(result, {evaluator.metricName: "areaUnderROC"})
auprc = evaluator.evaluate(result, {evaluator.metricName: "areaUnderPR"})
print("Area under ROC Curve: {:.4f}".format(auroc))
print("Area under PR Curve: {:.4f}".format(auprc))

# COMMAND ----------

# MAGIC %md ## Confusion Matrix

# COMMAND ----------

from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.types import FloatType

#important: need to cast to float type, and order by prediction, else it won't work
preds_and_labels = result.select(['prediction','label']).withColumn('label', F.col('label').cast(FloatType())).orderBy('prediction')

#select only prediction and label columns
preds_and_labels = preds_and_labels.select(['prediction','label'])
metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))

#Printing Confusion Matrix.
print(metrics.confusionMatrix().toArray())

# COMMAND ----------

# MAGIC %md ## 1.2) Logistic Regression (Multiple  Run)

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator

#Estimator
logr = LogisticRegression(featuresCol='features', labelCol='label')

#Hyper-parameter tuning using Grid Search
param_grid = ParamGridBuilder().\
      addGrid(logr.regParam, [0, 0.1, 0.2, 0.6, 1]).\
      addGrid(logr.elasticNetParam, [0, 0.1, 0.2, 0.6, 1]).\
      addGrid(logr.maxIter, [5,10,20,60,100]).\
      build()

#Evaluator
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")

#Cross Validation
cv = CrossValidator(estimator=logr, evaluator=evaluator, estimatorParamMaps=param_grid, numFolds=3)
cv_model = cv.fit(trainDF)  

#selecting Columns
print("------------Showing Columns------------")
show_columns = ['features', 'label', 'prediction', 'rawPrediction', 'probability']
pred_training_cv = cv_model.transform(trainDF)
pred_training_cv.select(show_columns).show(5, truncate=False)

# Prediction on Testing
print("------------PREDICTION ON TESTING------------")
pred_test_cv = cv_model.transform(testDF)
pred_test_cv.select(show_columns).show(5, truncate=False)

# Prediction on Testing
print("------------PRINTING COEFFICIENTS------------")
print('Intercept: ' + str(cv_model.bestModel.intercept) + "\n" 'coefficients: ' + str(cv_model.bestModel.coefficients))

print('Logistic Regression', "\n",'The best RegParam is: ', cv_model.bestModel._java_obj.getRegParam(), "\n",'The best ElasticNetParam is:', cv_model.bestModel._java_obj.getElasticNetParam(), "\n",'The best Iteration is:',cv_model.bestModel._java_obj.getMaxIter() , "\n", 'Area under ROC is:', cv_model.bestModel.summary.areaUnderROC)

print("------------PRINTING AVEERAGE METRICS------------")
cv_model.avgMetrics

# COMMAND ----------

# MAGIC %md ## 2.1) Support Vector Machine(Single Run)

# COMMAND ----------

from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Load training data
lsvc = LinearSVC(maxIter=10, regParam=0.1)

# Fit the model
lsvcModel = lsvc.fit(trainDF)

# Print the coefficients and intercept for linear SVC
print("Coefficients: " + str(lsvcModel.coefficients))
print("Intercept: " + str(lsvcModel.intercept))

lsvcresult = lsvcModel.transform(testDF)
lsvcresult.select("prediction","label","features").show(10)

#Compute accuracy of test
from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator()
print("evaluation: %(result)s" % {"result": evaluator.evaluate(lsvcresult)})

# Let's use the run-of-the-mill evaluator
svmevaluator = BinaryClassificationEvaluator()

# We have only two choices: area under ROC and PR curves :-(
svmauroc = svmevaluator.evaluate(lsvcresult, {svmevaluator.metricName: "areaUnderROC"})
svmauprc = svmevaluator.evaluate(lsvcresult, {svmevaluator.metricName: "areaUnderPR"})
print("Area under ROC Curve: {:.4f}".format(svmauroc))
print("Area under PR Curve: {:.4f}".format(svmauprc))

# COMMAND ----------

# MAGIC %md ## 2.2) Support Vector Machine(Multiple Run)

# COMMAND ----------

from pyspark.ml.classification import LinearSVC
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator

#Estimator
lsvm = LinearSVC(featuresCol='features', labelCol='label')

#GRID VECTOR
param_grid_svm = ParamGridBuilder().\
      addGrid(lsvm.regParam, [0, 0.1, 0.2, 0.5, 1]).\
      addGrid(lsvm.maxIter, [5,10,20,50,100]).\
      build()

#Evaluator
svmevaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")

#Cross Validation
cv_svm = CrossValidator(estimator=lsvm, evaluator=svmevaluator, estimatorParamMaps=param_grid_svm, numFolds=3)
cv_svm_model = cv_svm.fit(trainDF)

#selecting Columns
print("------------Showing Columns------------")
show_columns = ['features', 'label', 'prediction', 'rawPrediction']
pred_training_svm = cv_svm_model.transform(trainDF)
pred_training_svm.select(show_columns).show(5, truncate=False)

# Prediction on Testing
print("------------PREDICTION ON TESTING------------")
pred_test_svm = cv_svm_model.transform(testDF)
pred_test_svm.select(show_columns).show(5, truncate=False)

print('Support Vector Machine', "\n",'The best RegParam is: ', cv_svm_model.bestModel._java_obj.getRegParam(),  "\n",'The best Iteration is:',cv_svm_model.bestModel._java_obj.getMaxIter() , "\n", 'Area under ROC is:', svmevaluator.evaluate(pred_test_svm, {svmevaluator.metricName: "areaUnderROC"}))

print("------------PRINTING AVEERAGE METRICS------------")
cv_svm_model.avgMetrics

# COMMAND ----------

# MAGIC %md ## 3.1) Naive Bayes(Single Run)

# COMMAND ----------

from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# create the trainer and set its parameters
nb = NaiveBayes(smoothing=1.0, modelType="multinomial")

# train the model
nbmodel = nb.fit(trainDF)

# select example rows to display.
nbresult = nbmodel.transform(testDF)
nbresult.select("prediction","label","features").show(10)

# compute accuracy on the test set
nbevaluator = BinaryClassificationEvaluator()
accuracy = nbevaluator.evaluate(nbresult)
print("evaluations: %(nbresult)s" % {"nbresult": nbevaluator.evaluate(nbresult)})

# Let's use the run-of-the-mill evaluator
nbevaluator = BinaryClassificationEvaluator()

# We have only two choices: area under ROC and PR curves :-(
nbauroc = nbevaluator.evaluate(nbresult, {nbevaluator.metricName: "areaUnderROC"})
nbauprc = nbevaluator.evaluate(nbresult, {nbevaluator.metricName: "areaUnderPR"})
print("Area under ROC Curve: {:.4f}".format(nbauroc))
print("Area under PR Curve: {:.4f}".format(nbauprc))

# COMMAND ----------

# MAGIC %md ## 3.2) Naive Bayes(Multiple Run)

# COMMAND ----------

from pyspark.ml.classification import NaiveBayes
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator

#ESTIMATOR
nb = NaiveBayes(featuresCol='features', labelCol='label')

#GRID VECTOR
param_grid_nb = ParamGridBuilder().\
      addGrid(nb.smoothing, [0.0,1.0,2.0,4.0,6.0,8.0]).\
      addGrid(nb.modelType, ["multinomial", "bernoulli"]).\
      build()

#Evaluator
nbevaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")

#CROSS VALIDATION
cv_nb = CrossValidator(estimator = nb, evaluator = nbevaluator, estimatorParamMaps = param_grid_nb, numFolds=3)
cv_nb_model = cv_nb.fit(trainDF)  # fitiing data to my cross validation model

#selecting Columns
print("------------Showing Columns------------")
show_columns = ['features', 'label', 'prediction', 'rawPrediction', 'probability']
pred_training_nb = cv_nb_model.transform(trainDF)
pred_training_nb.select(show_columns).show(5, truncate=False)

# Prediction on Testing
print("------------PREDICTION ON TESTING------------")
pred_test_nb = cv_nb_model.transform(testDF)
pred_test_nb.select(show_columns).show(5, truncate=False)

print('Naive Bayes ',"\n",'The best Smoothening is: ', cv_nb_model.bestModel._java_obj.getSmoothing(), "\n",'The best model type is:', cv_nb_model.bestModel._java_obj.getModelType(), "\n", 'Area under ROC is:', nbevaluator.evaluate(pred_test_nb, {nbevaluator.metricName: "areaUnderROC"}))

print("------------PRINTING AVEERAGE METRICS------------")
cv_nb_model.avgMetrics

# COMMAND ----------

# MAGIC %md ## 4.1) Random Forest(Single Run)

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Train a RandomForest model.
rf = RandomForestClassifier( numTrees=10)

# Train model.  This also runs the indexers.
rfmodel = rf.fit(trainDF)

# Make predictions.
rfresult = rfmodel.transform(testDF)

# Select example rows to display.
rfresult.select("prediction","label","features").show(10)

# Select (prediction, true label) and compute test error
rfevaluator = BinaryClassificationEvaluator()
print("evaluations: %(rfresult)s" % {"rfresult": rfevaluator.evaluate(rfresult)})

# We have only two choices: area under ROC and PR curves :-(
rfauroc = rfevaluator.evaluate(rfresult, {rfevaluator.metricName: "areaUnderROC"})
rfauprc = rfevaluator.evaluate(rfresult, {rfevaluator.metricName: "areaUnderPR"})
print("Area under ROC Curve: {:.4f}".format(rfauroc))
print("Area under PR Curve: {:.4f}".format(rfauprc))

# COMMAND ----------

# MAGIC %md ## 4.2) Random Forest(Single Run)

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator

#ESTIMATOR
rf = RandomForestClassifier(featuresCol='features', labelCol='label')

#GRID VECTOR
param_grid_rf = ParamGridBuilder().\
      addGrid(rf.impurity,['gini']).\
      addGrid(rf.maxDepth, [2, 3, 4]).\
      addGrid(rf.minInfoGain, [0.0, 0.1, 0.2, 0.3]).\
      addGrid(rf.numTrees,[20,40,60,80,100]).\
      build()

#Evaluator
rfevaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")

#CROSS VALIDATION
cv_rf = CrossValidator(estimator=rf, evaluator=rfevaluator, estimatorParamMaps=param_grid_rf, numFolds=3)
cv_rf_model = cv_rf.fit(trainDF)  # fitiing data to my cross validation model

#selecting Columns
print("------------Showing Columns------------")
show_columns = ['features', 'label', 'prediction', 'rawPrediction', 'probability']
pred_training_rf = cv_rf_model.transform(trainDF)
pred_training_rf.select(show_columns).show(5, truncate=False)

# Prediction on Testing
print("------------PREDICTION ON TESTING------------")
pred_test_rf = cv_rf_model.transform(testDF)
pred_test_rf.select(show_columns).show(5, truncate=False)

print('Random forest ',"\n",'The best Max Depth is: ', cv_rf_model.bestModel._java_obj.getMaxDepth(), "\n",'The best min Info gain is:', cv_rf_model.bestModel._java_obj.getMinInfoGain(), "\n", 'Area under ROC is:', rfevaluator.evaluate(pred_test_rf, {rfevaluator.metricName: "areaUnderROC"}))

print("------------PRINTING AVEERAGE METRICS------------")
cv_rf_model.avgMetrics

# COMMAND ----------

# MAGIC %md ## 5.1) Gradient Boost(Single Run)

# COMMAND ----------

from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Train a GBT model.
gb = GBTClassifier()

# Train model.  This also runs the indexers.
gbmodel = gb.fit(trainDF)

# Make predictions.
gbresult = gbmodel.transform(testDF)

# Select example rows to display.
gbresult.select("prediction","label","features").show(5)

# Select (prediction, true label) and compute test error
gbevaluator = BinaryClassificationEvaluator()

print("evaluations: %(gbresult)s" % {"gbresult": gbevaluator.evaluate(gbresult)})

# We have only two choices: area under ROC and PR curves :-(
gbauroc = gbevaluator.evaluate(gbresult, {gbevaluator.metricName: "areaUnderROC"})
gbauprc = gbevaluator.evaluate(gbresult, {gbevaluator.metricName: "areaUnderPR"})
print("Area under ROC Curve: {:.4f}".format(gbauroc))
print("Area under PR Curve: {:.4f}".format(gbauprc))


# COMMAND ----------

# MAGIC %md ## 5.2) Gradient Boost(Multiple Run)

# COMMAND ----------

from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator

# ESTIMATOR
gbt = GBTClassifier(featuresCol='features', labelCol='label')


#GRID VECTOR
param_grid_gbt = ParamGridBuilder().\
    addGrid(gbt.maxDepth, [2, 3, 4]).\
    addGrid(gbt.minInfoGain, [0.0, 0.1, 0.2]).\
    addGrid(gbt.stepSize, [0.02, 0.05, 0.1]).\
    addGrid(gb.maxIter,[20,40,60,80,100]).\
    build()

#Evaluator
gbtevaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")

#CROSS VALIDATION
cv_gbt = CrossValidator(estimator=gbt, evaluator=gbtevaluator, estimatorParamMaps=param_grid_gbt)
cv_gbt_model = cv_gbt.fit(trainDF)  # fitiing data to my cross validation model

show_columns = ['features', 'label', 'prediction', 'rawPrediction', 'probability']
pred_training_gbt = cv_gbt_model.transform(trainDF)
pred_training_gbt.select(show_columns).show(5, truncate=False)

pred_test_gbt = cv_gbt_model.transform(testDF)
pred_test_gbt.select(show_columns).show(5, truncate=False)


print('Gradient Boosting ',"\n",'The best Max Depth is: ', cv_gbt_model.bestModel._java_obj.getMaxDepth(), "\n",'The best min Info gain is:',cv_gbt_model.bestModel._java_obj.getMinInfoGain(), "\n", 'step size: ', cv_gbt_model.bestModel._java_obj.getStepSize(),"\n" ,'Area under ROC is:', gbtevaluator.evaluate(pred_test_gbt, {gbtevaluator.metricName: "areaUnderROC"}))

cv_gbt_model.avgMetrics

# COMMAND ----------

# MAGIC %md ## ALL MODELS ACCURACY ON TRAINING DATA SET

# COMMAND ----------

print('Models and their Performance',"\n")
print('Logistic Regression',evaluator.evaluate(pred_training_cv, {evaluator.metricName: "areaUnderROC"}))
print('Support Vector Machine',svmevaluator.evaluate(pred_training_svm, {svmevaluator.metricName: "areaUnderROC"}))
print('Naive Bayes', nbevaluator.evaluate(pred_training_nb, {nbevaluator.metricName: "areaUnderROC"}))
print('Random forest', rfevaluator.evaluate(pred_training_rf, {rfevaluator.metricName: "areaUnderROC"}))
print('Gradient Boost', gbtevaluator.evaluate(pred_training_gbt, {gbtevaluator.metricName: "areaUnderROC"}))

# COMMAND ----------

# MAGIC %md ## ALL MODEL PREDICTION ACCURACY ON TEST DATA SET

# COMMAND ----------

print('Models and their Performance',"\n")
print('Logistic Regression',evaluator.evaluate(pred_test_cv, {evaluator.metricName: "areaUnderROC"}))
print('Support Vector Machine',svmevaluator.evaluate(pred_test_svm, {svmevaluator.metricName: "areaUnderROC"}))
print('Naive Bayes', nbevaluator.evaluate(pred_test_nb, {nbevaluator.metricName: "areaUnderROC"}))
print('Random forest', rfevaluator.evaluate(pred_test_rf, {rfevaluator.metricName: "areaUnderROC"}))
print('Gradient Boost', gbtevaluator.evaluate(pred_test_gbt, {gbtevaluator.metricName: "areaUnderROC"}))

# COMMAND ----------

# MAGIC %md ## ALL MODELS ROC V/S PR

# COMMAND ----------

print('Models and their Performance',"\n")
print('Logistic Regression: ROC: ',evaluator.evaluate(pred_training_cv, {evaluator.metricName: "areaUnderROC"}), ', PR: ',evaluator.evaluate(pred_training_cv, {evaluator.metricName: "areaUnderPR"}))
print('Support Vector Machine',svmevaluator.evaluate(pred_training_svm, {svmevaluator.metricName: "areaUnderROC"}), ', PR: ',svmevaluator.evaluate(pred_training_svm, {svmevaluator.metricName: "areaUnderPR"}))
print('Naive Bayes', nbevaluator.evaluate(pred_training_nb, {nbevaluator.metricName: "areaUnderROC"}),', PR: ' , nbevaluator.evaluate(pred_training_nb, {nbevaluator.metricName: "areaUnderPR"}))
print('Random forest', rfevaluator.evaluate(pred_training_rf, {rfevaluator.metricName: "areaUnderROC"}), ', PR: ', rfevaluator.evaluate(pred_training_rf, {rfevaluator.metricName: "areaUnderPR"}))
print('Gradient Boost', gbtevaluator.evaluate(pred_training_gbt, {gbtevaluator.metricName: "areaUnderROC"}),', PR: ' , gbtevaluator.evaluate(pred_training_gbt ,{gbtevaluator.metricName: "areaUnderPR"}))
