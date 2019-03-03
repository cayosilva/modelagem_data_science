#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql.functions import *
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local").appName("titanic_survival").getOrCreate()

direc="/home/void/dataprojects/titanic_survival_prediction/data"
sdf = spark.read.csv(direc + "/train.csv", header=True, inferSchema=True)
sdf.printSchema()
sdf = sdf.drop("Name", "Ticket")
sdf.show(20)

def index_cat(df, feature_names):
    from pyspark.ml.feature import StringIndexer
    counter = 0
    for i in feature_names:
        counter+=1
        print("working on feature " + str(counter) + " of " + str(len(feature_names)))
        print("indexing " + str(i) + " feature...")
        df = df.fillna("blank", i)
        indexer = StringIndexer(inputCol=i, outputCol=str(i)+"_ix")
        indexer_model = indexer.fit(df)
        indexed = indexer_model.transform(df)
        indexed.select(i, str(i) + "_ix").show(5)
        df = indexed
    return df
        
sdf=sdf.withColumn("CabinCat", substring(col("Cabin"), 1, 1))
sdf.show(5)
sdf = sdf.withColumn("CabinNumber", regexp_extract(col('Cabin'), '\d+', 0).cast("int")) \
         .withColumn("Fare", col("Fare").cast("float"))
sdf.show(5)

features_to_be_indexed=["Sex", "Embarked", "CabinCat"]
sdf = index_cat(sdf, features_to_be_indexed)
sdf.show(10)
sdf.printSchema()

def hot_encoding_var(df, feature_names):
    from pyspark.ml.feature import OneHotEncoder
    counter = 0
    for i in feature_names:
        counter+=1
        print("working on feature " + str(counter) + " of " + str(len(feature_names)))
        print("one hot encoding " + str(i) + " feature...")
        encoder = OneHotEncoder(inputCol=i, outputCol= str(i) + "_cd")
        encoded = encoder.transform(df)
        df = encoded
    return df

features_to_be_encoded=["Sex_ix", "Embarked_ix", "CabinCat_ix"]
sdf = hot_encoding_var(sdf, features_to_be_encoded)
sdf.show(10)
sdf.printSchema()

from pyspark.ml.feature import VectorAssembler
features = ["Pclass", "Sex_ix_cd", "Age", "SibSp", "Parch", "Fare", "CabinCat_ix_cd", "CabinNumber", "Embarked_ix_cd"]
sdf = sdf.fillna(0)
assembler = VectorAssembler(inputCols=features, outputCol="features")
assembled = assembler.transform(sdf)

assembled.printSchema()
assembled.show(5)

train = assembled.filter(col("PassengerID") <= 891)
test = assembled.filter(col("PassengerID") > 891)

from pyspark.ml.classification import LogisticRegression
log_reg = LogisticRegression(featuresCol="features", labelCol="Survived")
log_reg_model = log_reg.fit(train)

log_reg_model.intercept
log_reg_model.coefficients

def plot_iterations(summary):
  plt.plot(summary.objectiveHistory)
  plt.title("Training Summary")
  plt.xlabel("Iteration")
  plt.ylabel("Objective Function")
  plt.show()
plot_iterations(log_reg_model.summary)

log_reg_model.summary.areaUnderROC

log_reg_model.summary.roc.show(5)

def plot_roc_curve(summary):
  roc_curve = summary.roc.toPandas()
  plt.plot(roc_curve["FPR"], roc_curve["FPR"], "k")
  plt.plot(roc_curve["FPR"], roc_curve["TPR"])
  plt.title("ROC Area: %s" % summary.areaUnderROC)
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.show()

plot_roc_curve(log_reg_model.summary)

test_summary = log_reg_model.evaluate(test)

type(test_summary)
test_summary.areaUnderROC
plot_roc_curve(test_summary)

test_with_prediction = log_reg_model.transform(test)
test_with_prediction.show(5)

from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction"
                                          , labelCol="Survived"
                                          , metricName="areaUnderROC")
print(evaluator.explainParams())
evaluator.evaluate(test_with_prediction)

evaluator.setMetricName("areaUnderPR").evaluate(test_with_prediction)

pdf=test_with_prediction
pdf.count()

pdf.printSchema()

pdf.groupBy("prediction").count().show()
pdf.groupBy("survived").count().show()

pdf = pdf.withColumn("gotcha", col("prediction")==col("Survived"))
pdf.groupBy("gotcha").count().show()

ex = pdf.select("PassengerId", col("prediction").cast("int").alias("Survived"))
ex.toPandas().to_csv("results.csv", encoding="utf-8", index=False)