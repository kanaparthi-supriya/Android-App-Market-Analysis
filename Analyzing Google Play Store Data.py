# -*- coding: utf-8 -*-
Unveiling the Android App Market: Analyzing Google Play Store Data

Load Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
apps=pd.read_csv("/content/apps.csv")
reviews=pd.read_csv("/user_reviews.csv")

print(apps.head())

print(reviews.head())

"""Data Information"""

print(apps.info())

print("Apps Dataset Shape:", apps.shape)
print("Reviews Dataset Shape:", reviews.shape)

print(reviews.info())

"""Missing Values Check"""

print(apps.isnull().sum())

reviews.isnull().sum()

"""Remove Missing Values"""

apps=apps.dropna()
reviews=reviews.dropna()

"""Check Duplicate Records"""

apps.duplicated().sum()
reviews.duplicated().sum()

"""Remove Duplicates"""

apps=apps.drop_duplicates()

"""Statistical Summary"""

apps.describe()

print("Top App Sizes")
print(apps["Size"].value_counts().head(10))

"""Top Categories Analysis"""

print(apps["Category"].value_counts().head(5))

"""Category Visualization"""

plt.figure(figsize=(10,5))
apps["Category"].value_counts().head(10).plot(kind="bar")
plt.title("Top 10 Categories")
plt.show()

print("Observation:")
print("Family and Game categories have the highest number of apps.")

"""Rating Distribution"""

print("Average Rating:", apps["Rating"].mean())

plt.figure(figsize=(8,5))
sns.histplot(apps["Rating"])
plt.title("Rating Distribution")
plt.show()

print("Observation:")
print("Most app ratings are between 4 and 4.5.")

"""Free vs Paid Apps"""

apps["Type"].value_counts()
apps["Type"].value_counts().plot(kind="pie",autopct="%1.1f%%")
plt.title("Free vs Paid Apps")
plt.show()

print("Observation:")
print("Most applications in Google Play Store are free.")

"""Top Installed Apps"""

if apps['Installs'].dtype == 'object':
    apps['Installs'] = apps['Installs'].str.replace('+','')
    apps['Installs'] = apps['Installs'].str.replace(',','')
    apps['Installs'] = pd.to_numeric(apps['Installs'])
elif not pd.api.types.is_numeric_dtype(apps['Installs']):
    # This handles cases where Installs might be in an unexpected non-numeric, non-object format
    apps['Installs'] = pd.to_numeric(apps['Installs'], errors='coerce')

apps[['App','Installs']].sort_values(
    by='Installs',
    ascending=False
).head(10)

print("Observation:")
print("Top installed apps are the most popular apps among users.")

"""Rating vs Installs"""

plt.figure(figsize=(10,5))

sns.scatterplot(
    x="Installs",
    y="Rating",
    data=apps
)

plt.title("Rating vs Installs")
plt.show()

print("Observation:")
print("Apps with higher ratings generally receive more installs.")

"""Sentiment Analysis"""

reviews["Sentiment"].value_counts()

"""Sentiment Visualization"""

reviews["Sentiment"].value_counts().plot(kind="bar")
plt.title("Sentiment Analysis")
plt.show()

print("Observation:")
print("Positive reviews are higher than negative reviews.")

print(reviews[['Sentiment_Polarity', 'Sentiment_Subjectivity']].describe())
