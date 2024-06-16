# blood-transfusion-prediction-with-machine-learning
blood transfusion prediction with supervised machine learning

## Description
This project involves creating a machine learning model to predict whether a user donated blood in March 2007 or not, using the Julia programming language. The dataset used is from the Blood Transfusion Service Center Data Set.

In the following sections, we will develop the problem to be solved more specifically considering our database and its specific characteristics.
A certain number of patterns are classified into a certain number of predetermined outputs.

## PROBLEM DESCRIPTION
In this section, we will describe in detail the database of the problem we have decided to solve.

Our group has decided to choose a database downloaded from the internet. In the following link, you can access it.
https://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center
This database collects data taken from the Blood Transfusion Service Center in the city of Hsin-Chu in Taiwan, which corresponds to a classification problem
This database is composed of 748 instances; that is, 748 patterns and 5 attributes. They do not present "missing-values," and it corresponds to a classification task. Other important characteristics of this database are that its dataset is multivariate, that is, it presents different values for its attributes. Later we will indicate which ones. The area in which this database is classified is business, and the donation date is 2008-10-03.
The source of this database corresponds to Professor I-Cheng Yeh, from the Department of Information Management at Chung-Hua University, Taiwan. It collects data from a blood transfusion service bus from the Blood Transfusion Service Center in the city of Hsin-Chu, which collects donated blood approximately every 3 months. For the construction of this database, we selected 748 donors at random from the donor database. Each of these 748 donor data includes:
R (Recency - months since the last donation)
F (Frequency - total number of donations)
M (Monetary - total blood donated in c.c.)
T (Time - months since the first donation)
and a binary variable representing whether blood was donated in March 2007 (1
means blood was donated; 0 means no blood was donated).
These values ​​are called attributes, which as previously indicated, are 5.
These attributes contain the name of the variable, the unit of measurement, and a brief description.

Next, we will carry out a more detailed analysis of each attribute since, for the resolution of the proposed problem, we consider all the attributes to be of great relevance.
We show a part of our database (first 5 patterns) to get an idea of ​​how the patterns and attributes are distributed in it.
Recency Frequency Monetary Time If blood wasdonated or not


2 50 12500 98 1
0 13 3250 28 1
1 16 4000 35 1
2 20 5000 45 1
1 24 6000 77 0

## Attributes
The attribute recency indicates how many months have passed since the individual's last blood donation. This attribute is quantitative where the unit of measurement is the number of months. We consider it relevant because it provides us with information about the time elapsed since the individual last donated.
The frequency attribute indicates how many times the individual has donated blood. This attribute is quantitative, and the unit of measurement is the number of times. We consider it important because the higher this attribute is, the more blood donated, and therefore, it provides more information to our database.
The monetary attribute indicates the amount of blood donated by the individual. This attribute is quantitative, and the unit of measurement is c.c (cubic centimeters). We consider it important because the higher this quantity, the higher the frequency attribute, as previously indicated, and therefore, these two attributes must be clearly related.
The time attribute indicates the number of months elapsed since the first donation until the creation of this database. This attribute is qualitative, and the unit of measurement is the number of months. We consider it important to obtain information about the time elapsed since the first time each individual donated, and we will relate it to the frequency and, in turn, to the c.c of donated blood, since, logically, the longer the time since the first donation, the higher the number of frequency and c.c of blood donated, but as we will show later, this does not have to be the case.
The attribute if blood was donated or not indicates if blood donation occurred in March 2007. This attribute is boolean, and its unit of measurement is 1 if this donation occurred or 0 if it did not. This attribute is the most relevant of all since our problem is based on the classification of the database according to this attribute.

## Scatter plots and conclusions
Finally, we will show some scatter plots on these attributes.
This first scatter plot relates the frequency of donation to the amount of blood (in c.c) donated.
As can be seen, this relationship is proportional since the higher the frequency, the greater the amount of blood (in c.c) donated.
There is a clear relationship between these two attributes.
Another relevant scatter plot is one that relates the frequency of donation to the number of months elapsed since the first donation.
This should also be a proportional relationship, but as mentioned earlier, this may not be the case. An individual who has a very high value in the time variable means that more months have passed since their first donation than other individuals, but this does not mean that they have donated more times.
This fact is clearly observed in the plot. There are time values, for example, 30 months since their first donation, whose frequency is 22 times donated, while there are values of 43 times since their first donation who have only donated 10 times.
Another relevant diagram will be between the frequency attribute of the number of donations versus the boolean attribute if this individual donated this time or not.
It is observed that both individuals who usually donate many times (high frequency attribute) and individuals with the low-frequency attribute have donated in March 2007.
A higher number of individuals who have donated is observed compared to those who have not.
The last diagram we will show relates the amount of blood (in c.c) donated by each individual versus whether the donation was made in March 2007 or not.
As observed in the first scatter plot, the frequency/blood amount (in c.c) relationship is proportional, which leads us to deduce that this diagram will be very similar to the one shown above.
It is observed that both individuals with a high monetary attribute and individuals with this attribute low have donated in March 2007.
A higher number of individuals who have donated is observed compared to those who have not.





