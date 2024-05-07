# blood-transfusion-prediction-with-machine-learning
blood transfusion prediction with supervised machine learning
INTRODUCTION
In the realization of this project, a machine learning model is created to solve problems that could not be solved with classical programming. These innovative techniques allow computers to "learn," meaning the models significantly improve based on their experience (training).
The problems this model can face are primarily two: classification problems and regression problems, but in this section, we will only address a classification problem.
What are classification problems?
Classification problems are those in which the model to be developed takes certain characteristics of an element, and from these, it is capable of classifying it into one or several known classes, giving us a correspondence between inputs and desired outputs. This is what we call Supervised Learning.
Although it should be noted that there are more branches of this matter related to artificial intelligence.

As we can see in the diagram, the characteristics of the element will be the inputs of the model, and the outputs to be predicted will be boolean type.
For these models to perform classification well, they must take a set of examples, patterns (also called instances), of which we know all the desired outputs.
They take a certain number of patterns, for each of these, they have a series of characteristics, and the value of the desired outputs is predicted.
A very basic example would be the following:

Therefore, we have two matrices, one with the patterns and characteristics, and another with the desired outputs.
If these matrices are passed as parameters to a TRAINING function, it will return a model with the input/output relationship of the instances, but this model is not perfect; there will always be a certain level of error.
Why is it important?
Classification learning is very useful in multiple disciplines. For example, in information retrieval. Classifying is the central part of information retrieval problems. Matching queries with documents along with their relevance is a data training task.
Examples of areas of information retrieval where it is used:

Automatic translation
Computational biology, protein structure predictions.
Also, for simpler and more common cases:
It can be used to process loan requests in solvent or risky situations
Differentiate incoming email messages as spam or important
Know if a person's face appears in a photograph or not,
These tasks can be performed by a human, but models make these tasks much simpler, faster, and if properly trained, effective.
Therefore, the advantages of solving a model are quite clear.
➢ Effectiveness: if the model is well trained, even if it has an error rate.
➢ Speed and fluency: The model will be much faster than human capacity, allowing significant time savings.
Having understood the conflict of the problem to be solved, the objectives of this project can be listed.
The main objective will be to train a model that classifies the patterns of our database in the best possible way.
Different stages will have to be developed:
➢ Take a dataset with a series of instances and characteristics.
➢ Create a model that allows us to classify the inputs (patterns) with certain
characteristics into desired outputs effectively.
➢ For this:
➢ Normalize this data using normalization functions.
➢ Train the model using functions to improve its effectiveness and accuracy in
classification.
In the following sections, we will develop the problem to be solved more specifically considering our database and its specific characteristics.
A certain number of patterns are classified into a certain number of predetermined outputs.
PROBLEM DESCRIPTION
In this section, we will describe in detail the database of the problem we have decided to solve.
It is important to note that this database must not have a very low number of patterns (<100) or a very high number (>1000). Another relevant data is that it cannot have blank values in some of the table's characteristics. Also, it must have problems related to classification or regression.
Our group has decided to choose a database downloaded from the internet. In the following link, you can access it.
https://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center
This database collects data taken from the Blood Transfusion Service Center in the city of Hsin-Chu in Taiwan, which corresponds to a classification problem
Below we attach an image that shows the main characteristics of our database.
As can be seen in the image, this database is composed of 748 instances; that is, 748 patterns and 5 attributes. They do not present "missing-values," and it corresponds to a classification task. Other important characteristics of this database are that its dataset is multivariate, that is, it presents different values for its attributes. Later we will indicate which ones. The area in which this database is classified is business, and the donation date is 2008-10-03.
The source of this database corresponds to Professor I-Cheng Yeh, from the Department of Information Management at Chung-Hua University, Taiwan. It collects data from a blood transfusion service bus from the Blood Transfusion Service Center in the city of Hsin-Chu, which collects donated blood approximately every 3 months. For the construction of this database, we selected 748 donors at random from the donor database. Each of these 748 donor data includes:
R (Recency - months since the last donation)
F (Frequency - total number of donations)
M (Monetary - total blood donated in c.c.)
T (Time - months since the first donation)
and a binary variable representing whether blood was donated in March 2007 (1
means blood was donated; 0 means no blood was donated).
These values ​​are called attributes, which as previously indicated, are 5.
These attributes contain the name of the variable, the unit of measurement, and a brief description.
In the following table, we show an example of training from our database by taking 500 random samples. These are the data that will be trained. The remaining elements (248) will be the test set.
Table 1: An example of descriptive statistics of the data.
Variable Data type Measure Description min max mean std
Recency quantitative Months Input 0.03 74.4 9.74 8.07
Frequency quantitative Times Input 1 50 5.51 5.84
Monetary quantitative c.c.
blood
Input 250 12500 1378.68 1459.83
Time quantitative Months Input 2.27 98.3 34.42 24.32
If donated
blood in
March 2007
binary 1=yes
0=no
Output 0 1 1 (24%) or
(76%)
Next, we will carry out a more detailed analysis of each attribute since, for the resolution of the proposed problem, we consider all the attributes to be of great relevance.
We show a part of our database (first 5 patterns) to get an idea of ​​how the patterns and attributes are distributed in it.
Recency Frequency Monetary Time If blood was
donated or not
2 50 12500 98 1
0 13 3250 28 1
1 16 4000 35 1
2 20 5000 45 1
1 24 6000 77 0
The attribute recency indicates how many months have passed since the individual's last blood donation. This attribute is quantitative where the unit of measurement is the number of months. We consider it relevant because it provides us with information about the time elapsed since the individual last donated.
The frequency attribute indicates how many times the individual has donated blood. This attribute is quantitative, and the unit of measurement is the number of times. We consider it important because the higher this attribute is, the more blood donated, and therefore, it provides more information to our database.
The monetary attribute indicates the amount of blood donated by the individual. This attribute is quantitative, and the unit of measurement is c.c (cubic centimeters). We consider it important because the higher this quantity, the higher the frequency attribute, as previously indicated, and therefore, these two attributes must be clearly related.
The time attribute indicates the number of months elapsed since the first donation until the creation of this database. This attribute is qualitative, and the unit of measurement is the number of months. We consider it important to obtain information about the time elapsed since the first time each individual donated, and we will relate it to the frequency and, in turn, to the c.c of donated blood, since, logically, the longer the time since the first donation, the higher the number of frequency and c.c of blood donated, but as we will show later, this does not have to be the case.
The attribute if blood was donated or not indicates if blood donation occurred in March 2007. This attribute is boolean, and its unit of measurement is 1 if this donation occurred or 0 if it did not. This attribute is the most relevant of all since our problem is based on the classification of the database according to this attribute.
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





