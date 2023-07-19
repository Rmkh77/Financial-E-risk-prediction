# Financial-E-risk-prediction
# E-Siging Loan Approval

E-signing loan approval is an application of the classification technique. As the process of loan approval is a time taking task which reduces the trust of the clients and the other hand the loan cannot be approved without verification too. 
Customer and business loan processing done manually and on paper leads to exceptions or errors that are expensive and time-consuming to fix, which results in a bad customer experience and downstream legal and compliance problems
To solve this problem the E-signing loan approval method was introduced. This takes less time for approval. Electronic verification enables the loan process to fully digitize, in addition to the other advantages of technology. It has other benefits too, costs associated with document handling and unintentional errors can be decreased, reduces future risks by doing careful verification, enhance client satisfaction by getting rid of delays and hassles and boost the count of the clients.

In this model, six different algorithms have been used to find out the best algorithm that best fits the model. 
KNN:
Knn is one of the most commonly used algorithms in machine learning projects. Knn is a supervised learning technique that is used when there is the use of Euclidean distance measure. According to Ali et al (2019), KNN considers the k nearest neighbors of the new data and the majority class label of these data points will be the new class label of the new data.
It is a non-parametric model thus it doesn’t take any assumptions about the underlying data. It is a lazy learner so it doesn’t learn at the time of training rather it evaluates when the new data comes to the existing training data. It can be more efficient if the training data is more.

Decision tree:
A Decision Tree is the most widely used hierarchy algorithm which has a tree-like structure. It consists of three nodes namely the root node, the intermediate node, and the last leaf node. Where the root node is the feature that has a high influence on the target column, after selecting this root node the dataset is divided according, and the root of these parts is obtained using some measure. This measure may be the Gini index or information gain. 
Decision trees do have a limitation which is observed when there is a small change in the dataset which leads to a huge change in a tree structure (Priyanka and Kumar 2020). Not only this but also it makes things more complex.

Random Forest:
Random forest is a collection of decision trees that are used to improve the accuracy of the model. The random forest is a supervised learning technique which is an idea of hybrid model making. Hence the random forest can be made up of two or more decision trees. 
One decision tree has the potential to produce inaccurate forecasts, but the random forest, which is a group of decision trees based on their voting-making predictions, can increase accuracy. The outcome of the random forest can be interpreted as whatever category has received the most tree voting (Mazumdar, et al 2021). The random forest uses all of the outcomes from the decision tree to predict outcomes. Since more decision trees are used to produce exact answers, random forest requires less training than other models. 
Overfitting is a drawback for random forests, has a random forest is a collection of trees that may lead to overfitting. For that reason, we must select a limited number of decision trees in the case of a random forest.


Gradient boosting:
Gradient Boosting Machine (GBM) is one of the most popular forward-learning ensemble methods in machine learning. It is a powerful technique for building predictive models for regression and classification tasks (Saleem et al 2021). GBM helps us to get a predictive model in the form of an ensemble of weak prediction models such as decision trees. Whenever a decision tree performs as a weak learner then the resulting algorithm is called gradient-boosted trees. It enables us to combine the predictions from various learner models and build a final predictive model having the correct prediction. 
They are so many advantages of using gradient boosting but even it possesses some drawbacks that are overfitting as well as overemphasizing the outliers. Gradient boosting algorithm continuously focuses to minimize errors and requires multiple trees hence, it is computationally expensive.


Naïve Bayes:
The Naïve Bayes algorithm is a supervised learning algorithm, which is based the on Bayes theorem and used for solving classification problems. It is mainly used in text classification that includes a high-dimensional training dataset. Naïve Bayes Classifier is one of the simple and most effective Classification algorithms which helps in building fast machine learning models that can make quick predictions. It is a probabilistic classifier, which means it predicts based on the probability of an object. Some popular examples of the Naïve Bayes Algorithm are spam filtration, Sentimental analysis, and classifying articles.
The naïve Bayes assumes that the occurrence of a certain feature is independent of the occurrence of other features. Hence each feature individually contributes to identifying that it is a target column without depending on each other (Sachdev et al 2019). And it is called Bayes because it depends on the principle of Bayes' Theorem. Which states that the occurrence of Event B such that Event A already occurred is nothing but conditional probability.


Neural network:
The term "Artificial Neural Network" is derived from Biological neural networks that develop the structure of a human brain. Similar to the human brain which has neurons interconnected to one another, artificial neural networks also have neurons that are interconnected to one another in various layers of the networks. These neurons are known as nodes. The structure of an artificial neural network consists of three layers 
Input Layer: As the name suggests, it accepts inputs in several different formats provided by the programmer
Hidden Layer: The hidden layer presents in-between input and output layers. It performs all the calculations to find hidden features and patterns.
Output Layer: The input goes through a series of transformations using the hidden layer, which finally results in output that is displayed using this layer (Pimpalkar 2022).
The artificial neural network takes input and computes the weighted sum of the inputs and includes a bias. This computation is represented in the form of a transfer function.


  'Decision Tree Classification' :   55.4%
  'Random Forest Classifier'     :   62%
  'Naive Bayes Classification'   :   56.9%
  'Gradient Boosting Classifier' :   62.2%
  'K Neighbors Classifier'       :   56.3%
  'MLP Classifierr'              :   49.5%

Among these algorithms, Gradient Boosting Classifier shows highest accuracy hence it is considered for next phases
