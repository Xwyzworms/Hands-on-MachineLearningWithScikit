# This Is the Exercise Solution : i'll give you mine , :D if you want author answer ,get the book :D

## This My answer
1. How would you define Machine Learning?
  * Answer : Build System that can learn from data,getting better and better while we feeding the system with data.
2. Can you name four types of problems where it shines?
3. What is a labeled training set?
  * Answer : Target that we want to achieve,every observation need to have it.
4. What are the two most common supervised tasks?
  * Answer : Classification , Regression
5. Can you name four common unsupervised tasks?
  * Answer : Clustering , Dimensionality Reduction , Visualization , Association Rule learning
6. What type of Machine Learning algorithm would you use to allow a robot to
walk in various unknown terrains?
  * Answer : Reinforcement Learning,Remember Agents ?
7. What type of algorithm would you use to segment your customers into multiple
groups?
  * Answer : Clustering , if you dont know the groups .Classification , if you know the group
8. Would you frame the problem of spam detection as a supervised learning problem or an unsupervised learning problem?
  * Answer : Yep ,this is Supervised Learning
9. What is an online learning system?
  * Answer : Can learn incrementally , You feed your system with mini batch data.
10. What is out-of-core learning?
  * Answer : Handles if quantities of data cannot fit into main memory
11. What type of learning algorithm relies on a similarity measure to make predictions?
  * Answer : Instance based learning
12. What is the difference between a model parameter and a learning algorithm’s
hyperparameter?
13. What do model-based learning algorithms search for? What is the most common strategy they use to succeed? How do they make predictions?
 * Answer : Model that have optimal value for the model parameters (W0,W1,W3,..) ,that is, by reducing the cost function and give penalty if we use Regularization
14. Can you name four of the main challenges in Machine Learning?
  * Answer : Uninformative/irrelevant feature , Not Representative Data , Poor data quality , Underfitting , overfitting
15. If your model performs great on the training data but generalizes poorly to new instances, what is happening? Can you name three possible solutions?
  * Answer : Overfitting
  * Solution --> add more data , use Regularization , Add/remove feature , reducing noise in your data  
16. What is a test set, and why would you want to use it?
  * Answer : To Get The Generalization Error .That is, how well your model perform on unseen data ,*BEFORE* the model is launched.
17. What is the purpose of a validation set?
  * Answer : used to compare models and hyperparameter tuning
18. What is the train-dev set, when do you need it, and how do you use it?
  * Answer : Used when you have data Mismatch problem , If your model is performs well on this train-dev set & Training Set but not on validation set then probably DATA MISMATCH happen
19. What can go wrong if you tune hyperparameters using the test set?
* Answer : *Your model will perform worst on production* Cuz you overfit the test set and the Generalization Error will bi Certain :D
