# Chapter 1 : The Machine Learning landscape

* Purpose Of the Chapter :
Introduce What Exactly The machine learning is, the buzzword and jargon
* Note from Author :
This chapter introduces a lot of fundamental concepts (and jargon) that every data scientist should know by heart. It will be a high-level overview (it’s the only chapter without much code), all rather simple, but you should make sure everything is crystal clear to you before continuing on to the rest of the book. So grab a coffee and let’s get
started!
* ##  <-- Indicate New title
## What is Machine Learning ?
 * Machine Learning is the science and art of programming computers so they can *learn from data* : input =  Data and Algorithm , Output = Rules
    * Example Of Machine Learning :
      * Spam filter emails
    * *Jargon* :
      Training Set : Observation used by the models / ML Algorithm
## Why Use Machine learning
  * Think how you write a spam filter using traditional programing techniques?
      * Notice That , There's a many possibilities :D , you can't figure every single possibilities ,if you can , i don't care btw.
    * Step if you doing Traditional Programming techniques
      * Take The data
      * FOR LOOP SECTION  
      * Study The problem
      * Code/Write The Rules
      * Evaluate The Rules / Code
        * Condition Here:
          * if(Rules() == "Near Perfect"):
            * launch()
          * else:
            * AnalyzeErrors()
            * Back to for Loop :D
    * Have you Thinking How many Rules you should WRITE ? LOL..
    * I'll make Sure that You would face the hardest part of maintain the code
  * So What Machine Learning solution ?
    * Automatically learn through the data
    * in contrast ,  Machine learning techniques can or can't Automatically notices the noise
    * Here's how it's work
      * Take The data
      * **FOR LOOP SECTION**
      * Study the problem
      * Feed the Train Machine learning Algorithm with the data you've studied
      * Evaluate The Result
        * Condition Here:
          * if(EvaluateResult() == "Satisfied"):
            * Launch()
          else :
            * AnalyzeErrors()
            * Back To For loop
      * What if There's A CHANGE ? simple
      * Take The Updated Data add it to previous data
      * TRAIN the ML Algorithm with modified data
      * Evaluate Solution
      * Launch() if you Satisfied , Ups.
  * Remember Machine learning can Accomplish problem that are too complex for traditional approaches .
    * Example
      * Speech Recognition
      * Image Recognition
  * Machine Learning Can Help Humans To
    * Take the Data
    * Study the problem
    * Train ML Algorithm
    * Inspect the solution
    * AHA, i get the idea ,why the output like this (get New IDEA from the data)
    * Iterate if needed ,nah you don't need it,only joking,obvio you need it.
  * Machine learning is Great for :
    * Problems for which existing solutions require a lot of fine-tuning or long lists of rules
    * Complex Problems
    * Fluctuating environments (Unstable environments such as your life)
    * Getting insights about complex problem
    * LARGE AMOUNTs OF DATA
# Types Of Machine Learning
## Supervised / Unsupervised Learning
  * Notice that : attribute is a unit within , while feature is unit + it's value
    * Machine learning can be classified as 4 categories :
      * Supervised Learning --> Have Label/target Values (Can be Regression / Classification Problem)
        * Regression output is Continous
        * Classification Output is .... You get the idea, for example Determine whether you gay or not , ups.
        * Model/Algorithm
          * KNN *(Regression and Classification)*
          * Logistic *Only Classification*
          * Linear *Only Regression*
          * Neural Network *(Regression and Classification)*
          * SVM *(Regression and Classification)*
          * Decision tree / Random Forest *(Regression and Classification)*
      * Unsupervised learning -->  Doesn't Have labels (Try to find the something hidden in data)
        * Clustering
          * K-Means
          * DBSCAN
          * Hierarchical Cluster Analysis
        * Anomaly Detection
          * One-class SVM
          * Isolation Forest
        * Visualizaion and Dimensionality Reduction
          * PCA
          * Kernel PCA
          * Locally Linear Embedding
          * t-Distributed Stochastic Neighbor Embedding
        * Association rule Learning
          * Apriori
          * Eclat
        * Example For Each method
            * **CLUSTERING**
              * Say you want to try detect groups of your website , you would like to know who your audience is .
            * **Visualization**
              * Visualize your Data in 2d/3d
            * **Dimensionality Reduction**
              * Simplify data without losing 'TOO' much information
              * Feature Extraction --> Merges Two /more highly correlated features
            * **Anomaly Detection**
              * Detect Unusual credit card Transaction
              * Automatically Removing outliers
              **Novelty Detection** --> Aims To detect new instances that look different from all instances
            * **Association rule learning**
              * Discover Intresting RELATION BETWEEN *ATTRIBUTES*
      * Semisupervised learning
        * In Real World Dataset is messy , and sometime not every observation have its own label,of course it's time consuming labeling each obs
          * Semisupervised Can help you deal with this thing.
          * For instances,imagine you doing classification for triangle and square , but for some observation the label are missing ,thus imagine it as dot(UNlabeled data) .and then you want to predict something Semisupervised Will determine the class tho.
          * You may Ask , Why ? Have you used Google Photo ? If you have ,then you should know the answer .
          * SemiSupervised is a combination of Supervised and Unsupervised Learning
        * Learn more on google.com , JAAJAJAJ ,
      * Reinforcement learning
        * Using an agent to observe the environments , select action and get reward , mostly used in game , such AI DOTA
        * Agent Learn by it self,tried to find the best strategy using "POLICY" .
## Batch and Online Learning
  Sometime we need to determine whether the system can learn incrementally from a stream of incoming data or not
  * Batch learning --> System **can't** learn incrementally in other words your machine learning model can't learn from a stream of incoming data
    * Why ? Because Batch Learning means you trained your Observation All at once ,this will take a lot bigger computation and resources , often people say it *Offline Learning* ,so in case you want to use this on production and need to learn from incoming data, you need to train ALL data {OLD And New}and replace the old machine learning model, Kinda Bad right ?
  * Online Learning {Incremental Learning}--> System **can** learn incrementally , Yep your model can learn from a stream of incoming data ?
    * You will train your machine learning model incrementally by feeding it with new data within a groups ,called *Mini Batches* , Cheap and Fast make it more reliable
    * Once you've updated your machine learning model then you can't discard the data,saves more Storage
    * Also this often used when you feeding Neural Network with A big data set,you should know "Batch Size" :D
    * *Remember*  Learning Rate is one of the important Hyperparameter of learning system "How fast They should adapt to changing data"
      * Big ? Rapidly adapt to new data , tend quickly forget the old DATA
      * Small ? Slowly learning but less sensitive to noise / outliers
    A Big Challenge with Online Learning is that if "BAD DATA" Fed into the system,well slowly its performance will gradually decline
    Sometime you need to use Anomaly detection Algorithm before fed upcoming data into the system

## Instances Based Versus Model-Based learning
###  Machine learning Models Purpose is able to GENERAlize on unseen data.
  * There are 2 main approaches to generalization :
    * Instance-based learning --> {"Learn the examples by heart" , Generalize New cases by using a similarity , KNN for example}
    * Model-based learning --> {"Build model to make Prediction" , Generalize New Cases by using model , {SVM,LinReg,LogReg,MLP,...}}
  * Jargon --> {Utility Function --> How Good your model is , Cost Function --> How Bad your model is, Model Selection --> Choosing the type of model and fully specified its Hyperparameter / architecture
             Training Model -->  Try To find Best Fit to data }
  *  **Code Section**
    [x] ["Running and training a Linear Model"](https://github.com/Xwyzworms/Hands-on-MachineLearningWithScikit/blob/master/The%20Fundamentals%20of%20Machine%20Learning/The%20Machine%20Learning%20Landscape/Code/RunningandtrainingaLinearModel.ipynb)        
* In Summary :
    * We Studied the DATA
    * Selected Model
    * Trained The Model (using parameter values that minimize cost func)
    * Applied Model to make Prediction (Hope Generalize Well on unseen data)
# Main Challenges of machine learning
  * Bad Algorithm
  * Bad Data
## BAD DATA  
### insufficient Quantity of Training Data
  * Means That Machine learning Algorithm need to have a lot of data in order to Accomplish problems even the simple one.
  * Author Suggest to read : The Unreasonable Effectiveness of data. *i'll Update the pdf soon*   
### Nonrepresentative Training Data
  * Our purpose is make the model generalize well on unseen data , right ? So you need data that *REPRESENTATIVE* so you can answer the question. this is important wheter you used instance-based model or model-based learning :D , because it can make misleading predicition :D .
  * Jargon --> {*Sampling noise{result of chance}* because Sample to small  and *Sampling bias* because the sampling method is flawed}
  * **POINT IS IN ORDER TO MAKE THE MACHINE LEARNING MODEL GENERALIZE WELL ONE OF THE KEY *REPRESENTATIVE DATA* ,Always check your data**
### Poor Quality of data
  * Obvio , ma corido . If your TRAINING DATA is full of errors,outliers,and noise,it will Automatically make system harder to detect the underlying pattern,that's why your system is less likely to perform well. So make sure you put more effort when cleaning Data :D.
### Irrelevant features
  * As i said before , need Representative data ,it mean you have Relevant features/variables/columns within the dataset.
  * Jargon --> {*Feature Engineering* --> Create New feature that improve the system.}
    * Steps For Feature Engineering :
      * *Feature Selection* --> Selecting Most relevant Features , how ? EDA
      * *Feature Extraction* --> Combining Existing features to produce more Useful one :D , *Dimensionality Reduction* Can help us.
      * *Create New Feature By gathering new data*  
## BAD ALGORITHM
### Overfitting The Training Data
   Jargon --> {*Overfitting* --> Performs well on the training data but not on unseen data , Not Generalize Well}
   * This Pict will easy to understand Overfitting
   * Overfitting happens when the model is too complex relative to the amount and noisiness of the training data :D
       * Sometime machine learning model can detect pattern,but careful if the pattern is in the noise itself,cause misleading information
   * Overfitting Solution
      * Simplify Model , use fewer parameters , {PARAMETER : It will changed inside training such as {W1,W2,wn} , Hyperparameter : It will not changed inside training such as {learning rate , Polynomial Degree } , often people use this interchangebly. }
      * Gather More Training DATA
      * Cleaning Your Data from NOISES (Outliers/error data/NA,ect)
   * Jargon --> {*Regularization* --> Make simple model and reduce OverFitting}
### Underfitting the Training data
   Jargon --> {*UnderFitting* --> When your model is too simple to learn the underlying pattern of the data,your training data score more likely bad..}
   * Underfitting Solution
    * Select a more powerful model , with more parameters
    * Feed better features to the learning Algorithm (Feature Engineering)
    * Reduce the constraints on the model (Reduce Regularization)
## Stepping Back {Summary from author i guess}:
  * Machine learning is about making machine get better at some task by learning from data .
  * Different types of ML systmes : Supervised / Unsupervised ,batch or Online,Instance Based or model-Based
  *  In an ML project you gather data in a training set, and you feed the training set to a learning algorithm. If the algorithm is model-based, it tunes some parameters to fit the model to the training set (i.e., to make good predictions on the training set itself), and then hopefully it will be able to make good predictions on new cases as well. If the algorithm is instance-based, it just learns the examples by heart and generalizes to new instances by using a similarity measure to compare them to the learned instances.
  * The system will not perform well if your training set is too small, or if the data is not representative, is noisy, or is polluted with irrelevant features (garbage in,garbage out). Lastly, your model needs to be neither too simple (in which case it will underfit) nor too complex (in which case it will overfit).

# Evaluate Model
### Testing and validating
  * The Only way You know that your model will generalize well on unseen data is to try it out on new cases.
   * Option : Split your data into training and test set , common Split(80% Training , 20% test , if the data is 10000000000 you know it use 99.5% for training ,and rest for testing :D,Based on how big your dataset is,okay ? ).
   * If your Training Error is high --> UnderFitting
   * If your Training error low but your Testing Error is high --> Overfitting
  * Jargon --> {*Generalization Error* / *OutOfSample Error* --> Error Rate on Unseen data / Test set}
    * Generalization Error will tell you how good your model performs on unseen data.
### Hyperparameter tuning and model Selection
  * Use Cross Validation and GridSearchCV ,Why ?
    * It Will Tell You How good your model is in different Test set with the best Hyperparameters.
### Data Mismatch  
    * You know mismatch right? Solution Below .
    * train-dev set is used when there is a risk of mismatch between the training data and the data used in the validation and test datasets (which should always be as close as possible to the data used once the model is in production).
####Example:
      * Say You want to build up car classifier app , you gathering data from website ,also from taken car pict by phone ,say you  take car pict from website 10000 pict while by phone is 1000 , your model performs well and not overfitting , but when you launched it , many complaints from user.
      * Solution is -->  hold out some of the training pictures (from the web) in another set(train-dev-Set) . That is , create 1 train-dev set that is part of training but not trained . If your model is performs well on this train-dev set & Training Set but not on validation set then probably DATA MISMATCH happen , you need to manipulate/gather more data or try to improve the training data to make it look more like the validation + test data.
## WOhoo End of the Chapter ":D" , I m sorry if there's many mistakes ,it's good enough if you guys review this README.md :D I'd love to hear it , Thanks For Reading .    
