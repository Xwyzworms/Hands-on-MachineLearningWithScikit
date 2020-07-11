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
      * FOR LOOP SECTION
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
    * Iterate if needed ,nah you don't need it.
  * Machine learning is Great for :
    * Problems for which existing solutions require a lot of fine-tuning or long lists of rules
    * Complex Problems
    * Fluctuating environments (Unstable environments such as your life)
    * Getting insights about complex problem
    * LARGE AMOUNTs OF DATA
## Types Of Machine Learning
  * Supervised / Unsupervised Learning
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
# CONTINUE READING THE BOOK LATER ON , PAGE 14  
