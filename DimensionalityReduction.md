## DimensionalityReduction
 <details>
 <summary>1️⃣Curse of dimensionality</summary>
   <br>
  
 **📍Curse of dimentionality**
 
  -  Harder to visualize or see structure of 
  -  Hard to search in high dimension (more runtime)
  -  Need more data to get a good estimation of the data
 
 **📍To combat the curse of dimensionality**
 
  -  **🌱Feature selection**: Keep only "Good" features
     - Drop features having missing values
     - Drop low variace column (a feature that is a constant)
     - Drop the feature by using forward (increase one by one feature) and backward (decrease one by one feature) elimination 

     > **Pro:** Useful when the user wants to know which feature matters

     > **Con:** Hard to select good features automatically

   **NOTE of Feature Selection** <br>

   0. Ask domain expertise which feature matters
   1. Use in Hackathon level (time limit days-a week), don't recommend to use in        other cases
   2. Proper methods for feature selection
       - Choose algorithms that handles high dimension well and do selection as a       by product ex Regression with L1 regularization (dan't same as L1 loss),          Tree-based classifiers (random forest, XGBoost)
       - Generic Algorithm: Optimization method, has objective and decision             variable that we want to change (Natural Selection), this algorithm also          can use for tuning hyperparameters in a neural network, and tuning                augmentation algorithms
   
  -  **🌱Feature transformation (Feature extraction)**: Transform the original features into a smaller set of features, New features come from the combination of old features (Greedy algorithm)

             F(x1,x2,...,x10) --> (y1, y2)

     > **Pro:** more powerful

     > **Con:** harder to interpret the model

Before doing feature transformation, recommend to revise **Linear Algebra**, which has following importantant topics:
        

<li> 1. Matrix as a sequence of column vectors</li>
        
  <img width="400" src="https://github.com/RadchaneepornC/DataScience-MachineLearning/blob/main/Images/LinearAlgebra_matrix1.png" style="display: block; margin-left: auto; margin-right: auto;">

  <img width="400" src="https://github.com/RadchaneepornC/DataScience-MachineLearning/blob/main/Images/LinearAlgebra_matrix2.png" style="display: block; margin-left: auto; margin-right: auto;">

  Fig: Imagine matrix multiplication as a sequence of column vectors [source](https://www.youtube.com/results?search_query=Matrix+as+a+sequence+of+column+vectors)

  




  **📍Goals of dimentional reduction**
  -  For better machine learning models
  -  For data visualization
 
</details>

<details>
 <summary>2️⃣How to reduce dimension
</summary>

 


 
</details>

<details>
<summary>Topic
</summary>
 abcd
          <ul><details><summary>Subtopics</summary> </details></ul>
          <ul><details><summary>Subtopics</summary></details></ul>
 
</details>
