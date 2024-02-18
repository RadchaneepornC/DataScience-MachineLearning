## DimensionalityReduction
 <details>
 <summary>1️⃣Curse of dimensionality</summary>
   <br>
  
 **📍Curse of dimentionality**
 
  -  Harder to visualize or see structure of 
  -  Hard to search in high dimension (more runtime)
  -  Need more data to get a good estimation of the data
 
 **📍To combat the curse of dimensionality**
 
 - **🌱Feature selection**: Keep only "Good" features
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

Before doing feature transformation, recommend to revise **Linear Algebra**, which has following importantant topics in the below topics


  -  **🌱Important Linear Algebra Concepts**:


**<li> 1. Matrix as a sequence of column vectors</li>**
        
  <img width="500" src="https://github.com/RadchaneepornC/DataScience-MachineLearning/blob/main/Images/LinearAlgebra_matrix1.png" style="display: block; margin-left: auto; margin-right: auto;">

  <img width="500" src="https://github.com/RadchaneepornC/DataScience-MachineLearning/blob/main/Images/LinearAlgebra_matrix2.png" style="display: block; margin-left: auto; margin-right: auto;">

  Fig: visualise matrix multiplication as a sequence of column vectors [source](https://www.youtube.com/watch?v=kYB8IZa5AuE&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=5)

  **<li> 2. View Eigendecomposition(ED) and Singular Value Decomposition (SVD) as rotations and stretches</li>**

  **<li> 3.Change basis from x,y coordinates to be on u by using PROJECTION</li>**
since we will use the concept of projection to project of each feature to the matrix and 
then maximize the variance after projection using 

                                 argmaxVar(wTx)


subject to the constraint that w is a unit vector. This maximization ensures that the chosen principal component (direction) captures the most significant variance present in the data




   


  
  **<li> 4.Covariance Matrix</li>**
          1. symmetric: real eigen values, eigen vectors are matually orthogonal 
          2. positive semi-definite(Convex function): semi because sometimes the variance can be zero, eigen              values are nonnegative
          3. positive definite: eigen values are positive --> garantee invertible

There are two aspects which has the same meaning: 

**Aspect I)** given a set of features as Random variables (RVs) 
**Aspect II)** see each data points as vector and go to cross product, minute mean, and  average
Covariance maxtrix in term of the vector view
- column of matrix stands for each vector of data point
- row of matrix stands for each feature
  




  **📍Goals of dimentional reduction**
  -  For better machine learning models
  -  For data visualization
 
</details>

<details>
 <summary>2️⃣How to reduce dimension</summary>
 <br>  
 
 **<li>📍Principle Component Analysis: PCA(unsupervised)</li>** solve Eigenvector ordered by Eigen value
             
  - goal: reduce dimension, but remain information
  - min dimensions = N train data 
       
   but not suit for classification problem, for this, use LCA instead

PCA does indeed transform a set of possibly correlated variables into a smaller number of uncorrelated variables (the principal components), the goal is to remove the least important principle components (the least of the variance one in the dataset) 

**Ex of PCA**

$$ 
X =
\begin{bmatrix}
1 & 2 \\
3 & 4 \\
5 & 6
\end{bmatrix}
$$





Step of PCA are following belows:
I) create principal components which are guaranteed to be uncorrelated if the data is projected onto them correctly

II) select the first few principal components(represent in eigenvectors), those with the largest eigenvalues(retain the majority of the useful information: max variance), and disregard the others, this step maybe include sub-step about optimization

  - Lagrange multiplier take a responsibility of maximizing the variance of the projected data while subjecting to the constraint the projection vector(principal component) has unit length (This standardization allows us to focus solely on the direction in which unit vector points, rather than being influenced by its magnitude, in othe words, allow us to compare the variance captured by different vectors measured on the same scale)
 
   
 **<li>📍LCA(supervised)</li>** got normalized Eigenvector
    
   - goal for classification
   - min dimensions = C-1 


</details>

 
 <details>
 <summary>3️⃣Visualization</summary>
   <br>
</details>
 
</details>

<details>
<summary>Topic
</summary>
 abcd
          <ul><details><summary>Subtopics</summary> </details></ul>
          <ul><details><summary>Subtopics</summary></details></ul>
 
</details>
