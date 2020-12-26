Non-parametric Regression Tree
---

All objective functions based on absolute value (square, absolute, Huber... ) have implicit parametric distribution assumption in disguise (Gaussian, Laplace...), thus fail badly when assumption not satisfied.

To have a robust (outliers-free) regression tree, we should use objective function only depends on ***relative value***.




Background
---

Let's say we have a decision stump on the *dth* dimension that splits data points into reds(xi=1) and blues(xi=0).

![](icon/pic1.png)

To find the ***best split***, first define a objective function depends only on relative values of data points.

### Non-parametric approach

[Kendall's Tau](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient), also known as ***bubble sort distance***. Defined as follow: 

![](https://latex.codecogs.com/svg.latex?\tau=\frac{2}{n(n-1)}\sum_{i%3Cj}sgn(x_i-x_j)sgn(y_i-y_j))

When i, j have same color, ![](https://latex.codecogs.com/svg.latex?sgn(x_i-x_j)) goes to zero, thus it's equivalent to:

![](https://latex.codecogs.com/svg.latex?\frac{2}{n(n-1)}\sum_{i\in%20Red;j\in%20Blue}sgn(y_i-y_j))

![](https://latex.codecogs.com/svg.latex?=\frac{2|Red||Blue|}{n(n-1)}\frac{1}{|Red||Blue|}\[N(red%3Eblue)-N(blue%3Ered)\])

![](https://latex.codecogs.com/svg.latex?=P(red,blue%20pairs)\times\[P(red%3Eblue)-P(blue%3Ered)\])

![](https://latex.codecogs.com/svg.latex?\simeq%20Gini%20impurity\times%20diff(red,blue))

There is a trade-off between balance & difference.

In practice, we can use other impurities (entropy...) & differences (log odds...) as long as 
![](https://latex.codecogs.com/svg.latex?diff(red,blue)=-diff(blue,red)).

**Note that:**

![](https://latex.codecogs.com/svg.latex?P(red,blue%20pairs)=\frac{n-1}{n}%202P_{Red}P_{Blue}\simeq%201-(P_{Red}^2+P_{Blue}^2))

*([Gini impurity](https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity))*

![](https://latex.codecogs.com/svg.latex?P(red>blue)+P(red=blue)+P(red<blue)=1)

*(conditional probability, draw one blue & one red without replacement)*



Problem Statement
---

![](https://latex.codecogs.com/svg.latex?s^*=arg\max_{s\in%20Stump}\tau^2(I(X,s),Y))

where ![](https://latex.codecogs.com/svg.latex?\tau)
is ***kendall rank correlation coefficient*** shown before.

and ![](https://latex.codecogs.com/svg.latex?I(X,s)=I(x_i>s))

### Compare with SSE


![](https://latex.codecogs.com/svg.latex?s^*=arg\min_{s\in%20Stump}SSE)

![](https://latex.codecogs.com/svg.latex?=arg\max_{s\in%20Stump}r^2(I(X,s),Y))

where *r* is ***pearson correlation coefficient***.



Algorithm
---

Calculate following for every possible stump,

![](https://latex.codecogs.com/svg.latex?\sum_{i\in%20Red;j\in%20Blue}sgn(y_i-y_j))

1. Initialize:

    Set all points to blue , `tau` = 0, `maximum` = 0  
    Sort samples by *`Xd`* as a queue

2. get next item `i` from queue, turn it red. 
3. Calculate tau, record stump if |`tau`| > `maximum`

    ![](https://latex.codecogs.com/svg.latex?\tau:=\tau+\sum_{b\in%20Blue}{sgn(y_i-y_b)}-\sum_{r\in%20Red}{sgn(y_r-y_i)})

    ![](https://latex.codecogs.com/svg.latex?=\tau+\sum_{j}sgn(y_i-y_j))

4. repeat **2.** until all points are red

In short:

![](https://latex.codecogs.com/svg.latex?\tau_i=cumsum_i\big(\sum_{j}sgn(y_i-y_j)\big))

### Dynamic Programing

With precompute:

+ ![](https://latex.codecogs.com/svg.latex?A=sgn(y-y^\top)\,%20A_{ij}=sgn(y_i-y_j))

+ sort every dimension

Time complexity ![](https://latex.codecogs.com/svg.latex?O(n^2+dnlogn))

Then for every depth:

+ calculate cummulate sum for each leaf

Time complexity ![](https://latex.codecogs.com/svg.latex?O(dn))



Further
---

Just like what elastic net does, we can consider both abosulte & relative value by weighted sum.

Ex: Objective function = ![](https://latex.codecogs.com/svg.latex?\alpha%20r^{2}+(1-\alpha)\tau^2)
