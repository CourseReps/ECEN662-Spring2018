# Challenge 2
___
## Thinking of the 2nd challenge
- Finding the distribution:  
I used [Weka](https://www.cs.waikato.ac.nz/ml/weka/): a machine learning and data mining tool to visualization every colomn of the features:  
This application shows me that all 8 colomns of the data have a similar distribution.
![Distribution]()
- Find the relation of each colomn  
For the fearture of each sample have different dimensions, some have 6 and some have 8. The very first thing I consider is to downsample
the feature to make the problem easy to compute and still have good accuracy.  
The following picture are the 2-dim combinations of each two of the feature[Y0,Y1,...,Y7]:  
![2-dim distribution]()  
We can find that each 2 of the features can have a pretty distinction between 0 and 1.
- Make assumptions:  
Knowing that each 2 dimension can have a great classification, I make the assumption that if I compress the feature space to 6 dimension,
the loss of the information should be limited. Then I only use the first 6 colomns to train the classifier.  
In this part, I choose SVM with rbf kernel designing the classifier.
- Result:  
Using SVM and the first 6 colomns, the test on the training set have a error rate of 0.686% the crossing test have an error rate of about 
7%, it is a great performance, so I choose this classifier.
