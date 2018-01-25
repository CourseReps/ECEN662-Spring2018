Challenge 1

For predicting the new label, my first mind is to compare the new point with the nearest point. If the nearest point is found, the new point's label can be set as the same as the nearest point's label. 

However, it will take a long time to compare every new point with all training set. My friend reminds me of the k-nearest neighbors algorithm. It is a model which can find the k nearest points around the new point. I choose k=1 to reduce the program time. And then, as I say, the program will find the nearest point and set the label of the new point. At last, print these new points.

I do not use the method on the class because I think it is difficult to find the precise PDF or CDF of the training sets. I use this method because I remind the knowledge I learned in last semester and think I can use the definition of distance to set the new points' label.
