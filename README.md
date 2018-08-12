
# Feature Selection - simplistic greedy approach
As we've now seen, it's fairly easy to overfit a model and as such we may need to make decisions about what variables or factors to include in the model and which to leave out. A simplistic way to do this is to add features individually, one by one.

## 1. Split the data into a test and train set.


```python
import pandas as pd
import numpy as np
df = pd.read_csv('Swiss_Healthcare_Premium_Prediction.csv.gz', compression='gzip')

df = df.fillna(value=0)
X = df[df.columns[:-1]]
y = df['Premium']
print(len(df))
df.head()
```

    53617





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>ID</th>
      <th>CAT_Insurer</th>
      <th>CAT_Region_Num</th>
      <th>205d_V2_CUR</th>
      <th>205d_V3_PRC</th>
      <th>212d_V1_CUR</th>
      <th>212d_V2_PRC</th>
      <th>213d_V2_CUR</th>
      <th>213d_V3_PRC</th>
      <th>...</th>
      <th>KG_SPS_226d_V1_CUR</th>
      <th>KG_SPS_227d_V1_PRC</th>
      <th>KG_SPS_229d_V1_CUR</th>
      <th>KG_SX_226d_V1_CUR</th>
      <th>KG_SX_227d_V1_PRC</th>
      <th>KG_SX_229d_V1_CUR</th>
      <th>KG_TOT_226d_V1_CUR</th>
      <th>KG_TOT_227d_V1_PRC</th>
      <th>KG_TOT_229d_V1_CUR</th>
      <th>Premium</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.326051</td>
      <td>0.215957</td>
      <td>0.322181</td>
      <td>0.234051</td>
      <td>0.377469</td>
      <td>0.213745</td>
      <td>...</td>
      <td>0.275441</td>
      <td>0.373868</td>
      <td>0.285167</td>
      <td>0.149472</td>
      <td>0.610291</td>
      <td>0.131174</td>
      <td>0.326051</td>
      <td>0.215957</td>
      <td>0.359147</td>
      <td>0.409432</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.000019</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.326051</td>
      <td>0.215957</td>
      <td>0.322181</td>
      <td>0.234051</td>
      <td>0.377469</td>
      <td>0.213745</td>
      <td>...</td>
      <td>0.275441</td>
      <td>0.373868</td>
      <td>0.285167</td>
      <td>0.149472</td>
      <td>0.610291</td>
      <td>0.131174</td>
      <td>0.326051</td>
      <td>0.215957</td>
      <td>0.359147</td>
      <td>0.394941</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.000037</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.326051</td>
      <td>0.215957</td>
      <td>0.322181</td>
      <td>0.234051</td>
      <td>0.377469</td>
      <td>0.213745</td>
      <td>...</td>
      <td>0.275441</td>
      <td>0.373868</td>
      <td>0.285167</td>
      <td>0.149472</td>
      <td>0.610291</td>
      <td>0.131174</td>
      <td>0.326051</td>
      <td>0.215957</td>
      <td>0.359147</td>
      <td>0.358463</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.000056</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.326051</td>
      <td>0.215957</td>
      <td>0.322181</td>
      <td>0.234051</td>
      <td>0.377469</td>
      <td>0.213745</td>
      <td>...</td>
      <td>0.275441</td>
      <td>0.373868</td>
      <td>0.285167</td>
      <td>0.149472</td>
      <td>0.610291</td>
      <td>0.131174</td>
      <td>0.326051</td>
      <td>0.215957</td>
      <td>0.359147</td>
      <td>0.321986</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.000075</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.326051</td>
      <td>0.215957</td>
      <td>0.322181</td>
      <td>0.234051</td>
      <td>0.377469</td>
      <td>0.213745</td>
      <td>...</td>
      <td>0.275441</td>
      <td>0.373868</td>
      <td>0.285167</td>
      <td>0.149472</td>
      <td>0.610291</td>
      <td>0.131174</td>
      <td>0.326051</td>
      <td>0.215957</td>
      <td>0.359147</td>
      <td>0.285634</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 196 columns</p>
</div>




```python
#Your code here
```

## 2. Find the [single] best feature to train a regression model on
Loop through all of the X features and train an unpenalized LinearRegression model using each of those single features. Find the feature that produces the lowest Mean squared test error.


```python
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
```


```python
#Your code here
from sklearn.linear_model import LinearRegression
#print('The single best predictor was: {}'.format(feat))
```

## 3. Generalize #2
Write a function that takes in a desired number of features and returns a model using the top n features (according to test set error). Be sure to do this iteratively. In other words, rather then simply taking the top n features based on how well each performs individually, first find the best feature and train a model, then loop back through all of the remaining features and select that which produces the best results in combination with the best feature already selected. Continue on finding the best third feature in combination with the previous 2 features, etc. This process will continue until you reach the desired number of features (or there are no features left).


```python
#Your code here
```

# Plotting Learning Curves
Iterate from 2 to 20 feature variables. Use your greedy classifier defined above to generate a linear regression model with successively more and more features incorporated into the model. Then plot the train and test errors as a function of the number of variables incorporated into each of these models.


```python
import matplotlib.pyplot as plt
%matplotlib inline

for i in range(2,21):
    #print('On iteration: {}'.format(i-1))
    #Train Greedy Classifier Model with this many features
    
    #Your code here
    
    #Calculate Training Mean Squared Error
    #Your code here
    
    #Calculate Test Mean Squared Error
    #Your code here
    #Plot Results
    #Your code here
    pass
#Add Legend and Descriptive Title/Axis Labels
#Your code here
```
