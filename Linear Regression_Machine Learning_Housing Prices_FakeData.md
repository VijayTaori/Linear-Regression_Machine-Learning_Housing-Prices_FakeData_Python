

```python
# Applying linear regression model to fake housing data to predict the prices
```


```python
# Usual imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```


```python
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
df = pd.read_csv('USA_Housing.csv')
```


```python
# Let's checkout the info() method which provides information about shape, data type, null-values
df.info() # clearly the 'Prices' is the target column
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5000 entries, 0 to 4999
    Data columns (total 7 columns):
    Avg. Area Income                5000 non-null float64
    Avg. Area House Age             5000 non-null float64
    Avg. Area Number of Rooms       5000 non-null float64
    Avg. Area Number of Bedrooms    5000 non-null float64
    Area Population                 5000 non-null float64
    Price                           5000 non-null float64
    Address                         5000 non-null object
    dtypes: float64(6), object(1)
    memory usage: 273.5+ KB
    


```python
df.describe()
```




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
      <th>Avg. Area Income</th>
      <th>Avg. Area House Age</th>
      <th>Avg. Area Number of Rooms</th>
      <th>Avg. Area Number of Bedrooms</th>
      <th>Area Population</th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5000.000000</td>
      <td>5000.000000</td>
      <td>5000.000000</td>
      <td>5000.000000</td>
      <td>5000.000000</td>
      <td>5.000000e+03</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>68583.108984</td>
      <td>5.977222</td>
      <td>6.987792</td>
      <td>3.981330</td>
      <td>36163.516039</td>
      <td>1.232073e+06</td>
    </tr>
    <tr>
      <th>std</th>
      <td>10657.991214</td>
      <td>0.991456</td>
      <td>1.005833</td>
      <td>1.234137</td>
      <td>9925.650114</td>
      <td>3.531176e+05</td>
    </tr>
    <tr>
      <th>min</th>
      <td>17796.631190</td>
      <td>2.644304</td>
      <td>3.236194</td>
      <td>2.000000</td>
      <td>172.610686</td>
      <td>1.593866e+04</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>61480.562388</td>
      <td>5.322283</td>
      <td>6.299250</td>
      <td>3.140000</td>
      <td>29403.928702</td>
      <td>9.975771e+05</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>68804.286404</td>
      <td>5.970429</td>
      <td>7.002902</td>
      <td>4.050000</td>
      <td>36199.406689</td>
      <td>1.232669e+06</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>75783.338666</td>
      <td>6.650808</td>
      <td>7.665871</td>
      <td>4.490000</td>
      <td>42861.290769</td>
      <td>1.471210e+06</td>
    </tr>
    <tr>
      <th>max</th>
      <td>107701.748378</td>
      <td>9.519088</td>
      <td>10.759588</td>
      <td>6.500000</td>
      <td>69621.713378</td>
      <td>2.469066e+06</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.columns
```




    Index(['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
           'Avg. Area Number of Bedrooms', 'Area Population', 'Price', 'Address'],
          dtype='object')




```python
# Generally speaking column_names without a underscore is not my thing and I like to convert it to one with underscores
df.columns = df.columns.str.replace(' ','_')
df.columns
```




    Index(['Avg._Area_Income', 'Avg._Area_House_Age', 'Avg._Area_Number_of_Rooms',
           'Avg._Area_Number_of_Bedrooms', 'Area_Population', 'Price', 'Address'],
          dtype='object')




```python
df.head()
```




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
      <th>Avg._Area_Income</th>
      <th>Avg._Area_House_Age</th>
      <th>Avg._Area_Number_of_Rooms</th>
      <th>Avg._Area_Number_of_Bedrooms</th>
      <th>Area_Population</th>
      <th>Price</th>
      <th>Address</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>79545.458574</td>
      <td>5.682861</td>
      <td>7.009188</td>
      <td>4.09</td>
      <td>23086.800503</td>
      <td>1.059034e+06</td>
      <td>208 Michael Ferry Apt. 674\nLaurabury, NE 3701...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>79248.642455</td>
      <td>6.002900</td>
      <td>6.730821</td>
      <td>3.09</td>
      <td>40173.072174</td>
      <td>1.505891e+06</td>
      <td>188 Johnson Views Suite 079\nLake Kathleen, CA...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>61287.067179</td>
      <td>5.865890</td>
      <td>8.512727</td>
      <td>5.13</td>
      <td>36882.159400</td>
      <td>1.058988e+06</td>
      <td>9127 Elizabeth Stravenue\nDanieltown, WI 06482...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>63345.240046</td>
      <td>7.188236</td>
      <td>5.586729</td>
      <td>3.26</td>
      <td>34310.242831</td>
      <td>1.260617e+06</td>
      <td>USS Barnett\nFPO AP 44820</td>
    </tr>
    <tr>
      <th>4</th>
      <td>59982.197226</td>
      <td>5.040555</td>
      <td>7.839388</td>
      <td>4.23</td>
      <td>26354.109472</td>
      <td>6.309435e+05</td>
      <td>USNS Raymond\nFPO AE 09386</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.pairplot(df)
```




    <seaborn.axisgrid.PairGrid at 0x18e61eaf550>




![png](output_9_1.png)



```python
df.hist(figsize=(12,9))
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x0000018E63E44B38>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000018E64024F28>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x0000018E64B9A438>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000018E63E63C18>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x0000018E643998D0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x0000018E640F0E80>]],
          dtype=object)




![png](output_10_1.png)



```python
sns.distplot(df['Price'],bins=20)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x18e6493f278>




![png](output_11_1.png)



```python
df.corr()
```




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
      <th>Avg._Area_Income</th>
      <th>Avg._Area_House_Age</th>
      <th>Avg._Area_Number_of_Rooms</th>
      <th>Avg._Area_Number_of_Bedrooms</th>
      <th>Area_Population</th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Avg._Area_Income</th>
      <td>1.000000</td>
      <td>-0.002007</td>
      <td>-0.011032</td>
      <td>0.019788</td>
      <td>-0.016234</td>
      <td>0.639734</td>
    </tr>
    <tr>
      <th>Avg._Area_House_Age</th>
      <td>-0.002007</td>
      <td>1.000000</td>
      <td>-0.009428</td>
      <td>0.006149</td>
      <td>-0.018743</td>
      <td>0.452543</td>
    </tr>
    <tr>
      <th>Avg._Area_Number_of_Rooms</th>
      <td>-0.011032</td>
      <td>-0.009428</td>
      <td>1.000000</td>
      <td>0.462695</td>
      <td>0.002040</td>
      <td>0.335664</td>
    </tr>
    <tr>
      <th>Avg._Area_Number_of_Bedrooms</th>
      <td>0.019788</td>
      <td>0.006149</td>
      <td>0.462695</td>
      <td>1.000000</td>
      <td>-0.022168</td>
      <td>0.171071</td>
    </tr>
    <tr>
      <th>Area_Population</th>
      <td>-0.016234</td>
      <td>-0.018743</td>
      <td>0.002040</td>
      <td>-0.022168</td>
      <td>1.000000</td>
      <td>0.408556</td>
    </tr>
    <tr>
      <th>Price</th>
      <td>0.639734</td>
      <td>0.452543</td>
      <td>0.335664</td>
      <td>0.171071</td>
      <td>0.408556</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.heatmap(df.corr(), annot=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x18e64742390>




![png](output_13_1.png)



```python
type(df)#/df.corr())
```




    pandas.core.frame.DataFrame




```python
a = 'Dr. Vijay Taori'
type(a)
```




    str




```python
type(df/df.corr())
```




    pandas.core.frame.DataFrame




```python
type(df/a)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    ~\Anaconda3\lib\site-packages\pandas\core\ops.py in na_op(x, y)
       1504         try:
    -> 1505             result = expressions.evaluate(op, str_rep, x, y, **eval_kwargs)
       1506         except TypeError:
    

    ~\Anaconda3\lib\site-packages\pandas\core\computation\expressions.py in evaluate(op, op_str, a, b, use_numexpr, **eval_kwargs)
        207     if use_numexpr:
    --> 208         return _evaluate(op, op_str, a, b, **eval_kwargs)
        209     return _evaluate_standard(op, op_str, a, b)
    

    ~\Anaconda3\lib\site-packages\pandas\core\computation\expressions.py in _evaluate_numexpr(op, op_str, a, b, truediv, reversed, **eval_kwargs)
        122     if result is None:
    --> 123         result = _evaluate_standard(op, op_str, a, b)
        124 
    

    ~\Anaconda3\lib\site-packages\pandas\core\computation\expressions.py in _evaluate_standard(op, op_str, a, b, **eval_kwargs)
         67     with np.errstate(all='ignore'):
    ---> 68         return op(a, b)
         69 
    

    TypeError: ufunc 'true_divide' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''

    
    During handling of the above exception, another exception occurred:
    

    TypeError                                 Traceback (most recent call last)

    <ipython-input-30-f9acc74584cd> in <module>
    ----> 1 type(df/a)
    

    ~\Anaconda3\lib\site-packages\pandas\core\ops.py in f(self, other, axis, level, fill_value)
       2034 
       2035             assert np.ndim(other) == 0
    -> 2036             return self._combine_const(other, op)
       2037 
       2038     f.__name__ = op_name
    

    ~\Anaconda3\lib\site-packages\pandas\core\frame.py in _combine_const(self, other, func)
       5118     def _combine_const(self, other, func):
       5119         assert lib.is_scalar(other) or np.ndim(other) == 0
    -> 5120         return ops.dispatch_to_series(self, other, func)
       5121 
       5122     def combine(self, other, func, fill_value=None, overwrite=True):
    

    ~\Anaconda3\lib\site-packages\pandas\core\ops.py in dispatch_to_series(left, right, func, str_rep, axis)
       1155         raise NotImplementedError(right)
       1156 
    -> 1157     new_data = expressions.evaluate(column_op, str_rep, left, right)
       1158 
       1159     result = left._constructor(new_data, index=left.index, copy=False)
    

    ~\Anaconda3\lib\site-packages\pandas\core\computation\expressions.py in evaluate(op, op_str, a, b, use_numexpr, **eval_kwargs)
        206     use_numexpr = use_numexpr and _bool_arith_check(op_str, a, b)
        207     if use_numexpr:
    --> 208         return _evaluate(op, op_str, a, b, **eval_kwargs)
        209     return _evaluate_standard(op, op_str, a, b)
        210 
    

    ~\Anaconda3\lib\site-packages\pandas\core\computation\expressions.py in _evaluate_numexpr(op, op_str, a, b, truediv, reversed, **eval_kwargs)
        121 
        122     if result is None:
    --> 123         result = _evaluate_standard(op, op_str, a, b)
        124 
        125     return result
    

    ~\Anaconda3\lib\site-packages\pandas\core\computation\expressions.py in _evaluate_standard(op, op_str, a, b, **eval_kwargs)
         66         _store_test_result(False)
         67     with np.errstate(all='ignore'):
    ---> 68         return op(a, b)
         69 
         70 
    

    ~\Anaconda3\lib\site-packages\pandas\core\ops.py in column_op(a, b)
       1126         def column_op(a, b):
       1127             return {i: func(a.iloc[:, i], b)
    -> 1128                     for i in range(len(a.columns))}
       1129 
       1130     elif isinstance(right, ABCDataFrame):
    

    ~\Anaconda3\lib\site-packages\pandas\core\ops.py in <dictcomp>(.0)
       1126         def column_op(a, b):
       1127             return {i: func(a.iloc[:, i], b)
    -> 1128                     for i in range(len(a.columns))}
       1129 
       1130     elif isinstance(right, ABCDataFrame):
    

    ~\Anaconda3\lib\site-packages\pandas\core\ops.py in wrapper(left, right)
       1581             rvalues = rvalues.values
       1582 
    -> 1583         result = safe_na_op(lvalues, rvalues)
       1584         return construct_result(left, result,
       1585                                 index=left.index, name=res_name, dtype=None)
    

    ~\Anaconda3\lib\site-packages\pandas\core\ops.py in safe_na_op(lvalues, rvalues)
       1527         try:
       1528             with np.errstate(all='ignore'):
    -> 1529                 return na_op(lvalues, rvalues)
       1530         except Exception:
       1531             if is_object_dtype(lvalues):
    

    ~\Anaconda3\lib\site-packages\pandas\core\ops.py in na_op(x, y)
       1505             result = expressions.evaluate(op, str_rep, x, y, **eval_kwargs)
       1506         except TypeError:
    -> 1507             result = masked_arith_op(x, y, op)
       1508 
       1509         result = missing.fill_zeros(result, x, y, op_name, fill_zeros)
    

    ~\Anaconda3\lib\site-packages\pandas\core\ops.py in masked_arith_op(x, y, op)
       1024         if mask.any():
       1025             with np.errstate(all='ignore'):
    -> 1026                 result[mask] = op(xrav[mask], y)
       1027 
       1028     result, changed = maybe_upcast_putmask(result, ~mask, np.nan)
    

    TypeError: ufunc 'true_divide' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''



```python
sns.heatmap(df.corr(),annot=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x21d7d195470>




![png](output_18_1.png)



```python
# Copy the numerical column names from df.columns excluding the 'Price' column to make new data frame
# This data frame is the set of all the veriables that affect the price of the perticular house
# and Y as the target or the 'Price' dataframe
X = df[['Avg._Area_Income', 'Avg._Area_House_Age', 'Avg._Area_Number_of_Rooms',
       'Avg._Area_Number_of_Bedrooms', 'Area_Population']]
y = df['Price']
```


```python
# Let's check X and y (head())
print(X.head())
print('--------------')
print('Data Type of X is: ',type(X))
print('**************')
print(y.head())
print('--------------')
print('Data Type of y is: ',type(y))
```

       Avg._Area_Income  Avg._Area_House_Age  Avg._Area_Number_of_Rooms  \
    0      79545.458574             5.682861                   7.009188   
    1      79248.642455             6.002900                   6.730821   
    2      61287.067179             5.865890                   8.512727   
    3      63345.240046             7.188236                   5.586729   
    4      59982.197226             5.040555                   7.839388   
    
       Avg._Area_Number_of_Bedrooms  Area_Population  
    0                          4.09     23086.800503  
    1                          3.09     40173.072174  
    2                          5.13     36882.159400  
    3                          3.26     34310.242831  
    4                          4.23     26354.109472  
    --------------
    Data Type of X is:  <class 'pandas.core.frame.DataFrame'>
    **************
    0    1.059034e+06
    1    1.505891e+06
    2    1.058988e+06
    3    1.260617e+06
    4    6.309435e+05
    Name: Price, dtype: float64
    --------------
    Data Type of y is:  <class 'pandas.core.series.Series'>
    


```python
from sklearn.model_selection import train_test_split
```


```python
# unpacking train_test_split tuple
# This will split the dataset in two groups called training set and test set
# Because we have y which is our target, the model will work with training set and then we can confirm the accuracy on the test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
# Random state is chosen to 101 to get exactly same result what I got last time as last time also I had chosen 101
```


```python
# import the LinearRegression model
from sklearn.linear_model import LinearRegression
# Make an instance of this model or instantiate the model
lm = LinearRegression()
```


```python
# Let's fit the training set to the model
lm.fit(X_train, y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)




```python

```


```python
print(lm.intercept_)
```

    -2640159.796851911
    


```python
lm.coef_
```




    array([2.15282755e+01, 1.64883282e+05, 1.22368678e+05, 2.23380186e+03,
           1.51504200e+01])




```python
X_train.columns
```




    Index(['Avg._Area_Income', 'Avg._Area_House_Age', 'Avg._Area_Number_of_Rooms',
           'Avg._Area_Number_of_Bedrooms', 'Area_Population'],
          dtype='object')




```python
X.columns
```




    Index(['Avg._Area_Income', 'Avg._Area_House_Age', 'Avg._Area_Number_of_Rooms',
           'Avg._Area_Number_of_Bedrooms', 'Area_Population'],
          dtype='object')




```python
cdf = pd.DataFrame(lm.coef_,index=X_train.columns,columns=['Coeff'])
cdf.head()
```




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
      <th>Coeff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Avg._Area_Income</th>
      <td>21.528276</td>
    </tr>
    <tr>
      <th>Avg._Area_House_Age</th>
      <td>164883.282027</td>
    </tr>
    <tr>
      <th>Avg._Area_Number_of_Rooms</th>
      <td>122368.678027</td>
    </tr>
    <tr>
      <th>Avg._Area_Number_of_Bedrooms</th>
      <td>2233.801864</td>
    </tr>
    <tr>
      <th>Area_Population</th>
      <td>15.150420</td>
    </tr>
  </tbody>
</table>
</div>




```python
lm.fit(X_train,y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)




```python
X_train, X_test, y_train, y_test = train_test_split(df1, df2, test_size=0.2, random_state=4)
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
```


```python
lm1.fit(X_train,y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)




```python
# We need two important things from this model
# 1. intercept which you can get it from the attribute lm.intercept_
# 2. coefficient which you can get it from the attribute lm.coef_
print('Intercept is: ',lm.intercept_)
print('Set of coefficients is: ',lm.coef_)
```

    Intercept is:  -2640159.796851911
    Set of coefficients is:  [2.15282755e+01 1.64883282e+05 1.22368678e+05 2.23380186e+03
     1.51504200e+01]
    


```python
# Intercept shown above do not talk to us directly 
# So we use a better way of representing them
# Let's make a data frame out of it
# the columns names corresponding to the co-efficients can be index
# The coefficients can be the body of the DataFrame
# and column name can be coeff
Coeff_df = pd.DataFrame(lm.coef_, index=X.columns, columns=['Co-efficients'])
Coeff_df
```




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
      <th>Co-efficients</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Avg._Area_Income</th>
      <td>21.528276</td>
    </tr>
    <tr>
      <th>Avg._Area_House_Age</th>
      <td>164883.282027</td>
    </tr>
    <tr>
      <th>Avg._Area_Number_of_Rooms</th>
      <td>122368.678027</td>
    </tr>
    <tr>
      <th>Avg._Area_Number_of_Bedrooms</th>
      <td>2233.801864</td>
    </tr>
    <tr>
      <th>Area_Population</th>
      <td>15.150420</td>
    </tr>
  </tbody>
</table>
</div>




```python
import matplotlib.pyplot as plt
```


```python
# We still haven't passed the test set to this model 
# And we still haven#t compared its generation of y values and our already available y values
pred_values = lm.predict(X_test)
print(type(pred_values))
print('***************')
a = pred_values/y_test
print(a)
```

    <class 'numpy.ndarray'>
    ***************
    1718    1.007408
    2511    0.947930
    345     1.026779
    2521    0.916032
    54      1.052625
    2866    0.884542
    2371    0.928265
    2952    1.212477
    45      0.963788
    4653    0.933742
    891     1.008175
    3011    1.016302
    335     0.987783
    3050    0.887521
    3850    0.964823
    834     1.094442
    3188    0.849544
    4675    0.848376
    2564    1.136980
    1866    0.865799
    1492    1.263964
    3720    1.218885
    618     1.044775
    3489    0.998544
    2145    0.963020
    3200    0.987091
    4752    0.985603
    602     1.011727
    4665    1.016204
    79      1.035449
              ...   
    4668    0.923755
    3762    1.342201
    236     1.026918
    4897    0.948148
    1283    1.157236
    2443    1.282226
    3600    1.032961
    2138    1.067112
    254     1.174238
    3987    0.918753
    527     0.977258
    1362    0.988514
    4577    0.940858
    2642    0.984865
    4297    0.967352
    1114    0.971898
    1041    0.984857
    3666    1.055580
    4571    0.926021
    3698    0.955747
    412     0.986969
    3385    0.995617
    3057    0.896668
    3863    1.048679
    4165    1.000425
    1776    1.017135
    4269    0.959212
    1661    2.456270
    2410    1.015919
    2302    1.004457
    Name: Price, Length: 2000, dtype: float64
    


```python
a.plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x18e68b3f978>




![png](output_39_1.png)



```python
plt.scatter(y_test,pred_values,edgecolors='red')
```




    <matplotlib.collections.PathCollection at 0x18e682c60f0>




![png](output_40_1.png)



```python
f = (lambda x: x>320000)
above_320T = y_test.apply(f)
print(above_320T)
```

    1718     True
    2511     True
    345      True
    2521     True
    54       True
    2866     True
    2371     True
    2952     True
    45       True
    4653     True
    891      True
    3011     True
    335      True
    3050     True
    3850     True
    834      True
    3188     True
    4675     True
    2564     True
    1866     True
    1492     True
    3720     True
    618      True
    3489     True
    2145     True
    3200     True
    4752     True
    602      True
    4665     True
    79       True
            ...  
    4668     True
    3762     True
    236      True
    4897     True
    1283     True
    2443     True
    3600     True
    2138     True
    254      True
    3987     True
    527      True
    1362     True
    4577     True
    2642     True
    4297     True
    1114     True
    1041     True
    3666     True
    4571     True
    3698     True
    412      True
    3385     True
    3057     True
    3863     True
    4165     True
    1776     True
    4269     True
    1661    False
    2410     True
    2302     True
    Name: Price, Length: 2000, dtype: bool
    


```python

```


```python
sns.scatterplot(x=y_test,y=a,hue=above_320T)
# To me this is a great plot 
# This plot explains that the accuracy of the model is better for predicting houses with price above 320,000
```




    <matplotlib.axes._subplots.AxesSubplot at 0x18e6724c1d0>




![png](output_43_1.png)

