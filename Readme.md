

# Project: Investigate a Dataset (Movie Data Base!)

## Table of Contents
<ul>
<li><a href="#intro">Introduction</a></li>

<li><a href="#wrangling">Data Wrangling</a></li> 
<ul>
    <li><a href="#General Properties">General Properties</a></li>
  
   <li> <a href="#Data Cleaning">Data Cleaning</a></li>
    <ul>
    <li> <a href="#Dropping Columns">Dropping Columns</a></li>
    <li> <a href="#Splitting Values">Splitting Values</a></li> 
    <li> <a href="#notes about Null values">notes about Null values</a></li>
    </ul>
</ul>
<li><a href="#eda">Exploratory Data Analysis</a></li> 
<ul>
    <li><a href="#Proprities of data and box plot">Proprities of data and box plot</a></li>
    <li><a href="#notes about box plots and outliers">notes about box plots and outliers</a></li> 
    <li><a href="#Questions">Questions</a></li>
   </ul>
<li><a href="#limitations">limitations</a></li>
<li><a href="#conclusions">Conclusions</a></li>
</ul>

<a id='intro'></a>
# Introduction
I've chosen the TMDB data set becuase I'm intersted in movies and want to know some facts about the market


those are the questions:
- Question 1 : Which data  are correlated ?
- Question 2 (generes sorted with runtime!)
- Question 3 (most popular genre for every year!)
- Question 4 (what genres that producer would like to consider to get highest profit?)
- Question 5 (most popular genres of all time?)
- Question 6 (what are top 10 highest profit movies?)
- Question 7 (what are top 10  titles with longest time?)
- Question 8 (Directors that are most productive?)
- Question 9 (Top 10 directors that make movies with high profit?)
- Question 10 (Most hired Acotrs?)
- Question 11 (Most productive Companies?)
- Question 12 (Companies with highest profit?)
- Question 13 (What does change when movies seperated from TV shows?)

I used padnas n, numpy ,matplotlib


```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
%matplotlib inline


 
```

<a id='wrangling'></a>
# Data Wrangling


<a id='General Properties'></a>
### General Properties
showing null values and some box plots 


```python
#reading our data
data=pd.read_csv('tmdb-movies.csv')

data.isna().sum().sort_values(ascending =False)

     
```




    homepage                7930
    tagline                 2824
    keywords                1493
    production_companies    1030
    cast                      76
    director                  44
    genres                    23
    imdb_id                   10
    overview                   4
    popularity                 0
    budget                     0
    revenue                    0
    original_title             0
    revenue_adj                0
    budget_adj                 0
    runtime                    0
    release_date               0
    vote_count                 0
    vote_average               0
    release_year               0
    id                         0
    dtype: int64




```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10866 entries, 0 to 10865
    Data columns (total 21 columns):
    id                      10866 non-null int64
    imdb_id                 10856 non-null object
    popularity              10866 non-null float64
    budget                  10866 non-null int64
    revenue                 10866 non-null int64
    original_title          10866 non-null object
    cast                    10790 non-null object
    homepage                2936 non-null object
    director                10822 non-null object
    tagline                 8042 non-null object
    keywords                9373 non-null object
    overview                10862 non-null object
    runtime                 10866 non-null int64
    genres                  10843 non-null object
    production_companies    9836 non-null object
    release_date            10866 non-null object
    vote_count              10866 non-null int64
    vote_average            10866 non-null float64
    release_year            10866 non-null int64
    budget_adj              10866 non-null float64
    revenue_adj             10866 non-null float64
    dtypes: float64(4), int64(6), object(11)
    memory usage: 1.7+ MB
    


```python
 
print(data[data.budget==0].budget.count())
print(data[data.revenue==0].budget.count())
print(data[data.budget_adj==0].budget.count())
print(data[data.revenue_adj==0].budget.count())
#drop not needed columns
    
data.drop(labels=['budget','revenue','id','imdb_id','homepage','release_date'],inplace=True,axis=1)

```

    5696
    6016
    5696
    6016
    

<a id='Data Cleaning'></a>
# Data Cleaning 
<a id='Dropping Columns'></a>
### Dropping Columns
some columns can be dropped like:
- **budget**     (we use budget_adj as gives more accurate information based on inflation)               
- **revenue**    (we use revune_adj as gives more accurate information based on inflation)  
- **id**         (as its title is enough)
- **imdb_id**    (as its title is enough)
- **home_page**  (I think no need for it)
<a id='Splitting Values'></a>

### Splitting Values:

some columns uses '|' to combine many values ...I will seperate each value for seperate row in the following columns :
 - genres 
 - director
 - cast
 - production_companies
 
 note that I create N **rows** for each N values  sperated ***not columns*** to make calling ```groupby``` easier as I think sepreating into columns will make it harder 
 I can easily call groupby("cast") and do some operations to find names of most hired actors
 I think using https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.get_dummies.html will lead to harder approach when using ```groupby()``` that lead me building that function  ```split_value_in_column_to_rows``` to help my analyzing **nominal** data like Actor's names , companies' names and directors to know **most hired actors , most productive companies .......**
 
    for example: when movie with cast "actor1|actor2": 
    there will be 2 rows for the same movie with same values excpect for cast value in one it "actor1" and in other it is "actor2"
    that makes it easier to make groupby(genres) instead of spliting each value into sepreate columns that gives alot of columns





<a id='notes about Null values'></a>

## notes about Null values 
- there are too many missing homepages , taglines ,production companies:
    - we may get rid of tagline column as I think genres is enough for my questions
    - we may leave companies as it is ....it can be helpful later
    - for budget , budget_adj ,  revenue ,revenue_adj :
        - there is no null or nan values but there is too many values with zeros (we may consider zero as null in this case)




```python
#this part for counting maximum number of combined values for columns we want to split the values inside it
"""
    for example: when movie with genres "Action|commedy": 
    there will be 2 rows for the same movie with same values excpect for genres value in one it "action" and in other it is "comedy"
    that makes it easier to make groupby(genres) instead of spliting each value into sepreate columns that gives alot of columns
    """
d_size=data.dropna(subset=['genres','director','cast','production_companies']) 
"""
cast_size : maximum amount of cast names contained in one cell
genres_size
director_size
production_companies_size

"""

d_size.genres=d_size.genres.apply(lambda x :len(x.split('|')))
genres_size=int(d_size.genres.max())

d_size.cast=d_size.cast.apply(lambda x :len(x.split('|')))
cast_size=int(d_size.cast.max())

d_size.director=d_size.director.apply(lambda x :len(x.split('|')))
director_size=int(d_size.director.max())

d_size.production_companies=d_size.production_companies.apply(lambda x :len(x.split('|')))
production_companies_size=int(d_size.production_companies.max())

print((cast_size,genres_size,director_size,production_companies_size))

```

    (5, 5, 36, 5)
    

    C:\ProgramData\Anaconda3\lib\site-packages\pandas\core\generic.py:4405: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      self[name] = value
    


```python
def split_value_in_column_to_rows(d,col,max_size): 
    """
    for example: when movie with genres "Action|commedy": 
    there will be 2 rows for the same movie with same values excpect for genres value in one it "action" and in other it is "comedy"
    that makes it easier to make groupby(genres) instead of spliting each value into sepreate columns that gives alot of columns
    """
    #d_return : final result
    #dx : dummy data frame that I'll modify its column values
    
    d_return=pd.DataFrame()
    d=d.copy()
    d.dropna(subset=[col],inplace=True) #we won't be able to split those values if we have null values  
    for i in range(0, max_size):
        dx=d.copy()
        dx[col]=dx[col].apply(lambda x :x.split('|')[i]  if len(x.split('|'))>i else '' )
        d_return=d_return.append(dx)
        d_return=d_return[d_return[col]!='']
    return d_return
#using sepreate data frame for each intrested Column to maintain as many data as possible as I dropped null rows in each operation 
#to analyze genres 
d_genres=split_value_in_column_to_rows(data,'genres',genres_size)
#to analyze cast
d_cast=split_value_in_column_to_rows(data,'cast',cast_size)
#to analyze directors
d_director=split_value_in_column_to_rows(data,'director',director_size)
#to analyze companies
d_companies=split_value_in_column_to_rows(data,'production_companies',production_companies_size)#to analyze companies

```

<a id='eda'></a>
## Exploratory Data Analysis

<a id='Proprities of data and box plot'></a>
 ### Proprities of data and box plot


```python
data_described=data.describe()
data_described

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
      <th>popularity</th>
      <th>runtime</th>
      <th>vote_count</th>
      <th>vote_average</th>
      <th>release_year</th>
      <th>budget_adj</th>
      <th>revenue_adj</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>10866.000000</td>
      <td>10866.000000</td>
      <td>10866.000000</td>
      <td>10866.000000</td>
      <td>10866.000000</td>
      <td>1.086600e+04</td>
      <td>1.086600e+04</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.646441</td>
      <td>102.070863</td>
      <td>217.389748</td>
      <td>5.974922</td>
      <td>2001.322658</td>
      <td>1.755104e+07</td>
      <td>5.136436e+07</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.000185</td>
      <td>31.381405</td>
      <td>575.619058</td>
      <td>0.935142</td>
      <td>12.812941</td>
      <td>3.430616e+07</td>
      <td>1.446325e+08</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000065</td>
      <td>0.000000</td>
      <td>10.000000</td>
      <td>1.500000</td>
      <td>1960.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.207583</td>
      <td>90.000000</td>
      <td>17.000000</td>
      <td>5.400000</td>
      <td>1995.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.383856</td>
      <td>99.000000</td>
      <td>38.000000</td>
      <td>6.000000</td>
      <td>2006.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.713817</td>
      <td>111.000000</td>
      <td>145.750000</td>
      <td>6.600000</td>
      <td>2011.000000</td>
      <td>2.085325e+07</td>
      <td>3.369710e+07</td>
    </tr>
    <tr>
      <th>max</th>
      <td>32.985763</td>
      <td>900.000000</td>
      <td>9767.000000</td>
      <td>9.200000</td>
      <td>2015.000000</td>
      <td>4.250000e+08</td>
      <td>2.827124e+09</td>
    </tr>
  </tbody>
</table>
</div>




```python
'''
plot box plots to know about outliers  
'''
for col in data_described.columns:
    pd.DataFrame(data[col]).boxplot(figsize=(20,12))
    plt.show()
```


![png](output_14_0.png)



![png](output_14_1.png)



![png](output_14_2.png)



![png](output_14_3.png)



![png](output_14_4.png)



![png](output_14_5.png)



![png](output_14_6.png)



```python

```

<a id='notes about box plots and outliers'></a>
## notes about box plots and outliers: 
- there are many outliers in :
 
    -runtime(will be discussed in **Question 10** and  **Question 7** )
    
    -profit(i think it's movies that made it in box office)
    
    -budget(i think it's like Big franchises)
    
    -vote count (I think it has relation to increasing population every year)
    
    -popularity(I think it's for Big franchises or very famous movies)
 


```python

sn.heatmap(data.corr(), annot=True)
plt.show()


```


![png](output_17_0.png)


<a id='Questions'></a>
 
# Questions:

## Research Question 1 : Which data  are correlated ?

we see that most of data has no correlation but the following are correlated:
- profit and budget 
- profit and popularity 
- profit and vote_count
- vote_count and popularity 



```python
def plot_scatter_with_line(x,y,xlabel,ylabel,title):
    plt.plot(x, y, 'o')
    m, b = np.polyfit(x, y, 1)#linear regression
    plt.plot(x, m*x + b)
    plt.ylabel(ylabel);
    plt.xlabel(xlabel);
    plt.title(title);
    plt.show()
```


```python
y=data.revenue_adj
x=data.budget_adj
plot_scatter_with_line(x,y,'budget','profit','budgetVSprofit')
(x,y)=(data.popularity,data.revenue_adj)
plot_scatter_with_line(x,y,'popularity','popularity','revenue_adj VS popularity')
(x,y)=(data.vote_count,data.popularity)
plot_scatter_with_line(x,y,'vote_count','popularity','vote_count VS popularity')

```


![png](output_21_0.png)



![png](output_21_1.png)



![png](output_21_2.png)


### Research Question 2 (generes sorted with runtime!)

which genre that has most running time ?
Answer: History , War ,western are the top 3


```python
d_genres.groupby('genres').mean()['runtime'].sort_values().plot.bar(color=(.2,.2,1,0.6),title='genres according to mean runtime');

 
```


![png](output_23_0.png)


### Research Question 3 (most popular genre for every year!)



```python
top_gen=pd.DataFrame()#for storing our results
for year in d_genres.release_year.unique():#pick every year and find the mean value of popularity in each genre
    top_gen=top_gen.append(d_genres.query('release_year == @year').groupby('genres',as_index=False).mean().sort_values(by='popularity')[['genres','release_year']].tail(1).copy())
top_gen.reset_index(drop=True)#just for making our data frame looking tidy
 
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
      <th>genres</th>
      <th>release_year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Adventure</td>
      <td>2015.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Adventure</td>
      <td>2014.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Action</td>
      <td>1977.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Adventure</td>
      <td>2009.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Adventure</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Adventure</td>
      <td>1999.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Fantasy</td>
      <td>2001.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Adventure</td>
      <td>2008.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Western</td>
      <td>2011.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Fantasy</td>
      <td>2002.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Crime</td>
      <td>1994.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Western</td>
      <td>2012.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Fantasy</td>
      <td>2003.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Science Fiction</td>
      <td>1997.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Adventure</td>
      <td>2013.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Family</td>
      <td>1985.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Fantasy</td>
      <td>2005.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Fantasy</td>
      <td>2006.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Fantasy</td>
      <td>2004.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Crime</td>
      <td>1972.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Science Fiction</td>
      <td>1980.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Fantasy</td>
      <td>2007.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Action</td>
      <td>1979.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Family</td>
      <td>1984.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Adventure</td>
      <td>1983.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Animation</td>
      <td>1995.0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Animation</td>
      <td>1992.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Adventure</td>
      <td>1981.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Crime</td>
      <td>1996.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Adventure</td>
      <td>2000.0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>War</td>
      <td>1982.0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>War</td>
      <td>1998.0</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Animation</td>
      <td>1989.0</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Animation</td>
      <td>1991.0</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Action</td>
      <td>1988.0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>History</td>
      <td>1987.0</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Mystery</td>
      <td>1968.0</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Mystery</td>
      <td>1974.0</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Adventure</td>
      <td>1975.0</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Adventure</td>
      <td>1962.0</td>
    </tr>
    <tr>
      <th>40</th>
      <td>War</td>
      <td>1964.0</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Fantasy</td>
      <td>1971.0</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Adventure</td>
      <td>1990.0</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Animation</td>
      <td>1961.0</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Thriller</td>
      <td>1960.0</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Crime</td>
      <td>1976.0</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Fantasy</td>
      <td>1993.0</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Animation</td>
      <td>1967.0</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Animation</td>
      <td>1963.0</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Adventure</td>
      <td>1986.0</td>
    </tr>
    <tr>
      <th>50</th>
      <td>Animation</td>
      <td>1973.0</td>
    </tr>
    <tr>
      <th>51</th>
      <td>Animation</td>
      <td>1970.0</td>
    </tr>
    <tr>
      <th>52</th>
      <td>Music</td>
      <td>1965.0</td>
    </tr>
    <tr>
      <th>53</th>
      <td>Crime</td>
      <td>1969.0</td>
    </tr>
    <tr>
      <th>54</th>
      <td>Music</td>
      <td>1978.0</td>
    </tr>
    <tr>
      <th>55</th>
      <td>Animation</td>
      <td>1966.0</td>
    </tr>
  </tbody>
</table>
</div>



### Research Question 4 (what genres that producer would like to consider to get highest profit?)
the top three are :
- Adventure
- Fantasy
- Action


```python
#i'll use this color too many times
color_for_bar=(.2,.2,1,0.6)

```


```python
d_genres.groupby('genres').mean()['revenue_adj'].sort_values().plot.bar(color=color_for_bar,title='genres according to revenue')
 
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1c9fcc68da0>




![png](output_28_1.png)


### Research Question 5 (most popular genres of all time?)
the top three are :
- Adventure
- Fantasy
- sci-fi


```python
#data is grouped by genre using d_genres and sorted according to mean popularity
d_genres.groupby('genres').mean()['popularity'].sort_values().plot.bar(color=color_for_bar,title='genres according to popularity')
plt.ylabel('Popularity');
```


![png](output_30_0.png)


### Research Question 6 (what are top 10 highest profit movies?)
the top three are :
- Avatar
- Star Wars
- Titanic


```python

##using groupby() , max() and sorting with respect to profit  we can plot our results  
data.groupby('original_title')['revenue_adj'].max().sort_values().tail(10).plot.bar(color=color_for_bar,title='Top 10 movies with highest  profit')
plt.ylabel('Profit');
```


![png](output_32_0.png)


### Research Question 7 (what are top 10  titles with longest time?)
the runtime appears very large which is strange ... when I searched for those titles with highest runtime they appeared to be **Tv shows** 
- the pacific https://www.themoviedb.org/tv/16997-the-pacific
- the Story of Film: An Odyssey   https://www.imdb.com/title/tt2044056/


that leads to two suggestions:

   - the data set should have colmun like **type** which describe the title is cenamtic movie or TV show
   - genres should be updated to include **Tv show** or  **Movie**
   
I think knowing the type of show will make more good results


```python
##using groupby() , max() and sorting with respect to profit  we can plot our results  

data.groupby('original_title').max()['runtime'].sort_values().tail(10).plot.bar(color=(.2,.2,1,0.6),title='titles with higest runtime');
plt.ylabel('runtime');

```


![png](output_34_0.png)


### Research Question 8 (Directors that are most productive?)



```python
d_director.groupby('director').count()['original_title'].sort_values().tail(10).plot.bar(color=(.2,.2,1,0.6),title='directors that made a lot of titles')
plt.ylabel('Number of titles');
```


![png](output_36_0.png)


### Research Question 9 (Top 10 directors that make movies with high profit?)



```python
d_director.groupby('director')['revenue_adj'].mean().sort_values().tail(10).plot.bar(color=(.2,.2,1,0.6),title='directors with movies that had best profit')
plt.ylabel('profit');
```


![png](output_38_0.png)


### Research Question 10 (Most hired Acotrs?)



```python
'''
'''
d_cast.groupby('cast').count()['original_title'].sort_values().tail(10).plot.bar(color=(.2,.2,1,0.6),title='Actors that made a lot of title')
plt.ylabel('Number of Titles');
```


![png](output_40_0.png)


### Research Question 11 (Most productive Companies?)


```python
def pie_chart(data,title_):
    '''
    plots pie plot for dataframe with title
    data:required data frame to plot its data
    title_:title of graph
    '''
    ax=data.plot.pie(title=title_,autopct='%1.2f%%',figsize=(15,20),legend=False)
    #plot pie plot with percentage
     
    patches, labels =ax.get_legend_handles_labels()
    plt.legend(patches, labels,bbox_to_anchor=(1.20, 1.20)); #set place of legend
     
    plt.show()
    
```


```python
'''using groupby (companies) we can count them according to higshet number of movies
'''
pie_chart(d_companies.groupby('production_companies').count()['original_title'].sort_values().tail(15),'Most Productive Companies')
```


![png](output_43_0.png)


### Research Question 12 (Companies with highest profit?)


```python
'''
using groupby (companies) we can find mean profit and sort them
'''
pie_chart(d_companies.groupby('production_companies')['revenue_adj'].mean().sort_values().tail(15),'Companies with highest')
```


![png](output_45_0.png)


### Research Question 13 (What does change when movies seperated from TV shows?)

- accroding to this article https://en.wikipedia.org/wiki/List_of_longest_films the movies with very long time like Experimental films , longest extended cuts and Films released in separate parts not in our list(that on bar plot) in **question7**  and that lead to most of titles with longest time are Tv shows 

- we can set a limit about 200 min to tell if title is movie or TV show 
- this approach may have fualts as there is maybe some longer movies 

from results below :
**most correlations are the similair like before(but correlation factor is lower) but there can be correlation between runtime of TV show and Vote_average** ...the more successful TV show the more it can last longer

the relation isn't strong enough but if data were clearer , I could make progress



```python
'''
trying to find relations when filtering titles with long time>250 min

'''
sn.heatmap(data[data.runtime>250].corr(), annot=True)
plt.show()
'''
scatter plot for voting and runtime
'''
(x,y)=(data[data.runtime>250].runtime,data[data.runtime>250].vote_average)
plot_scatter_with_line(x,y,'runtime','vote_average','runtime VS vote_average')

data[data.runtime>250].count().runtime## we see that the number of data after flitering is small but most of them are TV show
print('I think Those are Tv shows not movies')
print(data[data.runtime>250].original_title)
```


![png](output_47_0.png)



![png](output_47_1.png)


    I think Those are Tv shows not movies
    415                                         Show Me a Hero
    1183                                             Ascension
    1235                                              Klondike
    1865                                                  Life
    2107                                                Carlos
    2170                              The Pillars of the Earth
    2214                                           The Pacific
    2630                                  Storm of the Century
    2722                                      Band of Brothers
    2843                                       The Blue Planet
    3141                                       Generation Kill
    3356                                            John Adams
    3886                                        Mildred Pierce
    3894                         The Story of Film: An Odyssey
    4041                                                 Taken
    4098                                              Rose Red
    4198                                             The Stand
    4306                                                 Riget
    4788                                     World Without End
    4864                                     Political Animals
    4939                             The Men Who Built America
    5121                                     Angels in America
    5330                                           The Shining
    6008     Crystal Lake Memories: The Complete History of...
    6176                                                 Shoah
    6181                               North and South, Book I
    6829                                         The Lost Room
    6894                                          Planet Earth
    7256                                             SoupÃ§ons
    7267                                        Long Way Round
    7608                                               Tin Man
    8173                                   Pride and Prejudice
    8766                                      The 10th Kingdom
    8768                                  Frank Herbert's Dune
    9300                                         Lonesome Dove
    10304                                           Gettysburg
    Name: original_title, dtype: object
    

<a id='limitations'></a>
## limitations
the data set doesn't have clear evidence to know if a certain title is movie or tv show as in Q67 and Q10 we may use condition like ```(runtime>250) ``` to tell that is Tv show (as most movies about 3 hours) but we should find better way or update or data set from **tmdb**

<a id='conclusions'></a>
## Conclusions

- most popular genre is Adventure
- most hired Actor is Robert De Niro
- most hired director is Robert woody alan
- the top higest profit movie is Avatr
- the director that makes movies with great profit is Hamilton Luske
- the company which has highest profit is Hoya
- genres with highest run time is history(I think becuase it has many TV shows)
- Tv show with long runtime is The Story of Film: An Odyssey
- we see that most of data has no correlation but the following are correlated:
    - profit and budget 
    - profit and popularity 
    - profit and vote_count
    - vote_count and popularity 


```python

```
