Gender-Street-Plot
==================

- use Plotly to have some great plot (https://plot.ly/ for their awsome work !)
- Use Openstreetmap & overpass-turbo.eu to have some data about a city (licence ODbL)
- Use previous model (https://github.com/armgilles/Street-Gender-CUB) to find gender's streets (files are in data directory).

Result for Bordeaux (French city) : https://plot.ly/~babou/39

Python library :
 - Pandas
 - json
 - matplotlib
 - numpy
 - nltk
 - math

How to do :

1) First you have to sign in to https://plot.ly if you want to try it (if you just want a classic plot it is not required)

2) Go to http://overpass-turbo.eu. Change the request (in the begin of the .py) "my_city 33000" to your city name and paste it  at overpass-turbo. Warning check if your request is correct and if not, be more accurate (postcode, country etc...). Save file "export.json" in the directory.

3) If you want to use Plotly :
  - Change the title ("My city's title") and the name ("My city") of your city's plot (at the end of the py file) and run it ! At the end, a web page will be pop with your graph on plotly.
  
   If you don't want to use plotly :
  - You should try !
  - You can use Matplotlib. Just uncomment the matplolib code and comment the plotly one.


PS : You can use it with a non french's city, but the gender model is based on French firstname open data. If you want to test it, just change "french" at "stopw = stopwords.words('french')" to "english".

Any help is appreciated to improve personnality & title file for a better matching :]



