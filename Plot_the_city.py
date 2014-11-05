# -*- coding: utf-8 -*-
import pandas as pd
import json
import matplotlib.pyplot as plt
import plotly.plotly as py
from plotly.graph_objs import *
import os
import numpy as np
from nltk.corpus import stopwords
from math import radians, cos, sin, asin, sqrt

"""""""""""""""""""""
# Script http://overpass-turbo.eu/ :

# Code to use for overpass_turbo
# Change the value "my_city 33000" without ""
# version 1 takes only the aera of the city (smaller than version 2 bbox)

<osm-script output="json" timeout="25">
  <id-query {{nominatimArea:my_city 33000}} into="area"/>
  <union>
    <query type="way">
      <has-kv k="highway"/>
      <area-query from="area"/>
    </query>
  </union>
  <!-- print results -->
  <print mode="body"/>
  <recurse type="down"/>
  <print mode="skeleton" order="quadtile"/>
</osm-script>

# Go to export data and "Données brutes"

# Version 2 usualy have bigger data than version 1

<osm-script output="json" timeout="25">
  <union>
    <query type="way">
      <has-kv k="highway"/>
      <has-kv k="name"/>
      <bbox-query {{nominatimBbox:my_city 33000}}/>
    </query>
  </union>
  <!-- print results -->
  <print mode="body"/>
    <recurse type="down"/>
  <print mode="skeleton" order="quadtile"/>
</osm-script>

# Go to export data and "Données brutes"

"""""""""""""""""""""

"""""""""""""""""""""
 Function
"""""""""""""""""""""

def haversine(lon1, lat1, lon2, lat2):
    """
    To calcul the distance between two points in KM
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6367 * c
    return km


def clean_stopwords(x, y):
    """
    To split way name for analyse
    """
    x = x.replace('É', 'e') # "É" can't be lower()
    x = x.replace('-', ' ') # To split some funny way
    x = x.replace("l'", "") # To better split and matching
    try:
        x = [w for w in x.split() if w.lower() not in stopw]
        return x[y].lower()
    except:
        return('')


stopw = stopwords.words('french')

"""""""""""""""""""""
Loading File
"""""""""""""""""""""

cwd = os.getcwd()
cwd_data = os.path.join(cwd, 'data')

# Dico_gender
file_name_proportion = 'Dico_gender_proportion.csv'
data_path_dico = os.path.join(cwd_data, file_name_proportion)

# Title list
file_name_title = 'title_list.csv'
data_path_title = os.path.join(cwd_data, file_name_title)

# Personality list
file_name_perso = 'personality_list.csv'
data_path_perso= os.path.join(cwd_data, file_name_perso)

print "loading data files..."

dico = pd.read_csv(data_path_dico)
# rework for dico
dico = dico[['prenoms', 'sexe']]
dico.columns = ['firstname', 'gender']

title = pd.read_csv(data_path_title, sep=';')
# rework for title
title.columns = ['title_name', 'title_gender']
#title['title_name'] = title['title_name'].str.encode('utf-8')

personality = pd.read_csv(data_path_perso, sep=';')
# rework for personality
personality.columns = ['perso_name', 'perso_gender']
#personality['perso_name'] = personality['perso_name'].str.encode('utf-8')


print "Working with OpenStreetMap data..."

db = json.load(open('export.json'))

"""""""""""""""""""""
Dataframe
"""""""""""""""""""""

print "Analyse ways..."
way = []
autre = 0
for rec in db[u'elements']:
    if rec['type'] == 'way':
        for node in rec[u'nodes']:
            try:
                way.append({'way_name' : rec[u'tags'][u'name'],
                            'way_id' : rec['id'],
                            'way_node_id': node})
            except:
                autre = autre + 1
    else:
        pass

df_way = pd.DataFrame(way)

print "Analyse nodes..."
node = []

for rec in db[u'elements']:
    if rec['type'] == 'node':
        node.append({'node_id' : rec[u'id'],
                     'node_lat': rec[u'lat'],
                     'node_lon': rec[u'lon']})
    else:
        pass

                            
df_node = pd.DataFrame(node)

data = pd.merge(df_way, df_node, left_on='way_node_id', right_on='node_id', how='left')

print "Working on street's name..."
street = pd.DataFrame(data[['way_name', 'way_id']])
street.drop_duplicates(inplace=True)

# Create news colums to decode name of street
street['way_name_1'] = street['way_name'].apply(lambda x: clean_stopwords(x.encode('utf-8'), int(0)))
street['way_name_2'] = street['way_name'].apply(lambda x: clean_stopwords(x.encode('utf-8'), int(1)))
street['way_name_3'] = street['way_name'].apply(lambda x: clean_stopwords(x.encode('utf-8'), int(2)))


# Merging with list of personnality
result = pd.merge(street, personality, left_on='way_name_2', right_on='perso_name', how='left')
result = pd.merge(result, personality, left_on='way_name_3', right_on='perso_name', how='left', suffixes=['_1', '_2'])

result['perso_gender_1'] = result['perso_gender_1'].fillna('Inconnu')
result['perso_gender_2'] = result['perso_gender_2'].fillna('Inconnu')


result['way_gender'] = 'Inconnu' # Init way_gender

result['way_gender'] = np.where(result['perso_gender_1'] != 'Inconnu',
                                result['perso_gender_1'],
                                result['perso_gender_2'])

# Merging street & title list
result = pd.merge(result, title, left_on='way_name_2', right_on='title_name', how='left')
result = pd.merge(result, title, left_on='way_name_3', right_on='title_name', how='left', suffixes=['_1', '_2'])

result['title_gender_1'] = result['title_gender_1'].fillna('Inconnu')
result['title_gender_2'] = result['title_gender_2'].fillna('Inconnu')

# Create a boolean for title matching (0 = False)
#result['title_matching'] = 0
#result['title_matching'][(~result['title_name_1'].isnull()) | (~result['title_name_2'].isnull())] = 1

result['way_gender'] = np.where(result['way_gender'] == 'Inconnu',
                                np.where(result['title_gender_1'] != 'Inconnu',
                                         result['title_gender_1'], 
                                         result['title_gender_2']),
                                result['way_gender'])

# Merging street & dico
result = pd.merge(result, dico, left_on='way_name_2', right_on='firstname', how='left')
result = pd.merge(result, dico, left_on='way_name_3', right_on='firstname', how='left', suffixes=('_join_1', '_join_2'))

result['gender_join_1'] = result['gender_join_1'].fillna('Inconnu')
result['gender_join_2'] = result['gender_join_2'].fillna('Inconnu')

result['way_gender'] = np.where(result['way_gender'] == 'Inconnu',
                                np.where(result['gender_join_1'] != 'Inconnu',
                                         result['gender_join_1'],
                                         result['gender_join_2']),
                                result['way_gender'])
                  
print "Finishing to find gender..."

data = pd.merge(data, result, left_on='way_id', right_on='way_id', how='left', suffixes=['','_2'])
data = data[['way_id', 'way_name', 'way_gender', 'node_lat', 'node_lon']]

data['way_name']= data['way_name'].str.encode('utf-8')

if len(data) > 10000:
    # To reduce number of points by street
    print "Delete some points ..."
    
    index_to_delete = []    
    
    print "Number of total points : " + str(len(data))  
    for i, grp, in data.groupby(['way_id', 'way_gender']):
        if len(grp) > 3:
            # we take some nodes between the first and the last one of a street
            # less node for pretty the same line of street
            index_to_delete.extend(range(grp.index.tolist()[1], grp.index.tolist()[-1],2))
        else:
            pass    
    data.drop(index_to_delete, inplace=True)
    data['next_lat'] = data['node_lat'].shift(-1)
    data['next_lon'] = data['node_lon'].shift(-1)

    data['distance'] = data.apply(lambda row: haversine(row['node_lon'], row['node_lat'], row['next_lon'], row['next_lat']), axis=1)
    data = data[data.distance > data.distance.quantile(q=0.1)]
    print 'Number line delete :' + str(len(index_to_delete)) + ' len data : ' +str(len(data))
        
else:
        pass

df_i = data[data['way_gender'] == 'Inconnu']
df_m = data[data['way_gender'] == 'M']
df_f = data[data['way_gender'] == 'F']

"""""""""""""""""""""
Matplotlib graph
"""""""""""""""""""""
"""

fig = plt.figure(figsize=(12,8))
#for index, grp, in city.groupby(['way_id']):
#    plt.plot(grp.node_lon, grp.node_lat, color='g', alpha=0.35)
for index, grp, in df_i.groupby(['way_id']):
    plt.plot(grp.node_lon, grp.node_lat, color='k', alpha=0.4, linewidth=0.2, label=grp.way_name.unique())
    #plt.scatter(grp.node_lon, grp.node_lat, color='k', alpha=0.5, label=grp.way_name.unique(), s=0.2)
for index, grp, in df_m.groupby(['way_id']):
    plt.plot(grp.node_lon, grp.node_lat, color='b', alpha=0.4, linewidth=0.2, label=grp.way_name.unique())
    #plt.plot(grp.node_lon, grp.node_lat, color='b', alpha=0.65)
for index, grp, in df_f.groupby(['way_id']):
    plt.plot(grp.node_lon, grp.node_lat, color='m', alpha=0.4, linewidth=0.2, label=grp.way_name.unique())
    #plt.plot(grp.node_lon, grp.node_lat, color='m', alpha=0.65)

plt.title("Bordeaux's gender streets")


plt.show()

"""
"""""""""""""""""""""
End Matplotlib graph
"""""""""""""""""""""



"""""""""""""""""""""
Graph Plotly
"""""""""""""""""""""

colors = dict(
    Inconnu='#414649', 
    M='#5EB4E6', 
    F='#DC0671'
)

sizemode='area'

def make_trace(X, street_gender, color):  
    return Scatter(
        x=X['node_lon'],  # GDP on the x-xaxis
        y=X['node_lat'],    # life Exp on th y-axis
        name=street_gender,    # label continent names on hover
        mode='lines',    # (!) point markers only on this plot
        line= Line(width=0.8,   
                   shape='spline',
                   color=color,
                   opacity=0.7)
    )

# Initialize data object 
data_plot = Data()

for way, X in data.groupby(['way_gender', 'way_id']):
    for txt in X.way_gender:
        street_gender = txt
    
    color = colors[way[0]]
    if street_gender == 'Inconnu':
        street_gender = 'Unknow'
    elif street_gender == 'M':
        street_gender = 'Male'
    else:
        street_gender = 'Female'

    data_plot.append(
        make_trace(X, street_gender, color)  # append trace to data object
    )

title = "My City's streets gender"

layout = Layout(
    title=title,             # set plot title
    xaxis=XAxis(
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        autotick=True,
        ticks='',
        showticklabels=False
     ),
    yaxis=YAxis(
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        autotick=True,
        ticks='',
        showticklabels=False
    ),
    annotations=Annotations([
    Annotation(
        text='Twitter : @arm_gilles',  # annotation text
        showarrow=False,                     # remove arrow 
        xref='paper',   # use paper coords
        yref='paper',   #  for both x and y coordinates
        x=0.97,         # x-coord (slightly of plotting area edge)
        y=0.01,         # y-coord (slightly of plotting area edge)
        font=Font(size=14),   # increase font size (default is 12)
        #bgcolor='#FFFFFF',    # white background
        borderpad=4           # set border/text space (in pixels)
    )
    ])
)    
#plot_bgcolor='#F8FAEE'   # set plot color to grey



# Make Figure object
fig = Figure(data=data_plot, layout=layout)

# Update the layout object
fig['layout'].update(
    hovermode='closest',   # (!) hover -> closest data pt
    
    showlegend=False     # remove legend (info in hover)
)
"""
    autosize=False,       # turn off autosize
    width=650,            # plot width
    height=500,           # plot height
    """


# Define a hover-text generating function (returns a list of strings)
def make_text(X):
    return 'Name: \
    <br>%s'\
    % (X['way_name'])
    
# Again, group data frame by continent sub-dataframe (named X),
#   make one trace object per continent and append to data object   
i_trace = 0                                        # init. trace counter
for way, X in data.groupby(['way_gender', 'way_id']):
    for txt in X.way_name:
        street_name =  "Name: \<br>" + txt  
    text = X.apply(make_text, axis=1).tolist()      # get list of hover texts
    fig['data'][i_trace].update(text=text)         # update trace i
    i_trace += 1                                   # inc. trace counter

#py.plot(fig, filename="My City's streets gender") 


""""""""""""""""""""" 
End Graph Plotly
"""""""""""""""""""""


