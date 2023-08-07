#!/usr/bin/env python
# coding: utf-8

# # **TASK 1**

# In[1]:


get_ipython().system('pip3 install beautifulsoup4')
get_ipython().system('pip3 install requests')

##Import Libraries
import requests
from bs4 import BeautifulSoup
import pandas as pd


# In[2]:


url1 = 'https://en.wikipedia.org/wiki/Toy_Story_3'

#Upload Data
response = requests.get(url1)

#Convert to a BS object
soup = BeautifulSoup(response.text, "html.parser")

print(soup.title.string)


# In[3]:


#Print all of the HTML
#Use pretiffy to visualize and understand the structure of HTML and XML documents
Info = soup.prettify()
Info


# In[4]:


#Use the find function to find just a table
Info_box = soup.find(class_ = 'infobox vevent')
Info_rows = Info_box.find_all('tr')

for row in Info_rows:
    print(row.prettify())


# In[5]:


#Create a blanc dictionary
movie_info = {}


#We want to brake it into two cases: If it has a list and if it doesn't
def get_content_value(row_data):
    if row.find('li'):
        return [li.get_text(' ', strip=True).replace("\xa0", " ") for li in row_data.find_all('li')]
    else:
        return row_data.get_text(' ', strip=True).replace("\xa0", " ")

#Get index and row at the same time
for index, row in enumerate(Info_rows):
    if index == 0:
        movie_info['title'] = row.find("th").get_text()
    elif index == 1:
        continue
    else:
        content_key = row.find('th').get_text(' ', strip=True)
        content_value = get_content_value(row.find('td'))
        movie_info[content_key] = content_value


movie_info


# # **TASK 2** : Get info box for all Disney Movies
# 

# In[6]:


#Download Info
url = 'https://en.wikipedia.org/wiki/List_of_Walt_Disney_Pictures_films'

response = requests.get(url)

#Convert to a BS object
soup = BeautifulSoup(response.text, "html.parser")

#print out content
contents = soup.prettify()
print(contents)



# In[7]:


#Select lets us get information from a table
movies = soup.select('.wikitable.sortable i')
movies[0:20]
movies[9].a['href']


# In[8]:


def get_content_value(row_data):
        if row_data.find('li'):
            return [li.get_text(' ', strip=True).replace("\xa0", " ") for li in row_data.find_all('li')]
        elif row_data.find('br'):
            return [text for text in row_data.stripped_strings]
        else:
            return row_data.get_text(' ', strip=True).replace("\xa0", " ")

def clean_tags(soup):
    for tag in soup.find_all(['sup', 'span']):
        tag.decompose()   
        
def get_info_box(url):
        
    response = requests.get(url)
    
    soup = BeautifulSoup(response.text, "html.parser") 
    info_box = soup.find(class_ = 'infobox vevent')
    info_rows = info_box.find_all('tr')
    
    clean_tags(soup)   
    
    movie_info = {}    
    for index, row in enumerate(info_rows):
        if index == 0:
            movie_info['title'] = row.find("th").get_text(" ", strip=True)
        else:
            header = row.find("th")
            if header:
                content_key = row.find('th').get_text(' ', strip=True)
                content_value = get_content_value(row.find('td'))
                movie_info[content_key] = content_value


    return movie_info


# In[9]:


url = 'https://en.wikipedia.org/wiki/List_of_Walt_Disney_Pictures_films'

response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")
movies = soup.select('.wikitable.sortable i a')

base_path = "https://en.wikipedia.org/"

movie_info_list = []
print(len(movies))

for index, movie in enumerate(movies):
    if index % 10 == 0:
        print(index)
    try:
        relative_path = movie['href']
        full_path = base_path + relative_path
        title =  movie['title']
        
        
        movie_info_list.append(get_info_box(full_path))
    
    except Exception as e:
        
        print(movie.get_text())
        print(e)


# In[10]:


get_info_box('https://en.wikipedia.org/wiki/The_Reluctant_Dragon_(1941_film)')


# In[11]:


movie_info_list[9]


# ### Save/Reload Movie Data

# In[37]:


import json

def save_data(title, data):
    with open(title, 'w', encoding = 'utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# In[ ]:


import json

def load_data(title):
    with open(title, encoding = 'utf-8') as f:
        return json.load(f)


# In[ ]:


save_data('disney_data_cleaned.json', movie_info_list)


# # Task 3: Data Cleaning

# In[ ]:


movie_info_list = load_data("disney_data_cleaned.json")


# ### Tasks
# 
# 1. ~~Clean up references [1] [2] etc.~~
# 2. ~~convert running time into an integer~~
# 3. convert dates into datetime object
# 4. ~~split up the long string~~
# 5. Convert Budget & Box office to numbers

# In[ ]:


# clean up references (remove [1] [2] etc..)
# Done within the original code



# In[ ]:


# Split up the long string
# Done within the original code




# In[ ]:


# Convert all of the running timos into integer

movie_info_list[-10]


# In[ ]:


print([movie.get('Running time', 'N/A') for movie in movie_info_list])


# In[18]:


#74 minutes
def minutes_to_integer(running_time):
    if running_time == "N/A":
        return None
    
    if isinstance(running_time, list):
        return int(running_time[0].split(" ")[0])
    
    else: # is a string
        return int(running_time.split(" ")[0])

for movie in movie_info_list:
    movie['Running time (int)'] = minutes_to_integer(movie.get('Running time', "N/A"))    


# In[19]:


movie_info_list[-10]


# In[22]:


print([movie.get("Release date",'N/A') for movie in movie_info_list])


# In[21]:


movie_info_list[-15]


# In[34]:


from datetime import datetime

dates = [movie.get('Release date', 'N/A') for movie in movie_info_list]

def clean_date(date):
    return date.split("(")[0].strip()

def date_conversion(date):
    if isinstance(date, list):
        date = date[0]
    
    if date == 'N/A':
        return None
        
    date_str = clean_date(date)
    print(date_str)    
    fmts = ["%B %d, %Y", "%d %B %Y"]
    for fmt in fmts:
        try:
            return datetime.strptime(date_str, fmt)
        except:
            pass
        return None


# In[35]:


for movie in movie_info_list:
    movie['Release date (datetime)'] = date_conversion(movie.get('Release date', 'N/A'))


# In[36]:


movie_info_list[8]


# In[48]:


import pickle

def save_data_pickle(name, data):
    with open(name, 'wb') as f:
        pickle.dump(data, f)


# In[53]:


import pickle

def load_data_pickle(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


# In[54]:


save_data_pickle('disney_movie_data_cleaned_more.pickle', movie_info_list)


# In[55]:


a = load_data_pickle('disney_movie_data_cleaned_more.pickle')

a[5]


# In[56]:


a == movie_info_list


# ### TASK 4: Attach IMDB/Rotten Tomatoes/Metascore scores
# 

# In[58]:


movie_info_list = load_data_pickle('disney_movie_data_cleaned_more.pickle')


# In[59]:


movie_info_list[-60]


# In[60]:


#https://www.omdbapi.com/?apikey=[yourkey]&


# In[69]:


import requests
import urllib
import os

def get_omdb_info(title):
    base_url = 'https://www.omdbapi.com/?'
    parameters = {'apikey': os.environ['OMDB_API_KEY'], 't': title}
    params_encoded = urllib.parse.urlencode(parameters)
    full_url = base_url + params_encoded
    requests.get(full_url).json()
    return requests.get(full_url).json()

def get_roteen_tomato_score(omdb_info):
    ratings = omdb_info.get('Ratings', [])
    for rating in ratings:
        if rating['Source'] == 'Rotten Tomatoes':
            return rating['Value']
        return None
    
get_omdb_info('Into the woods')


# In[71]:


for movie in movie_info_list:
    title =  movie['title']
    omdb_info = get_omdb_info(title)
    movie['imdb'] = omdb_info.get('imdbRating', None)
    movie['metascore'] = omdb_info.get('Metascore', None)
    movie['rotten_tomatoes'] = get_totten_tomato_score(omdb_info)
    


# In[72]:


save_data_pickle('disney_movie_data_final.pickle', movie_info_list)


# ### Task 5: Save data as JSON & CSV

# In[74]:


movie_info_copy = [movie.copy() for movie in movie_info_list]

for movie in movie_info_copy:
    current_date = movie['Release date (datetime)']
    if current_date:
        movie['Release date (datetime)'] = current_date.strftime("%B %d, %Y")
    else:
         movie['Release date (datetime)'] = None


# In[76]:


save_data('disney_data_final.json', movie_info_list)


# In[79]:


#conver to CSV
import pandas as pd
df = pd.DataFrame(movie_info_list)
df.head()


# In[80]:


df.to_csv("disney_data_final.csv")


# In[ ]:





# In[ ]:





# In[ ]:




