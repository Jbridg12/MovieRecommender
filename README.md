# Volfix
Volfix is an implementation of a recommendation system into a pre-established html server. 
The program enables two types of recommendation: collaborative based and content based. The
information on the movies and the users are found in the "\movielens\movies.csv" and "\movielens\ratings.csv".
These recommendations are calculated by finding the cosine similarity between options
and providing the most similar options to the client.

# Dependencies
Alongside the initial packages the project utilized from the beginning, Volfix uses three more packages:
- tdqm
- numpy
- pandas

# How do I run Volflix?
Open two terminal windows. In one window, run "python recommendation_server.py". 
In the other terminal window, run"python html_server.py". You should now be capable
of opening the Volflix interface by navigating to http://localhost:8080/ in your browser.
From the Volfix browser page there are two options provided for you. You can select the
"Content-based filtering" tab to find movies similar to a given movie. Or you can select
the "Collaborative-based filtering" tab to see movies recommended to a certain user.

# Running Heads Up
The process as a whole is extremely slow to boot up as is. During the startup of the server, 
the calculations for the collaborative-based filtering require almost too much time. I recommend 
before running to go to line 297 and change the range(1, 611) to range(1, 5) or lower to allow
for a more realistic loading time. 

# Submission Info
- Joshua Bridges
- jbridg12
- 03/11/2021
