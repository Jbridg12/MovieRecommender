#   file:   recommendation_server.py
#   Name: Joshua Bridges
#   This code interacts witht eh html server with post requests and reports back a
#   set of 5 recommendations is JSON format. These recommendations are calculated at bootup 

from http.server import BaseHTTPRequestHandler, SimpleHTTPRequestHandler, HTTPServer
from tqdm import tqdm
import numpy as np
import pandas as pd
import math
import socketserver
import urllib
import json
import codecs

hostName = "localhost"
serverPort = 9000

class RecommendationEngine():
    
    #Define a constructor for the Recommender class to initialize some variables
    def __init__(self, s, m):
        
        #Check if the request is 0 for content-based or 1 for collaborative
        if m == 0:
            
            #Store the name, the movies initialize a list for instanced genres
            self.movie = s
            self.movTh = pd.read_csv("./movielens/movies.csv")
            self.genre = []
            
        else:
            #Store the Id and dataframes of the movies and users
            self.Id = int(s)
            self.movRa = pd.read_csv("./movielens/ratings.csv")
            self.movTh = pd.read_csv("./movielens/movies.csv")
    
    #RecommendationEngine process to construct a dataframe of top recommendations based on genre
    def do_ContBased(self):
        #Separate two columns from the large dataframe
        s = self.movTh['title']
        gens = self.movTh['genres']
        
        #initialize list for appending
        cosim = []
        
        #Filter out the users genre set
        temp = self.movTh.loc[self.movTh['title'] == self.movie, ['genres']]
        temp.reset_index(drop=True, inplace=True)
        
        #Break it up into a list of genres
        self.genre = temp['genres'][0].split('|')

        #Count up the index and pass over the genres to extract the lists
        k = 0
        for m in gens:
        
            #Make sure it isn't itself
            if s[k] != self.movie:
            
                #Create the list of genres
                genL = m.split('|')
                
                #Find the elements in common and count them
                coms = set(self.genre).intersection(genL)
                numerator = len(coms)
                
                #Take the square root of the squares of the lengths of the two genre lists
                deno = math.sqrt(len(self.genre)*len(genL))
                
                #Divide the numerator by the denominator to find the cosine similarity and append it to a running list
                cosim.append(numerator/deno)
                
            else:
                
                #If it is the same movie then we append 0 to imply no desire to rewatch immediately
                cosim.append(0)
                
            #Increment index
            k += 1        
            
        #Attach the cosine similarity column into the movie dataframe
        self.movTh['cosine_similarity'] = cosim
        
        #Sort the movies by their cosine similarity and return the top 5 movies
        sortrecs = self.movTh.sort_values(by=['cosine_similarity'], ascending=False)
        self.contrecs = sortrecs.drop_duplicates(subset=['title']).head()

    #RecommendationEngine process to construct a dataframe of top recommendations based on user ratings
    def do_CollBased(self):
        
        #Create a copy of the ratings dataframe
        t = self.movRa.copy()
        
        #Create a few subframes of the initial dataframe here
        # 1)self.rates : the frame isolated to the ratings made by the user
        # 2)user_rat : the frame of the user ratings indexed by movieId of only the ratings column
        # 3)num_user : a numpy array of the user_rat column
        #---------------------------------------------------------------------------------
        self.rates = t.loc[t['userId'] == self.Id, ['movieId', 'rating', 'timestamp']]
        user_rat = self.rates.set_index('movieId')['rating']
        num_user = user_rat.copy().to_numpy()
        
        #Get the norm of the user ratings array for use in the denominator of the cosine similarity
        usr_norm = math.sqrt(np.sum(num_user**2))
        
        #Declare a few structures for later
        coss = []
        l = []
        
        #This will be the storage of the all movies ratings based off the top 10 nearest users
        movies_done = pd.DataFrame(columns=['movieId', 'apprat'])
        
        #Loop over the amount of userIds that are in the file. Ideally this would be dynamically found by
        # deleting duplicates and then counting the length of the dataframe but that was an extra step I 
        # didn't have the time to implement 
        for k in range(1, 611):
        
            #Make sure it isn't comparing a user to themself
            if k != self.Id:
                
                #Isolate the userId's (k) movie ratings and index them by movie
                comp_rates = t.loc[t['userId'] == k, ['movieId', 'rating', 'timestamp']]
                doip = comp_rates.set_index('movieId')['rating']
                
                #Make a numpy array out of the ratings column
                vec = doip.to_numpy()
            
                #Calculate the norm of the (k) users rating vector
                norm = math.sqrt(np.sum(vec**2))
                
                #Align the length of the two arrays and fill with 0's to allow for dot product
                user_rat, doip = user_rat.align(doip, axis=0, fill_value=0)
                dotp = user_rat.dot(doip)
            
                #Calculate the cosine similarit and create a list of the similarity to add to a column of the dataframe
                cos_sim = dotp/(norm*usr_norm) 
                l = [cos_sim] * len(vec)
                
            else:
                #If it is itself then push a similarity of 0 to the list 
                l = [0] * len(num_user)
            
            #Extend this list with the cosine similarity each loop to create a new column the same size as the original frame
            coss.extend(l)
        
        #After the loop add the new column
        t['cos_sim'] = coss
        
        #Create a dataframe of anything but the user's movies with the cosine similarities attached
        t2 = t.loc[t['userId'] != self.Id, ['movieId', 'rating', 'timestamp', 'cos_sim']]
            
        #Loop over the movies the user hasn't seen 
        for i in t2['movieId']:
        
            #This checks to make sure movies aren'y being evaluated twice
            if i not in movies_done['movieId'].tolist():
                
                #Filter out all the ratings and cosine similarities and isolate the top 10 most similar user ratings and reset the index
                movers = t2.loc[t2['movieId'] == i, ['rating', 'cos_sim']]
                vip = movers.sort_values(by=['cos_sim'], ascending=False).head(10)
                vip.reset_index(drop=True, inplace=True)
                
                #Start sum counter
                sum = 0
                
                #Loop over the top similar ratings 
                for j in range(len(vip)):
                
                    #Keep running total of the product of the rating and cosine similarity
                    sum += (vip['rating'][j] * vip['cos_sim'][j])
                    
                #Add the movie to the movies_done dataframe to indicate it has been done
                movies_done = movies_done.append({'movieId' : i, 'apprat' : sum}, ignore_index=True)
        
        #return the results with all duplicates filtered out just in case
        self.collrec = movies_done.drop_duplicates(subset=['movieId'])
        

class RecommendationWebServer(BaseHTTPRequestHandler):
    def _set_headers(self):
        # WARNING: DO NOT TOUCH THIS CODE!
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        self.end_headers()
        
    def do_HEAD(self):
        # WARNING: DO NOT TOUCH THIS CODE!
        self._set_headers()
        
    # GET sends back a Hello world message
    def do_GET(self):
        self._set_headers()

        # A dummy list of recommendations and similarity scores.
        self.wfile.write(bytes(json.dumps({'hello': 'world', 'received': 'ok'}), 'utf-8'))
    
    # POST echoes the message adding a JSON field
    def do_POST(self):
        # read the message and convert it into a python dictionary
        length = int(self.headers['Content-Length'])
        message = urllib.parse.parse_qs(self.rfile.read(length), keep_blank_values=1)

        # - Uncomment the following the print out the message type received.
        print("Received: ")
        print(message)
        print(message[b'type'])

        # Define an empty dictionary that we will send back as a response.
        response = {}
		
        # Extract the request type, i.e. "content-based" or "collaborative"
        reqType = message[b'type'][0]
        
        
        # Choose what to do based on our request type.
        if reqType == b"content-based":
        
            # Extract the movie title from the HTTP request.
            movieTitle = message[b'title'][0].decode('utf-8')
            
            #Get a dataframe of the 'info' column which in these cases are the vectors of the top 5 recommendations information
            #Filtered to find the appropriate information for the selected movie title
            stuf = mov_recs.loc[mov_recs['name'] == movieTitle, ['info']].copy()
            stuf.reset_index(drop=True, inplace=True)
            
            
            # Here, your recommender system returns the Top 5 movie suggestions based on genre.
            
            response['movies'] = [
                {'title' : stuf.iloc[0]['info'][0][0], 'cosine_sim' : stuf.iloc[0]['info'][0][1]},
                {'title' : stuf.iloc[0]['info'][1][0], 'cosine_sim' : stuf.iloc[0]['info'][1][1]},
                {'title' : stuf.iloc[0]['info'][2][0], 'cosine_sim' : stuf.iloc[0]['info'][2][1]},
                {'title' : stuf.iloc[0]['info'][3][0], 'cosine_sim' : stuf.iloc[0]['info'][3][1]},
                {'title' : stuf.iloc[0]['info'][4][0], 'cosine_sim' : stuf.iloc[0]['info'][4][1]}
            ]

        elif reqType == b"collaborative":
            
            # Extract the user ID from the HTTP request.
            userId = int(message[b'user'][0].decode('utf-8'))
            
            #Pick out the info vector for the user specified by the 'userId' recommendations
            stu = usr_recs.loc[usr_recs['id'] == userId, ['info']].copy()
            stu.reset_index(drop=True, inplace=True)

            # Here, your recommender system returns the Top 5 movie suggestions based on ratings.
            response['movies'] = [
                {'title' : stu.iloc[0]['info'][0][0], 'rating' : stu.iloc[0]['info'][0][1]},
                {'title' : stu.iloc[0]['info'][1][0], 'rating' : stu.iloc[0]['info'][1][1]},
                {'title' : stu.iloc[0]['info'][2][0], 'rating' : stu.iloc[0]['info'][2][1]},
                {'title' : stu.iloc[0]['info'][3][0], 'rating' : stu.iloc[0]['info'][3][1]},
                {'title' : stu.iloc[0]['info'][4][0], 'rating' : stu.iloc[0]['info'][4][1]}
            ]

        else:
            # Return an empty response because the respType is invalid.
            response['movies'] = []
        
        # send the message back

        self._set_headers()
        self.wfile.write(bytes(json.dumps(response), 'utf-8'))


# If we call this file from the Terminal as: "python recommendation_server.py", then ...
if __name__ == "__main__":        

    #Fetch the information in dataframes from the csv files
    rating = pd.read_csv("./movielens/ratings.csv")
    movies = pd.read_csv("./movielens/movies.csv")
    
    #Create two dataframes for the different recommenders results
    mov_recs = pd.DataFrame(columns=['name', 'info'])
    usr_recs = pd.DataFrame(columns=['id', 'info'])
    
    #Do all the calculations for the Content-based Filtering in one loop
    for name in tqdm(movies['title']):
    
        #Create the recommendation engine and run content based recommendation for each movie 
        RE = RecommendationEngine(name, 0)
        RE.do_ContBased()
        
        #Get the results from the Engine
        stuf = RE.contrecs[['title', 'cosine_similarity']].copy()
        stuf.reset_index(drop=True, inplace=True)
        
        #Append it to the result dataframe in the format:
        #                                  [name , <title, cosine_similarity>]
        #With the second column being a column of the recommended movie and the similarity
        mov_recs = mov_recs.append({'name' : name, 'info' : [[stuf['title'][0], stuf['cosine_similarity'][0]], [stuf['title'][1], stuf['cosine_similarity'][1]], [stuf['title'][2], stuf['cosine_similarity'][2]], [stuf['title'][3], stuf['cosine_similarity'][3]], [stuf['title'][4], stuf['cosine_similarity'][4]]]}, ignore_index=True)
    
    #Do all the calculations for the Collaborative-based filtering in one loop
    for id in tqdm(range(1, 611)):
        
        #Create the recommendation engine and run collaborative based recommendation for each movie
        RE = RecommendationEngine(id, 1)
        RE.do_CollBased()
        
        #Get the results from the engine and get the top 5 recommendations
        bro = RE.collrec.copy()
        top = bro.sort_values(by=['apprat'], ascending=False).head(5)
        top.reset_index(drop=True, inplace=True)
        
        #Append in the same format as stated earier to a separate results frame
        usr_recs = usr_recs.append({'id' : id, 'info' : [[RE.movTh.loc[RE.movTh['movieId'] == top['movieId'][0]].iloc[0]['title'], top['apprat'][0]], [RE.movTh.loc[RE.movTh['movieId'] == top['movieId'][1]].iloc[0]['title'], top['apprat'][1]], [RE.movTh.loc[RE.movTh['movieId'] == top['movieId'][2]].iloc[0]['title'], top['apprat'][2]], [RE.movTh.loc[RE.movTh['movieId'] == top['movieId'][3]].iloc[0]['title'], top['apprat'][3]], [RE.movTh.loc[RE.movTh['movieId'] == top['movieId'][4]].iloc[0]['title'], top['apprat'][4]]]}, ignore_index=True) 
    
    # Instantiate and start serving an HTTP server
    webServer = HTTPServer((hostName, serverPort), RecommendationWebServer)
    print("[SERVING] Reccomendation Server : http://%s:%s" % (hostName, serverPort))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    # close the server after we've hit a keyboard interrupt
    webServer.server_close()
    print("Server stopped.")
