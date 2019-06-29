class recommender:

    def __init__(self, data_ratings,data_demo, product, k=1, metric='cosine', n=1,w1=0.0,w2=1.0):
        """ initialize recommender
        currently, if data_ratings is dictionary the recommender is initialized
        to it.
        For all other data types of data_ratings, no initialization occurs
        k is the k value for k nearest neighbor
        metric is which distance formula to use
        n is the maximum number of recommendations to make
        data_demo is demo info of all users
        w1 is the weight of demo info
        w2 is the weight of product ratings info
        """
        self.k = k
        self.n = n
        self.metric = metric
        self.w1=w1
        self.w2=w2
        self.product=product
        #
        # if data_ratings is dictionary set recommender data_ratings to it
        #
        
        if type(data_ratings).__name__ == 'dict':
            self.data_ratings = data_ratings
            self.data_demo=data_demo
            
           

    def userprofile(self, id):
        """Return n top ratings for user with id"""
        #print ("Ratings for " + self.data_ratings[id])
        ratings = self.data_ratings[id]
        print ('%i items rated'%len(ratings))
        ratings = list(ratings.items())
        ratings = [(k, v)
                   for (k, v) in ratings]
        print (self.data_demo[id])
        print (self.data_ratings[id])
        # finally sort and return
        '''
        ratings.sort(key=lambda artistTuple: artistTuple[1],
                     reverse = True)
        ratings = ratings[:n]
        for rating in ratings:
            print("%s\t%i" % (rating[0], rating[1]))
        '''
                
        
    def pearson(self, userid1, userid2,):
        '''
        Calculate pearson correlation given two user ID
        The first value is demo variable similarity
        The second value is product ratings similarity
        '''
        pearson_sim=[]
        for i in range(0,2):
            sum_xy = 0.0
            sum_x = 0.0
            sum_y = 0.0
            sum_x2 = 0.0
            sum_y2 = 0.0
            n = 0
            if i==1:
                rating1=self.data_ratings[userid1]
                rating2=self.data_ratings[userid2]
            else:
                rating1=self.data_demo[userid1]
                rating2=self.data_demo[userid2]
            for key in rating1:
                if key in rating2:
                    n += 1
                    x = rating1[key]
                    y = rating2[key]
                    sum_xy += x * y
                    sum_x += x
                    sum_y += y
                    sum_x2 += pow(x, 2)
                    sum_y2 += pow(y, 2)
            if n == 0:
                pearson_sim.append(0)
        # now compute denominator
            denominator = (sqrt(sum_x2 - pow(sum_x, 2) / n) * sqrt(sum_y2 - pow(sum_y, 2) / n))
            if denominator == 0:
                pearson_sim.append(0)
            else:
                pearson_sim.append(float(sum_xy - (sum_x * sum_y) / n) / denominator)

        return (self.w1 * pearson_sim[0] + self.w2 * pearson_sim[1])

    def cosine(self, userid1, userid2):
        '''
        Calculate cosine similarity given two user ID
        The first value is product rating similarity
        The second value is demo variable similarity
        '''
        cosine_sim=[]
        for i in range(0,2):
            sum_xy = 0.0
            sum_x2 = 0.0
            sum_y2 = 0.0
            n = 0
            if i==1:
                rating1=self.data_ratings[userid1]
                rating2=self.data_ratings[userid2]
            else:
                rating1=self.data_demo[userid1]
                rating2=self.data_demo[userid2]       
        
            for key in rating1:
                x = rating1[key]
                sum_x2 += pow(x, 2)
                if key in rating2:
                    n += 1                
                    y = rating2[key]                
                    sum_xy += x * y
        
            for key in rating2:
                y=rating2[key]             
                sum_y2 += pow(y, 2)
            
            if n == 0:
                cosine_sim.append(0)
        # now compute denominator
            denominator = sqrt(sum_x2) * sqrt(sum_y2)
            if denominator == 0:
                cosine_sim.append(0)
            else:
                cosine_sim.append(float(sum_xy) / denominator)
        return (self.w1 * cosine_sim[0] +self.w2 * cosine_sim[1])


    def computeNearestNeighbor(self, userid):
        """creates a sorted list of users based on their distance to
        username
        
        Compute distance with weighted demoa and ratings datasets(w1,w2)
        """
        distances = []
        if self.metric=='cosine':            
            for user in self.data_ratings:
                if user != userid:
                    similarity = self.cosine(userid,user)
                    distances.append((user, similarity))
        
        if self.metric=='pearson':            
            for user in self.data_ratings:
                if user != userid:
                    similarity = self.pearson(userid,user)
                    distances.append((user, similarity))               
        
        # sort based on distance -- closest first
        distances.sort(key=lambda artistTuple: artistTuple[1],
                       reverse=True)
        #print (distances[0:n])
        return distances[:self.k]



    def recommend(self, userid):
		
        """Give list of recommendations"""
        recommendations = {}
        # first get list of users  ordered by nearness
        nearest = self.computeNearestNeighbor(userid)
        #
        # now get the ratings for the user
        #
        userRatings = self.data_ratings[userid]
        #
        # now iterate through the k nearest neighbors
        # accumulating their ratings
        for product in self.product:	   
		# determine the total distance
            totalDistance = 0.0
            if product not in userRatings:               
                for i in range(self.k):
                    if product in self.data_ratings[nearest[i][0]]: 
                      # compute slice of pie 
                        weight = nearest[i][1] 
                        # get the name of the person
                        name = nearest[i][0]
                        # get the ratings for this person
                        neighborRatings = self.data_ratings[name]
                        # get the name of the person
                        # now find bands neighbor rated that user didn't
				 #print ('product: %s, name %s , weight %f'%(product, name, weight))
                        totalDistance+=weight
                        if product not in recommendations:
					recommendations[product] = (neighborRatings[product] * weight)
                        else:
					recommendations[product] = (recommendations[product]  + neighborRatings[product] * weight)
                try:                   
                    recommendations[product]=recommendations[product] / totalDistance
                except:
                    pass
               
        
        # now make list from dictionary
       
       #self.data_ratings[userid].update(recommendations)
       
        recommendations = list(recommendations.items())
        recommendations = [(k, v) for (k, v) in recommendations]
        # finally sort and return
        recommendations.sort(key=lambda artistTuple: artistTuple[1],
                             reverse = True)
        # Return the first n items
        return recommendations