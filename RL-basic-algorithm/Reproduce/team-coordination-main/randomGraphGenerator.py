
import random
import numpy as np

class RandomGraph:
    def __init__(self, V):
        '''
        Instantiate the Environment Graph denoted by EG.
        '''
        self.V = V # vertices or nodes size
        self.R_E = 0 # total risky edges size
        self.total_risky_edges = 0 # total risky edges with repetation (0,1) and (1,0)
        self.total_edges = 0 # total edges including risky edges
        self.edge_cost = 10 # all edge have same edge cost, this can be modified
        self.nodes_list = [i for i in range(self.V)] # list of the vertices of nodes
        self.adj= [[0 for column in range(self.V)] for row in range(self.V)] 
        self.adj_m = [[float('inf') for column in range(self.V)] for row in range(self.V)] 
        self.visisted_risky_edges = [] 
        self.S11 = (None, None) # agents start position
        self.Sgg = (None, None) # agents goal position
         
  
    def printEGadjcencyMatrixWithRAndS(self):
        for row in self.adj_m:
            print(row) 
                
    def printEGadjcencyMatrixWithNoRAndS(self):
         for row in self.adj:
                print(row)  
                
    def getRGWithRiskEdgesAndSupportNodes(self):
        return self.adj_m
    
    def getRGWithNoRiskEdgesAndSupportNodes(self):
        return self.adj 
    
    def getTotalRiskyEdges(self): # unrepeated
            return self.R_E
      
     # Returns count of edge in undirected graph
    def countEdgesEG(self):
        Sum = 0
        for i in range(self.V):
            for j in range(self.V):
                if self.adj[i][j]!=0:
                    Sum = Sum+1
        self.total_edges = int(Sum/2)
        return self.total_edges
    # get environment graph edges 
    def getEnvironmentGraphEdges(self):  
        #print("Environment Graph Edges") 
        EG_Edges =  []        
        for i in range(self.V):
            for j in range(self.V):
                if self.adj[i][j]!=0:  
                      EG_Edges.append((i,j, self.adj[i][j]))
        return EG_Edges 
           
    def generateRandomGraph(self):
        #print("?????????????????????????", self.R_E)
        np.random.seed(111)
        adjacency_matrix11 = np.random.randint(0,2,(self.V,self.V))
        adjacency_matrix1 = np.tril(adjacency_matrix11) + np.tril(adjacency_matrix11, -1).T
        adjacency_matrix= adjacency_matrix1.tolist()
        #print(">>>>>>>>>>>>>>>>>>>>>>>>>>", self.R_E)
        for i in range(self.V):
            for j in range(self.V): 
                if i==j:
                    self.adj_m[i][j]=0
                    self.adj[i][j] = 0
                if i!=j and adjacency_matrix[i][j]==1:
                    self.adj_m[i][j]=[self.edge_cost,()] 
                    self.adj[i][j] = self.edge_cost 
    def generateRGWithSupportNodesAndRiskyEdges(self, risky_edges):
        self.R_E = risky_edges
        self.total_risky_edges = 2*self.R_E
        #print("**************************", self.R_E) 
        #print("adj_m", self.adj_m)   
        for i in range(self.V):
            for j in range(self.V):  
                if i!=j and self.adj_m[i][j]!=float('inf'): 
                    #print("------------------------") 
                    #print(self.visisted_risky_edges, self.total_risky_edges)
                    if len(self.visisted_risky_edges)<self.total_risky_edges:
                        #print("++++++++++++++++++++++++++++++++++") 
                        risky_edge = random.sample(self.nodes_list, 2)
                        support_nodes = random.sample(self.nodes_list, 2)
                        # print("------------------------")
                        # print("Visted Risky Edges: {}".format(self.visisted_risky_edges))
                        # print("Risky Edge: {}".format(risky_edge))
                        # print("Support Node: {}".format(support_nodes))
                        # print("------------------------")
                        if risky_edge not in self.visisted_risky_edges:
                            self.visisted_risky_edges.append(risky_edge)
                            self.visisted_risky_edges.append([risky_edge[1], risky_edge[0]])
                        
                            i,j = int(risky_edge[0]),int(risky_edge[1])
                            self.adj_m[i][j] = [self.edge_cost,(support_nodes[0], support_nodes[1])]
                            self.adj_m[j][i] = [self.edge_cost,(support_nodes[0], support_nodes[1])]
                            self.adj[i][j] = self.edge_cost 
                            self.adj[j][i] = self.edge_cost  
                            
    