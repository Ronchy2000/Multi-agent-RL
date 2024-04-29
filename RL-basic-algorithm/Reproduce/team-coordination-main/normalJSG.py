from collections import defaultdict
import sys

class JSGraph():
    def __init__(self, V):
        '''
        Instantiate JSG Graph with the number of vertices of the graph.
        '''
        # EG variables
        self.V = V # vertices/nodes
        self.source = (None, None) # start position
        self.destination = (None, None) # goal position
        self.graph_1d = [[0 for column in range(V)] for row in
                      range(V)] # used adjM for Environment Graph (EG)
          
        # JSG Variables
        self.nodesets = [] # nodesets for JSG
        self.graph = [[[] for column in range(V)] for row in
                      range(V)] # used adjMatrix for Joint State Graph (JSG)
        
    # get adjMatrix of JSG
    def get_adjMatrix_2D_JSG(self):
        return self.graph
  
    # add adjMatrix of joint state graph
    def addEdge_2d(self, u, v,x,y, w):
        '''
        The Graph will be bidirectional and assumed to have positive weights.
        '''
        # # this works for adjList
        # if [w,(x,y)] not in self.graph_2d[(u,v)]:
        #     self.graph_2d[(u,v)].append([w,(x,y)])
        # if [w,(u,v)] not in self.graph_2d[(x,y)]:
        #     self.graph_2d[(x,y)].append([w,(u,v)])  
        
        # this works for adjMatrix 
        if [w,(x,y)] not in self.graph[u][v]:
            self.graph[u][v].append([w,(x,y)])
        if [w,(u,v)] not in self.graph[x][y]:
            self.graph[x][y].append([w,(u,v)])
    
    # add modes of JSG
    def add_nodesets(self):
        
        for u in range(self.V):
            for v in range(self.V):
                if (u,v) not in self.nodesets:
                    self.nodesets.append((u,v))
        return self.nodesets
    # calcualte edge cost
    def edge_cost(self,i,j,w,k,support=False):
        dist = 0
        
        if type(self.graph_1d[i][w])==list:
            dist_iw,_= self.graph_1d[i][w]
        else:
            dist_iw = self.graph_1d[i][w]     
                  
        if type(self.graph_1d[j][k])==list:
            dist_jk,_ = self.graph_1d[j][k]
        else:
            dist_jk = self.graph_1d[j][k] 
            
        dist = dist_iw + dist_jk
        if dist_iw == float('inf') or dist_jk==float('inf'):
            return float('inf')
        if (support):
            return  int(dist/2)    # if has support edge cost reduced by half   
        return dist  
      
 
    # convert Environment Graph to Joint State Space Graph    
    def construct_JSG(self):
        for (i,j) in self.nodesets:
            for (w,k) in self.nodesets:
                #print((i,j),(w,k))
                #A constant B moving (u,v)-(i,j) and (x,y)-(w,k)
                if i==w and j!=k and self.graph_1d[j][k]!=0 and self.graph_1d[j][k]!=float('inf') : # k belongs to N of j
                    support_node_jk = []
                    if type(self.graph_1d[j][k])==list:
                        _,support_node_jk = self.graph_1d[j][k]
                        
                    #print("??????????????????????????????????",support_node_jk )   
                    if i in support_node_jk:
                        dist = self.edge_cost(i,j,w,k,support=True)
                    else:
                        dist = self.edge_cost(i,j,w,k)
                    ############################
                    if dist!=0 and dist!=float('inf'): 
                        print("A Constant B moving")
                        print((i,w),(j,k),dist)  
                        self.addEdge_2d(i,j,w,k,dist)
                # B constant, A moving
                elif i!=w and j==k and self.graph_1d[w][i]!=0 and self.graph_1d[w][i]!=float('inf') :
                    support_node_iw = []
                    if type(self.graph_1d[i][w])==list:
                        _,support_node_iw= self.graph_1d[i][w]  
                           
                    if j in support_node_iw:
                        dist = self.edge_cost(i,j,w,k,support=True)
                    else:
                        dist = self.edge_cost(i,j,w,k)
                    ############################### 
                    if dist!=0 and dist!=float('inf'):  
                        print("B Constant A moving")
                        print((i,w),(j,k),dist)   
                        self.addEdge_2d(i,j,w,k,dist)  
                elif self.graph_1d[j][k]!=0 and self.graph_1d[j][k]!=float('inf') and self.graph_1d[i][w]!=0 and self.graph_1d[i][w]!=float('inf') :   
                    dist= self.edge_cost(i,j,w,k)
                    ############################
                    if dist!=0 and dist!=float('inf'): 
                        print("Both moving: from privious node to next node")
                        print((i,w),(j,k),dist) 
                        self.addEdge_2d(i,j,w,k,dist)  
                        
                        
    # transofrm Environment Graph to Joint State Space Graph                    
    def trans_Env_To_JSG(self):
        self.add_nodesets()
        self.construct_JSG()               
        
    