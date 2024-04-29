
from collections import defaultdict
import itertools
from vanillaDijkstra import Dijkstra

class CJSGraph:
    def __init__(self, V):
        '''
        Instantiate CJSG Graph with the number of vertices of the graph.
        '''
        # EG variables
        self.V = V # nodes/vertices
        self.S11 = (None,None) # start position
        self.Sgg = (None,None) # goal position
        self.adj =  [[0 for column in range(self.V)] for row in
                      range(self.V)] # adj with edge cost and support nodes
        self.adj_ns = [[0 for column in range(self.V)] for row in
                      range(self.V)] # adj with edge cost only
        
        # CJSG variables 
        self.M = set() #  CJSG nodesets
        self.H = defaultdict(list) # dict of CJSG nodesets and edgesets
        self.graph = [[[] for column in range(V)] for row in
                      range(V)] # used adjMatrix for Crtical Joint State Graph (CJSG)
        
        # Temoorary variables
        self.ESS = set() # add riky edges along with support nodes 
        
    # return adjMatrix of CJSG
    def get_adjMatrix_2D_CJSG(self):
        return self.graph
    
    # return node sets of CJSG
    def get_nodeSets_CJSG(self):
        return self.M
        
    # add critical nodesets for CJSG
    def add_critical_nodesets(self):
        for i in range(self.V):
            for j in range(self.V):
                if self.adj[i][j]!=float('inf') and self.adj[i][j]!=0 and self.adj[i][j][1]!=():
                    #print((i,j),adj[i][j])
                    # calculate node sets for CJSG   
                    ES = (i,j) # risky edge with support nodes
                    Sij=self.adj[i][j][1] # support nodes 
                    self.ESS.add(ES)
                    print(ES, Sij)
                    for u in ES:
                        for v in Sij:
                            self.M.add((u,v))
                            self.M.add((v,u))  
        self.add_start_and_goal()     
                      
    # add start and goal agents position if not present in the node sets
    def add_start_and_goal(self): 
        if self.S11 not in self.M:
            self.M.add(self.S11)
        if self.Sgg not in self.M:
            self.M.add(self.Sgg) 
        print("ESS set {}",format(self.ESS))
        
    # calculate edge sets for CJSG
    # add adj list of CJSG for node sets with edge cost
    # (u,v) one node, (x,y) another node
    def addEdge_H(self,u,v,x,y,w):
        
        '''
        The Graph will be bidirectional and assumed to have positive weights.
        '''
        # this works for adjList
        if [w,(x,y)] not in self.H[(u,v)] and w!= float('inf'):
            self.H[(u,v)].append([w,(x,y)])  
        if [w,(u,v)] not in self.H[(x,y)] and w!= float('inf'):
            self.H[(x,y)].append([w,(u,v)])
            
        # this works for adjMatrix 
        if [w,(x,y)] not in self.graph[u][v]:
            self.graph[u][v].append([w,(x,y)])
        if [w,(u,v)] not in self.graph[x][y]:
            self.graph[x][y].append([w,(u,v)])
    # agents movement without support p->r, q->l
    def edge_cost(self,p,r,q,l,support=False):
        Rpr = 0
        Rql = 0
        if self.adj[p][r]!=float('inf') and self.adj[p][r]!=0:
            Rpr = self.adj[p][r][0]
        else:
            Rpr = self.adj[p][r]
        
        if self.adj[q][l]!=float('inf') and self.adj[q][l]!=0:
            Rql = self.adj[q][l][0]
        else:
            Rql = self.adj[q][l] 
        R_result = Rpr+Rql
        if (support):
            return  R_result/2      
        return R_result
    # convert environt Graph to Critical Joint State Graph
    def construct_CJSG(self):
        count = 1    
        for x in self.M:
            print("Iteration: {}".format(count))
            for y in self.M:
                if x!=y: # assumption no self loop
                    # x: pq and y = rl so x[0]y[0]: pr and x[1]y[1]: ql
                    p,q = x[0],x[1]
                    r,l = y[0],y[1]
                    
                    # edge cost without support
                    Rpq_rl = self.edge_cost(p,r,q,l)
                    # edge ql in ES and p==r and belongs to Sql (support nodes for the risky edge ql)
                    if (q,l) in self.ESS and p==r and p in self.adj[q][l][1]: # Sij
                        print((p,q),(r,l))
                        print("QL in ES !!!!!!!!!") 
                        # edge cost with support
                        Cql_s = self.edge_cost(p,r,q,l, support=True)
                        w = min(Cql_s,Rpq_rl)
                        self.addEdge_H(p,q,r,l, w) 
                    # edge pr in ES and q==l and belongs to Spr (support nodes for the risky edge pr)
                    elif (p,r) in self.ESS and q==l and q in self.adj[p][r][1]:# Sij
                        print((p,q),(r,l))
                        Cpr_s = self.edge_cost(p,r,q,l, support=True)
                        w = min(Cpr_s,Rpq_rl)
                        print(w)
                        print("PR in ES !!!!!!!!!") 
                        self.addEdge_H(p,q,r,l, w) 
                    else:
                        print((p,q),(r,l))
                        w = Rpq_rl
                        if w!=float('inf'):
                            print("Yay!!!!!",w)
                        self.addEdge_H(p,q,r,l, w) 
            
            count=count+1
            
    # shortest path between nodes in EG        
    def vanilla_dijkstra(self,p,r,q,l):
        g = Dijkstra(self.V) 
        g.graph = self.adj_ns
        Rpr = g.dijkstra(src=p,dest=r)
        Rql = g.dijkstra(src=q,dest=l)
        return Rpr+Rql
    
    # add cjsg edgesets 
    def add_cjsg_edgesets(self): 
        for nodex in sorted(self.M):  
            for nodey in sorted(self.M):
                if nodex!= nodey:
                    present_next_nodes = [next_node for weight, next_node  in sorted(self.H[nodex])]
                    print(nodex, nodey, present_next_nodes)
                    if nodey not in present_next_nodes:
                        print(nodey)
                        p,q = nodex[0],nodex[1]
                        r,l = nodey[0],nodey[1]
                        # Rpr+Rql
                        print(p,r,q,l)
                        Rpq_rl = self.vanilla_dijkstra(p,r,q,l)
                        #Rpq_rl = self.edge_cost(p,r,q,l)
                        self.addEdge_H(p,q,r,l, w=Rpq_rl) 
                        
    # def add_remaning_cjsg_nodesets(self): 
    #     for nodex, nodey in itertools.product(sorted(self.M), repeat=2):
    #         if nodex != nodey:
    #             u, v = nodex
    #             present_next_nodes = [next_node for weight, next_node in self.graph[u][v]]
                
    #             if nodey not in present_next_nodes:
    #                 p, q = nodex[0], nodex[1]
    #                 r, l = nodey[0], nodey[1]
                    
    #                 Rpq_rl = self.vanilla_dijkstra(p, r, q, l)
    #                 self.addEdge_H(p, q, r, l, w=Rpq_rl)
    
    # transform Environment Grapoh to Critical Joint State Graph 
    def transform_EG_to_CJSG(self):
        self.add_critical_nodesets()
        self.construct_CJSG()
        self.add_cjsg_edgesets()
        
                            
    