
#-------------------------------------------------------------------------
import numpy as np
import random
import time
import queue
import argparse
import csv
#-------------------------------------------------------------------------

#-------------------------------------------------------
''' 
    Utility Functions
'''
#-------------------------------
def obtain_heuristic(state, H):
    '''
    Obttain either of the following heuristic functions:
        H1.  The square of the lightest Queen across all pairs of Queens attacking each other.  Moving that Queen is the minimum possible cost to solve the problem (in real life it will probably take more effort than that though).
        H2.  Sum across every pair of attacking Queens the squared weight of the lightest Queen.
    
    Input:
        state: numpy array of n by n
        H: type of heuristic function to be applied (either H1 or H2)
    Output:
        result: valoration of the state based on the heuristic funcion
    
    *** It is assumed that the state matrix does not have multiple values per column
    '''
    #First assert the validity of the input
    assert H=='H1' or H=='H2' 

    #Loop to check rows, columns and diagonals from left to right
    result = []
    n = state.shape[0]
    
    '''Add results of rows'''
    for i in range(n):
        if len( np.delete(state[i], np.where(state[i] == 0)) ) > 1:
            result.append( np.delete(state[i], np.where(state[i] == 0)) )

    '''Add results of diagonals from top left to bottom right'''
    for i in range(-(n-2), (n-1)):
        aux = np.diagonal(state, offset=i)

        if len( np.delete(aux, np.where(aux == 0)) ) > 1:
            result.append( np.delete(aux, np.where(aux == 0)) )
    
    '''Add results of diagonals from top right to bottom left'''
    for i in range(-(n-2), (n-1)):
        aux = np.diagonal(np.fliplr(state), offset=i)

        if len( np.delete(aux, np.where(aux == 0)) ) > 1:
            result.append( np.delete(aux, np.where(aux == 0)) )
    
    #print("result:",result)
    #Output dependending on the heuristic
    if len(result) == 0:
        result = 0
    elif H=="H1":
        mini = result[0][0]
        for i in result:
            if min(i) < mini: 
                mini = min(i) 
        result = mini**2
    else:
        result_ = 0
        for i in result:
            i = np.sort(i)
            n = len(i) - 1
            for j in i:
                result_ += (j**2)*n
                n -= 1
            result = result_

    return result

#-------------------------------------------------------
''' 
    Implementation of the Board with n queens
'''
#-------------------------------------------------------

class Board:
    '''
        List of Attributes: 
            s: the current state of the game, a numpy array of shape n by n. 
                s[i,j] = 0 denotes that the i-th row and j-th column is empty.
                s[i,j] = 1 denotes that the i-th row and j-th column is taken by a queen. 
    
            p: the parent node of this node 
            m: the move that it takes from the parent node to reach this node.  m is a tuple (r,c), r:row of move, c:column of the move. 
            c: a python list of all the children nodes of this node 
            x: the number of nodes expanded. Consider an expansion as when you visit a node and compute the cost for all of the possible next moves.
            l: the length of the solution path
            t: the cost to solve the puzzle
    '''
    #-------------------------------
    def __init__(self, s, H, p=None,m=None, c=None, x=None, g_c=None):
        self.s = s.copy() # the current state of the game
        self.H = H # the heuristic function to be used
        self.p = p # the parent node of the current node
        self.m = m # the move that it takes from the parent node to reach this node. 
        self.c = [] # a list of children nodes
        self.x = x # nodes expanded
        self.t = 0 # cost to solve the board
        self.g_c = g_c # greedy value of best children
        self.g_a = obtain_heuristic(state = self.s, H = self.H) # heuristic value actual state
        self.l = 0 # length of solution path 
        self.m_t = [] # the moves to reach this node. 
        self.time = time.time()
        self.astar = self.t + self.g_a 
    #-------------------------------
    def __lt__(self, other):
        return self.astar<other.astar
    #-------------------------------
    def __eq__(self,other):
        if isinstance(other, self.__class__):
            return  np.array_equal(self.s,other.s)
        else:
            return False
    #-------------------------------
    def __hash__(self):
        return hash(self.astar)
    # ----------------------------------------------
    def create_child_astar(self, state, move):
        '''  
            Create a child board with the carachteristics of the parent.
        Input:
            state: a new board configuration to be appended as a child
        Output:
            child: a Board class object
        '''
        child = Board(s=state, H=self.H)
        child.p = self # the parent node of the current node
        #child.m = move # the move that it takes from the parent node to reach this node. 
        #child.c = [] # a list of children nodes
        #child.x = x # nodes expanded
        child.t = self.t + abs(int(np.where(self.s.T[move[1]] > 0)[0]) - move[0])*np.sum(state, axis=0)[move[1]]**2 
        #+ abs(np.where(self.s.T[move[1]] > 0)[0][0] - move[0])*np.sum(state, axis=0)[move[1]]**2 # cost to move the board
        #child.g_c = g_c # greedy value of best children
        child.l = self.l + 1 # length of solution path 
        child.m_t = self.m_t + [move] # the moves to reach this node.
        child.time = self.time 
        child.astar = child.t + child.g_a 


        return child

    # ----------------------------------------------
    def expand_astar(self):
        '''  
            Expand the current tree node by adding one layer of children.
            Add a collection of the best children
        Input:
            self: the node to be expanded
        Output:
            result_value: a tuple of the best value of children, and all the configurations that apply with this resolution
        '''
        output = []

        #Define all possible spaces 
        x_0 , y_0 = np.where(self.s==0)
        #x ,   y   = np.where(self.s>0)
        
        weights = np.sum(self.s, axis=0)

        #Copy of the actual board
        #child_state = self.s.copy() 

        #Begin the creation of children nodes
        for i in range(len(x_0)):
            
            child_state = self.s.copy()
            #Define the column equal to 0
            child_state.T[y_0[i]] *= 0
            #Replace value
            child_state[(x_0[i], y_0[i])] = weights[y_0[i]] 
            child = self.create_child_astar(child_state, (x_0[i], y_0[i]))

            #Append to output
            if child != self.p:
                output.append(child)
        
        return output


class Board_gs:
    '''
        List of Attributes: 
            s: the current state of the game, a numpy array of shape n by n. 
                s[i,j] = 0 denotes that the i-th row and j-th column is empty.
                s[i,j] = 1 denotes that the i-th row and j-th column is taken by a queen. 
    
            p: the parent node of this node 
            m: the move that it takes from the parent node to reach this node.  m is a tuple (r,c), r:row of move, c:column of the move. 
            c: a python list of all the children nodes of this node 
            x: the number of nodes expanded. Consider an expansion as when you visit a node and compute the cost for all of the possible next moves.
            l: the length of the solution path
            t: the cost to solve the puzzle
    '''
    #-------------------------------
    def __init__(self, s, H, p=None,m=None, c=None, x=None, g_c=None):
        self.s = s.copy() # the current state of the game
        self.H = H # the heuristic function to be used
        self.p = p # the parent node of the current node
        self.m = m # the move that it takes from the parent node to reach this node. 
        self.c = [] # a list of children nodes
        self.x = 0 # nodes expanded
        self.t = 0 # cost to solve the board
        self.g_c = g_c # greedy value of best children
        self.g_a = obtain_heuristic(state = self.s, H = self.H) # greedy value actual state
        self.l = 0 # length of solution path 
        self.m_t = [] # the moves to reach this node. 
        self.time = time.time()
        self.start = self.s
    
    # ----------------------------------------------
    def create_child(self, state, move):
        '''  
            Create a child board with the carachteristics of the parent.
        Input:
            state: a new board configuration to be appended as a child
        Output:
            child: a Board class object
        '''
        child = Board_gs(s=state, H=self.H)
        child.p = self # the parent node of the current node
        child.m = move # the move that it takes from the parent node to reach this node. 
        #child.c = [] # a list of children nodes
        child.x = self.x + len(self.c) # nodes expanded
        child.t = self.t + abs(int(np.where(self.s.T[move[1]] > 0)[0]) - move[0])*np.sum(state, axis=0)[move[1]]**2
        #+ abs(np.where(self.s.T[move[1]] > 0)[0][0] - move[0])*np.sum(state, axis=0)[move[1]]**2 # cost to move the board
        #child.g_c = g_c # greedy value of best children
        child.g_a = obtain_heuristic(state = self.s, H = self.H) # greedy value actual state
        child.l = self.l + 1 # length of solution path 
        child.m_t = self.m_t + [move] # the moves to reach this node.
        child.time = self.time 
        child.start = self.s

        return child

    # ----------------------------------------------
    def expand(self):
        '''  
            Expand the current tree node by adding one layer of children.
            Add a collection of the best children
        Input:
            self: the node to be expanded
        Output:
            result_value: a tuple of the best value of children, and all the configurations that apply with this resolution
        '''
    
        #Define possible outputs and values of columns
        result_value = {}
        values = np.sum(self.s, axis=0)
        child_state = self.s.copy() 

        #Create the tuples of the coornitates to be used
        moves = np.where(child_state==0)
        loc = random.sample( range(len(moves[0])) , int(len(moves[0])*0.5))

        #Define a test case to keep track of the best children. First value equal to zero at the first column of state
        child_state.T[moves[1][loc[0]]] *= 0
        child_state[(moves[0][loc[0]], moves[1][loc[0]])] = values[moves[1][loc[0]]]

        '''Define the minimum value'''
        mini = obtain_heuristic(child_state, self.H)
        child_out = child_state.copy()
        child_out = self.create_child(child_out, (moves[0][loc[0]], moves[1][loc[0]])) 
        result_value[mini] = [child_out]

        #Begin a loop for all the columns
        for x_y in loc[1:]:
            
            #Generate another matrix subject to be modified
            child_state = self.s.copy()

            #Loop for all the available spaces at a given column
            child_state.T[moves[1][x_y]] *= 0
            child_state[(moves[0][x_y], moves[1][x_y])] = values[moves[1][x_y]]

            if obtain_heuristic(child_state, self.H) == mini:
                child_out = self.create_child(child_state, (moves[0][x_y], moves[1][x_y]))
                result_value[mini].append(child_out)

            elif obtain_heuristic(child_state, self.H) < mini:
                result_value = {}
                mini = obtain_heuristic(child_state, self.H)
                child_out = self.create_child(child_state, (moves[0][x_y], moves[1][x_y])) 
                result_value[mini] = [child_out]
            else:
                pass

        self.c = result_value[mini]
        self.g_c = mini
    
    # ----------------------------------------------
    def build_tree(self):
        '''
        Build greedy tree        
        '''
        #If conditions to expand are met, expand
        if time.time() - self.time < 10:
            self.expand()

            #If expansion is successful, end
            if self.g_c == 0:
                print('Start state:')
                print(self.start)
                print("\n")
                print("Number of nodes expanded:", self.c[0].x)
                
                a = time.time() - self.time
                print("Time to solve puzzle:", "%.2f" % a)
                
                b = self.c[0].x/(len(self.c[0].m_t) - 1)
                print("Effective branching factor:", "%.2f" % b)
                print("Cost:", self.c[0].t)
                print('Sequence of moves:', self.c[0].m_t, "\n")
                #print('The solution is:', "\n")
                #print(self.c[0].s, "\n")
                return (a, self.c[0].t)
                
            #If len(solution is too big, restart again)
            elif self.l >= len(self.s)*1.25:
                #print("No solution found with ", self.c[0].l, " steps")
                #print("Restart")

                child = self
                nodes = self.x
                while child.p != None:
                    child = child.p
                child.x = nodes
                #print("Restart now")
                return child.build_tree()    

            #Elect the best node to apply recursion
            else:
                for boards in self.c:
                    boards.expand()
                
                election = self.c[0]
                for boards in self.c:
                    if boards.g_c < election.g_c:
                        election = boards
                
                #apply recursion
                return election.build_tree()
        else:
            print("No solution found in 10 seconds")
            a = time.time() - self.time
            #print("Time to solve puzzle:", "%.2f" % a)
            #return self.c[0].x/(len(self.c[0].m_t) - 1)
            return (-10, -10)

def build(s, H):
    q=queue.PriorityQueue() # a PriorityQueue of the nodes that need to be expand
    b=Board(s, H)
    q.put(b)
    s=set()
    s.add(b)
    sum=1
    start_time=time.time()
    while not q.empty():
        current=q.get()
        if (time.time()-start_time)>10:
            print("No solution found in 10 seconds")
            
            break
        elif current.g_a==0: #if heuristic value=0, succeed
            print("Start state:")
            print(b.s)
            print("Number of nodes expanded:", sum)
            a=time.time()-start_time
            print("Time to solve the puzzle:", "%.2f" % a)
            branch=sum/len(current.m_t)
            print("Effective branching factor:", "%.2f" % branch)
            print("Cost:", current.t)
            print("Sequence of moves:",current.m_t)
            print("The solution:")
            print(current.s)
            break
        else:
            child=current.expand_astar()
            for i in child:
                if i not in s:
                    q.put(i)
                    sum+=1
                    s.add(i)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', metavar='F', type=str, nargs=1, help='a csv file, the name of the file cannot contain space')
    parser.add_argument('algorithm', metavar='A', type=str, nargs=1, help='A* or Greedy search')
    parser.add_argument('heuristic', metavar='H', type=str, nargs=1, help='Heuristic function H1 or H2')
    args = parser.parse_args()
    
    #Generate board
    with open(args.file[0], 'r',encoding='UTF-8-sig') as f:
        temp=list(csv.reader(f))
        result=np.array(temp)
        
    c=np.nonzero(result)
    n=len(c[0])
    state= np.zeros((n,n))
    state[c]=result[c]
    print(state)
    fun=int(args.algorithm[0])
    hf=args.heuristic[0]

    if fun==1:
        build(state, hf)
    elif fun==2:
        b=Board_gs(state, hf)
        b.build_tree()
    else:
        print("wrong input")