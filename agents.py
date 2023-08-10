import random
import math


BOT_NAME = "Spooky Scary Stegosaurus" #+ 19 


class RandomAgent:
    """Agent that picks a random available move.  You should be able to beat it."""
    def __init__(self, sd=None):
        if sd is None:
            self.st = None
        else:
            random.seed(sd)
            self.st = random.getstate()

    def get_move(self, state):
        if self.st is not None:
            random.setstate(self.st)
        return random.choice(state.successors())


class HumanAgent:
    """Prompts user to supply a valid move."""
    def get_move(self, state, depth=None):
        move__state = dict(state.successors())
        prompt = "Kindly enter your move {}: ".format(sorted(move__state.keys()))
        move = None
        while move not in move__state:
            try:
                move = int(input(prompt))
            except ValueError:
                continue
        return move, move__state[move]


class MinimaxAgent:
    """Artificially intelligent agent that uses minimax to optimally select the best move."""

    def get_move(self, state):
        """Select the best available move, based on minimax value."""
        nextp = state.next_player()
        best_util = -math.inf if nextp == 1 else math.inf
        best_move = None
        best_state = None

        for move, state in state.successors():
            util = self.minimax(state)
            if ((nextp == 1) and (util > best_util)) or ((nextp == -1) and (util < best_util)):
                best_util, best_move, best_state = util, move, state
        return best_move, best_state

    def max_value(self, state):
        if(state.is_full()):
            return state.utility()
        
        v = -math.inf
        for s in state.successors():
            v = max(v, self.min_value(s[1]))
        return v
    
    def min_value(self, state):
        if(state.is_full()):
            return state.utility()
        
        v = math.inf
        for s in state.successors():
            v = min(v, self.max_value(s[1]))
        return v
    
    def minimax(self, state):
        """Determine the minimax utility value of the given state.

        Args:
            state: a connect383.GameState object representing the current board

        Returns: the exact minimax utility value of the state
        """
      
        v = self.max_value(state)

        return v  


class MinimaxHeuristicAgent(MinimaxAgent):
    """Artificially intelligent agent that uses depth-limited minimax to select the best move."""

    def __init__(self, depth_limit):
        self.depth_limit = depth_limit

    def max_value(self, state, depth_limit):
        if(state.is_full()):
            return state.utility()
        if(depth_limit <= 0):
            return self.evaluation(state)

        v = -math.inf
        for s in state.successors():
            v = max(v, self.min_value(s[1], depth_limit - 1))
        depth_limit -= 1
        return v
    
    def min_value(self, state, depth_limit):
        if(state.is_full()):
            return state.utility()
        if(depth_limit <= 0):
            return self.evaluation(state)
        
        v = math.inf
        for s in state.successors():
            v = min(v, self.max_value(s[1], depth_limit - 1))
        return v
    
    def minimax(self, state):
        """Determine the heuristically estimated minimax utility value of the given state.

        The depth data member (set in the constructor) determines the maximum depth of the game 
        tree that gets explored before estimating the state utilities using the evaluation() 
        function.  If depth is 0, no traversal is performed, and minimax returns the results of 
        a call to evaluation().  If depth is None, the entire game tree is traversed.

        Args:
            state: a connect383.GameState object representing the current board

        Returns: the minimax utility value of the state
        """
       
        v = self.max_value(state, self.depth_limit)
        return v 

    def evaluation(self, state):
        """Estimate the utility value of the game state based on features.

        N.B.: This method must run in constant time for all states!

        Args:
            state: a connect383.GameState object representing the current board

        Returns: a heuristic estimate of the utility value of the state
        """
        #
        # Fill this in!
        #

        #check rows, see which player has access to 2 or more open spaces
        score1, score2 = state.scores()
        val = score1 + score2
        for r in state.get_rows():
            # x is element before spaces
            x = r[0]
            zeroes = 0
            for n in range(0, len(r)):
                if(r[n] == 0):
                    #space
                    zeroes += 1
                elif(zeroes >= 2):
                    if(x != 0):
                        val += x
                    if(n + 1 < len(r)):
                        val += r[n + 1]
                    x = r[n]
                else:
                    x = r[n]
        for c in state.get_cols():
            # x is element before spaces
            x = c[0]
            zeroes = 0
            for n in range(0, len(c)):
                if (zeroes >= 2 and x != 0):
                    val += x
                    break
                
                if(c[n] == 0):
                    #space
                    zeroes += 1
                else:
                    x = c[n]
        for d in state.get_diags():
            # x is element before spaces
            x = d[0]
            zeroes = 0
            for n in range(0, len(d)):
                if(d[n] == 0):
                    #space
                    zeroes += 1
                elif(zeroes >= 2):
                    if(x != 0):
                        val += x
                    if(n + 1 < len(d)):
                        val += d[n + 1]
                    x = d[n]
                else:
                    x = d[n]
        return val # Change this line!


class MinimaxPruneAgent(MinimaxAgent):
    """Smarter computer agent that uses minimax with alpha-beta pruning to select the best move."""
    def max_value(self, state, alpha, beta):
        if(state.is_full()):
            return state.utility()
        
        v = -math.inf
        for s in state.successors():
            v = max(v, self.min_value(s[1], alpha, beta))
            if(v >= beta):
                return v
            alpha = max(alpha, v)
        return v
    
    def min_value(self, state, alpha, beta):
        if(state.is_full()):
            return state.utility()
        
        v = math.inf
        for s in state.successors():
            v = min(v, self.max_value(s[1], alpha, beta))
            if(v <= alpha):
                return v
            beta = min(beta, v)
        return v
    

    def minimax(self, state):
        """Determine the minimax utility value the given state using alpha-beta pruning.

        The value should be equal to the one determined by MinimaxAgent.minimax(), but the 
        algorithm should do less work.  You can check this by inspecting the value of the class 
        variable GameState.state_count, which keeps track of how many GameState objects have been 
        created over time.  This agent does not use a depth limit like MinimaxHeuristicAgent.

        N.B.: When exploring the game tree and expanding nodes, you must consider the child nodes
        in the order that they are returned by GameState.successors().  That is, you cannot prune
        the state reached by moving to column 4 before you've explored the state reached by a move
        to to column 1.

        Args: 
            state: a connect383.GameState object representing the current board

        Returns: the minimax utility value of the state
        """

        v = self.max_value(state, -math.inf, math.inf)
        return v  

# N.B.: The following class is provided for convenience only; you do not need to implement it!

class OtherMinimaxHeuristicAgent(MinimaxAgent):
    """Alternative heursitic agent used for testing."""

    def __init__(self, depth_limit):
        self.depth_limit = depth_limit

    def minimax(self, state):
        """Determine the heuristically estimated minimax utility value of the given state."""
        #
        # Fill this in, if it pleases you.
        #
        return 26  # Change this line, unless you have something better to do.

