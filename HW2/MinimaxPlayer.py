import time as T
import math
MAX_UTILITY = 500
MIN_UTILITY = -500
FIRST_PLAYER = 1
SECOND_PLAYER = 2

class MinimaxPlayer:
    def __init__(self):
        self.loc = None
        self.board = None
        self.directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        self.make_move_flag = False
        self.set_rival_move_flag = False
        self.we_played_first = False


    def set_game_params(self, board):
        self.board = board
        for i, row in enumerate(board):
            for j, val in enumerate(row):
                if val == FIRST_PLAYER:
                    self.loc = (i, j)
                    break

    def make_move(self, time) -> (tuple):
        if not self.set_rival_move_flag:
            self.we_played_first = True
            self.make_move_flag = True
        ID_start_time = T.time()
        depth = 1

        move, numOfNodes, value = self.Minimax(self.board, depth, FIRST_PLAYER, self.loc)
        x = move[0] - self.loc[0]
        y = move[1] - self.loc[1]
        last_iteration_time = T.time() - ID_start_time
        next_iteration_time = self.time_bound(numOfNodes, last_iteration_time, depth)
        time_until_now = T.time() - ID_start_time

        while time_until_now + next_iteration_time < time:
            depth += 1
            iteration_start_time = T.time()
            move, numOfNodes, value = self.Minimax(self.board, depth, FIRST_PLAYER, self.loc)
            x = move[0] - self.loc[0]
            y = move[1] - self.loc[1]
            last_iteration_time = T.time() - iteration_start_time
            next_iteration_time = self.time_bound(numOfNodes, last_iteration_time, depth)
            time_until_now = T.time() - ID_start_time

        if move == self.loc:
            list_of_neighbors = self.succ(self.board, self.loc)
            move = list_of_neighbors[0]
            x = move[0] - self.loc[0]
            y = move[1] - self.loc[1]

        self.board[self.loc] = -1
        self.board[move] = FIRST_PLAYER
        self.loc = move

        return x, y


    def set_rival_move(self, loc):
        if not self.make_move_flag:
            self.set_rival_move_flag = True
        self.board[self.getLoc(self.board, SECOND_PLAYER)] = -1
        self.board[loc] = SECOND_PLAYER


    def Minimax_heuristic(self, board, loc, agentTurn):
        flag, res = self.is_final(board, agentTurn)
        if flag:
            return res
        num_steps_available = 0
        for d in self.directions:
            i = loc[0] + d[0]
            j = loc[1] + d[1]
            if 0 <= i < len(board) and 0 <= j < len(board[0]) and board[i][j] == 0:  # then move is legal
                num_steps_available += 1

        if num_steps_available == 0:
            return -1
        else:
            return 4 - num_steps_available

    def New_heuristic(self, board, loc, agentTurn):
            flag, res = self.is_final(board, agentTurn)
            if flag:
                return res
            board_factor = min(len(board), len(board[0]))
            board_factor = math.ceil(board_factor / 3)
            return (3 * self.CalcDistanceToRival(board, loc) + board_factor * self.CalcWhiteNeighbors(board, loc) -
                    self.CalcMinDistanceToFrame(board, loc))




    def CalcDistanceToRival(self,board, onesLocation):
        rivalLocation = self.getLoc(board, SECOND_PLAYER)
        xDist = abs(onesLocation[0] - rivalLocation[0])
        yDist = abs(onesLocation[1] - rivalLocation[1])
        return xDist + yDist

    def CalcMinDistanceToFrame(self, board, onesLocation):
        rowsDimentions = len(board) - 1
        colsDimentions = len(board[0]) - 1

        xDist = min(rowsDimentions - onesLocation[0], onesLocation[0])
        yDist = min(colsDimentions - onesLocation[1], onesLocation[1])
        return min( xDist, yDist)

    def CalcWhiteNeighbors(self,board, onesLocation):
        num_steps_available = 0
        for d in self.directions:
            i = onesLocation[0] + d[0]
            j = onesLocation[1] + d[1]
            if 0 <= i < len(board) and 0 <= j < len(board[0]) and board[i][j] == 0:  # then move is legal
                num_steps_available += 1
        return num_steps_available

    def Minimax(self, board, depth: int, agent: int, loc: tuple) -> (tuple, int, int):
        if depth == 0:
            return loc, 0, self.New_heuristic(board, self.getLoc(board, FIRST_PLAYER), agent)

        CurNumOfNodes = 0

        isFinal, Utility = self.is_final(board, agent)
        if isFinal:
            return loc, 1, Utility

        if agent == FIRST_PLAYER:
            agent_loc = self.getLoc(board, agent)
            list_of_neighbors = self.succ(board, agent_loc)
            CurMax = float('-inf')
            CurMaxLoc = None
            CurNumOfNodes = len(list_of_neighbors)
            for child in list_of_neighbors:
                temp_board = board.copy()
                temp_board[agent_loc] = -1
                temp_board[child] = FIRST_PLAYER
                res_loc, res_num_of_nodes, res_value = self.Minimax(temp_board, depth - 1, SECOND_PLAYER, child)
                if CurMax < res_value:
                    CurMax = res_value
                    CurMaxLoc = child
                    CurNumOfNodes += res_num_of_nodes
            return CurMaxLoc, CurNumOfNodes, CurMax

        else:  # Agent == 2
            agent_loc = self.getLoc(board, agent)
            list_of_neighbors = self.succ(board, agent_loc)
            CurMin = float('inf')
            CurMinLoc = None
            CurNumOfNodes = len(list_of_neighbors)
            for child in list_of_neighbors:
                temp_board = board.copy()
                temp_board[agent_loc] = -1
                temp_board[child] = SECOND_PLAYER
                res_loc, res_num_of_nodes, res_value = self.Minimax(temp_board, depth - 1, FIRST_PLAYER, child)
                if CurMin > res_value:
                    CurMin = res_value
                    CurMinLoc = child
                    CurNumOfNodes += res_num_of_nodes
            return CurMinLoc, CurNumOfNodes, CurMin

    def time_bound(self, numOfNodes: int, lastIterationTime, lastDepth) -> (float):
        averageTimePerNode = lastIterationTime / numOfNodes
        nextTreeNumOfNodes = numOfNodes + pow(3, lastDepth + 1)

        return ((averageTimePerNode * nextTreeNumOfNodes) + lastIterationTime)

    def succ(self, board, loc) -> (list):
        list_of_neighbors = list()
        for d in self.directions:
            i = loc[0] + d[0]
            j = loc[1] + d[1]

            if 0 <= i < len(board) and 0 <= j < len(board[0]) and board[i][j] == 0:  # then move is legal
                new_loc = (i, j)
                assert board[new_loc] == 0
                list_of_neighbors.append(new_loc)

        return list_of_neighbors

    def getLoc(self, board, agent: int) -> (tuple):
        for i, row in enumerate(board):
            for j, val in enumerate(row):
                if val == agent:
                    return i, j

    def is_final(self, board, agentTurn):
        agent_1 = self.getLoc(board, FIRST_PLAYER)
        agent_2 = self.getLoc(board, SECOND_PLAYER)

        list_of_neighbors_1 = self.succ(board, agent_1)
        list_of_neighbors_2 = self.succ(board, agent_2)

        if len(list_of_neighbors_1) == 0:
            return True, MIN_UTILITY

        else:
            if len(list_of_neighbors_2) == 0:
                if agentTurn == FIRST_PLAYER:
                    if not self.we_played_first:
                       for child in list_of_neighbors_1:
                           list_of_child_neighbors = self.succ(board, child)
                           if len(list_of_child_neighbors) != 0:
                               return True, MAX_UTILITY
                           else:
                               return True, MIN_UTILITY

                    else:
                        return True, MAX_UTILITY

                else:
                    return True, MAX_UTILITY
            else:
                return False, False
