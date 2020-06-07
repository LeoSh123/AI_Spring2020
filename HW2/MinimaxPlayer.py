import time as T

class MinimaxPlayer:
    def __init__(self):
        self.loc = None
        self.board = None
        self.directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]


    def set_game_params(self, board):
        self.board = board
        for i, row in enumerate(board):
            for j, val in enumerate(row):
                if val == 1:
                    self.loc = (i, j)
                    break

    def make_move(self, time) -> (tuple):
        ID_start_time = T.time()
        depth = 1

        move, numOfNodes, value = self.Minimax(self.board, depth, 1, self.loc)
        last_iteration_time = T.time() - ID_start_time
        next_iteration_time = self.time_bound(numOfNodes, last_iteration_time, depth)
        time_until_now = T.time() - ID_start_time

        while time_until_now + next_iteration_time < time:
            depth += 1
            iteration_start_time = T.time()
            move, numOfNodes, value = self.Minimax(self.board, depth, 1, self.loc)
            last_iteration_time = T.time() - iteration_start_time
            next_iteration_time = self.time_bound(numOfNodes, last_iteration_time, depth)
            time_until_now = T.time() - ID_start_time

        return move

    def set_rival_move(self, loc):
        self.board[self.getLoc(self.board, 2)] = -1
        self.board[loc] = 2

    # TODO: Implement a new one
    def Minimax_heuristic(self, board, loc):
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

    def Minimax(self, board, depth: int, agent: int, loc: tuple) -> (tuple, int, int):
        if depth == 0:
            return loc, 0, self.Minimax_heuristic(board, self.getLoc(board, 1))

        CurNumOfNodes = 0

        if agent == 1:
            agent_loc = self.getLoc(board, agent)
            list_of_neighbors = self.succ(board, agent_loc)
            CurMax = float('-inf')
            CurMaxLoc = None
            CurNumOfNodes = len(list_of_neighbors)
            for child in list_of_neighbors:
                temp_board = board
                temp_board[loc] = -1
                temp_board[child] = 1
                res_loc, res_num_of_nodes, res_value = self.Minimax(temp_board, depth - 1, 2, child)
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
                temp_board = board
                temp_board[loc] = -1
                temp_board[child] = 2
                res_loc, res_num_of_nodes, res_value = self.Minimax(temp_board, depth - 1, 1, child)
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
