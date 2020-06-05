import time

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
        pass

    def set_rival_move(self, loc):
        pass

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

    def Minimax(self, board, depth: int, agent: int, loc: tuple) -> (tuple, int):
        if depth == 0:
            return self.Minimax_heuristic(board, loc)

        if agent == 1:
            list_of_neighbors = self.succ(board, loc)
            CurMax = float('-inf')



    def time_bound(self, numOfNodes: int, lastIterationTime) -> (float):
        pass

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
