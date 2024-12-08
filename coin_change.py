from collections import defaultdict
'''
Problem from leetcode: 322. Coin Change
You are given an integer array coins representing coins of 
different denominations and an integer amount representing a total amount of money.
Return the fewest number of coins that you need to make up that amount. If that 
amount of money cannot be made up by any combination of the coins, return -1.
You may assume that you have an infinite number of each kind of coin.
'''


# classical recursive DP solution
def recursion_solution(coins, amount):
    MIN_COIN = min(coins)
    solution_to_subtree = {}

    def helper(sum_to_go):
        if sum_to_go in solution_to_subtree:
            return solution_to_subtree[sum_to_go]

        if sum_to_go == 0:
            return 0

        if sum_to_go < MIN_COIN:
            return float('inf')

        best_result = float('inf')
        for coin in coins:
            best_result = min(1 + helper(sum_to_go - coin), best_result)

        solution_to_subtree[sum_to_go] = best_result
        return best_result

    result = helper(amount)
    return result if result != float('inf') else -1


# Value iteration solution to the problem
def MDP(s, a, s_next):
    return 1


def R(s, a, s_next):
    return -1


def V_star(state, actions, Q):
    v_star = -float('inf')
    for a in actions:
        v_star = max(v_star, Q[(state, a)])
    return v_star


def value_iteration(coins, amount):
    theta = 1e-6
    V = defaultdict(int)
    states = [i for i in range(0, amount + 1)]
    actions = coins
    while True:
        Q = defaultdict(lambda: -float('inf'))
        for a in actions:
            for s in states:
                # no coins needed
                if s == 0:
                    Q[(s, a)] = 0
                    continue

                # next state is a sum to go
                s_next = s - a

                # very bad move
                if s_next < 0:
                    continue

                Q[(s, a)] = MDP(s, a, s_next) * (R(s, a, s_next) + V[s_next])

        max_diff = 0.0
        for s in states:
            # greedify
            V_new = V_star(s, actions, Q)
            # stopping condition
            max_diff = max(max_diff, abs(V_new - V[s]))
            # update the value function
            V[s] = V_new

        if max_diff < theta:
            break
    # since the rewars is -1 on every decision (every time we take a coin), gamma is 1
    # this means the value function for initial state corresponds to the number of coins
    number_of_coins = abs(V[amount])
    return number_of_coins if number_of_coins != float('inf') else -1


if __name__ == "__main__":
    coins_tests = [[1], [2], [1, 2, 5, 8, 17], [5, 306, 188, 467, 1088]]
    amount_tests = [0, 3, 110, 1078]

    for i in range(len(coins_tests)):
        input = {'coins': coins_tests[i], 'amount': amount_tests[i]}
        assert value_iteration(**input) == recursion_solution(**input)
