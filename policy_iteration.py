import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


def plot_policy_on_grid(env, policy, Q):
    """Visualizes the optimal policy on a grid, removing arrows for holes and rewards."""
    action_symbols = {0: '←', 1: '↓', 2: '→', 3: '↑'}
    grid_size = int(np.sqrt(env.observation_space.n))

    # Convert policy to action symbols
    actions = np.argmax(policy, axis=1)
    action_grid = np.array([action_symbols[a]
                            for a in actions]).reshape(grid_size, grid_size)

    # Mark holes and reward positions with blanks
    for s in range(env.observation_space.n):
        # whole and reward cells have no q-function since terminal
        if Q[s].sum() == 0:
            action_grid[s // grid_size][s % grid_size] = ''

    _, ax = plt.subplots()
    ax.axis('off')
    table = ax.table(cellText=action_grid, loc='center', cellLoc='center')
    table.scale(1, 1.5)
    table.set_fontsize(16)
    plt.show()


def policy_evaluation(env, policy: np.array, gamma: float) -> np.array:
    """"Estimation of the value function"""
    theta = 1e-6
    prev_Q = np.zeros((env.observation_space.n, env.action_space.n))
    MDP = env.unwrapped.P

    while True:
        # Bellman operator is a contraction
        # Iterate long enough -> Convergence
        Q = np.zeros((env.observation_space.n, env.action_space.n))
        for s in range(env.observation_space.n):
            for a in range(env.action_space.n):
                # Compute the Q-value for (state, action) pair
                # iterate over all possible outcomes from the (state, action)
                # for the next Q iterate over actions
                Q[s][a] = sum(
                    prob * (reward + gamma * (not done) *
                            sum(policy[next_state][next_action] *
                                prev_Q[next_state][next_action]
                                for next_action in range(env.action_space.n)))
                    for prob, next_state, reward, done in MDP[s][a])

        # Check for convergence
        if np.max(np.abs(prev_Q - Q)) < theta:
            break

        prev_Q = Q.copy()

    return prev_Q


def greedify(Q):
    """Greedify policy: policy improvement"""
    new_pi = np.zeros_like(Q)
    new_pi[np.arange(Q.shape[0]), np.argmax(Q, axis=1)] = 1
    return new_pi


def policy_iteration(env, policy, gamma=0.9, theta=1e-6):
    """Policy iteration"""
    while True:
        old_policy = policy.copy()
        Q = policy_evaluation(env, policy, gamma)
        policy = greedify(Q)
        if np.max(np.abs(old_policy - policy)) < theta:
            break

    return policy, Q


def value_iteration(env, gamma=0.9, theta=1e-6):
    # no eval step of the policy, no init policy
    V = np.zeros(env.observation_space.n)
    MDP = env.unwrapped.P

    # Find optimal Q
    while True:
        # Bellman operator is a contraction
        # Iterate long enough -> Convergence
        Q = np.zeros((env.observation_space.n, env.action_space.n))
        for s in range(env.observation_space.n):
            for a in range(env.action_space.n):
                # Compute the Q-value for (state, action) pair
                # iterate over all possible outcomes from the (state, action)
                # for the next Q iterate over actions
                Q[s][a] = sum(prob * (reward + gamma *
                                      (not done) * V[next_state])
                              for prob, next_state, reward, done in MDP[s][a])

        # Check for convergence
        if np.max(np.abs(V - Q.max(axis=1))) < theta:
            break

        V = Q.max(axis=1)

    # return optimal policy
    return greedify(Q), Q


if __name__ == "__main__":
    env = gym.make('FrozenLake-v1')

    # define random policy
    policy = np.random.uniform(0, 1,
                               (env.observation_space.n, env.action_space.n))
    policy /= policy.sum(axis=1, keepdims=True)

    optimal_policy_pi, Q_pi = policy_iteration(env, policy)
    optimal_policy_vi, Q_vi = value_iteration(env)

    assert np.all(np.isclose(Q_pi, Q_vi, atol=1e-5))
    assert np.all(optimal_policy_pi == optimal_policy_vi)

    # Plot the optimal policy on a grid
    plot_policy_on_grid(env, optimal_policy_vi, Q_vi)
