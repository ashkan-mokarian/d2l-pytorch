# Notes

## Markov Decision Process MDP
Is a model of how the state of a system evolves as different actions are applied to the system. It consists of the following quantities:

* Set of *States* $S$. A set of states where the agent can be at any time.
* Set of *Actions* $A$. Any action can change the state.
* A *Transition function* T such that $T(s,a,s') = P(s'|s, a)$. It is a probability distribution, i.e. $\sum_{s'\in S}T(s', a, s)=1$ for all $s\in S$ and $a\in A$.
* A *Reward function* $r:S\times A \rightarrow \mathbb{R}$.

These all form a MDP:$(S, A, T, r)$. Starting from an initial position, the goal of RL is to find a trajectory with the largest reward *return*. In order to compensate for infinite or too long trajectories, a *discounted return* is used where later actions are penalized by: $R(\tau)=r_0 + \gamma r_1 + \gamma^2r_2, ...$ where $\tau=(s_0, a_0, r_0, s_1, a_1, r_1, ...)$.

### Markovian assumption
The next state only depends on the current action and current state and none before.

## Value Iteration
An algorithm to find the best trajectory of an MDP.

*Stochastic Policy* or policy in short is $\pi (a | s) = P(a|s)$ (it is a conditional probability distribution). A deterministic policy is a special case of a stochastic policy.

### Value function
The value function for policy $\pi$ for the state s0 denoted by $V^{\pi}(s_0)$ is the expected $\gamma$-discounted return of agent beginning at state s0 and taking actions according to $\pi$:

$
V^{\pi}(s_0) = \mathbb{E}_{a_t \sim \pi(s_t)} [R(\tau)] = \mathbb{E}_{a_t \sim \pi(s_t)} [\sum_{t=0}^{\inf}\gamma^{t}r(s_t, a_t)]
$

All algorithms in RL follows the following pattern where based on the **Markov assumption**, the value at state s0 can be written in terms of two terms, the reward of taking an action a0 taken at s0 and the value function of the rest which is very intuitive:

$
V^{\pi}(s_0) = r(s_0, a_0) + \gamma \mathbb{E}_{a_0 \sim \pi(s_0)}[ \mathbb{E}_{s_1 \sim P(s_1 | s_0, a_0)}[ V^{\pi}(s_1) ] ]
$

We can write it w.r.t probabilities and for any state s as:

$
V^{\pi}(s) = \sum_{a\in A} \pi(a|s) [
    r(s,a) + \gamma \sum_{s' \in S} [
        P(s' | s, a) V^{\pi}(s')
    ]
]
$ for all $s \in S$.

### Action-Value function
Another useful notation is action value function which is the value function after taking action a:

$
Q^\pi(s_0, a_0) = r(s_0, a_0) + V^\pi (s_1)
$

or similar to before be can expand it in terms of itself as:

$
Q^\pi(s, a) = r(s, a) + \gamma \sum_{s' \in S}P(s'|s, a) \sum_{a' \in A}\pi(a' | s')Q^\pi(s', a')
$ for all $s \in S$ and $a \in A$.

## Optimal policy
$
\pi^* = \argmax_{\pi} V^\pi(s_0) 
$

Furthermore we can write $V^*\equiv V^{\pi^*}$ and $Q^*=Q^{\pi^*}$.

For a deterministic policy, we can write:

$
\pi^*(s) = \argmax_{a\in A} [
    r(s, a) + \gamma \sum_{s' \in S} P(s' | s, a) V^* (s')
]
$

A mnemonic to remember this statement is that the optimal action at state s (for a deterministic policy) is the one that maximizes the sum of reward from the first stage and the average return of the trajectories starting from s' over all possible next states s' from the second stage.

### Principle of dynamic programming
**The remainder of an optimal policy is also optimal**. This follows the formulas before by replacing the optimal polict. Hence for a deterministic policy, we have:

$
V^*(s) = \argmax_{a\in A} [
    r(s, a) + \gamma \sum_{s' \in S} P(s' | s, a) V^* (s')
$

Or for a stochastic policy, the optimal value function is:

$
V^{*}(s) = \sum_{a\in A} \pi^*(a|s) [
    r(s,a) + \gamma \sum_{s' \in S} [
        P(s' | s, a) V^{*}(s')
    ]
$ for all $s \in S$.

## Value Iteration (algorithm)
We can turn the *principle of dynamic programming* into an algorithm for finding the optimal value function called **Value iteration**. The idea is to intialize the value function at iteration 0 for all $s \in S$ to some arbitrary values. At the k-th iteration, the Value iteration algorithm updates the value function as

$
V_{k+1}(s) = \max_{a \in A} [
    r(s, a) + \gamma \sum_{s' \in S}P(s'|s, a)V_{k}(s')
]
$ for all $s \in S$.

It turns out that as $k \rightarrow \inf$, the Value function estimated by the value iteration algorithm converges to the optimal value function irrespective of the initialization $V_0$, i.e. $V^*(s) = \lim_{k \rightarrow \inf} V_k(s)$ for all $s \in S$.

The same value iteration algorithm can be written for the Action-value function.

## Policy Evaluation
Value iteration algorithm allows us to compute the optimal value function $V^{\pi^*}$ of the optimal **deterministic** policy $\pi^*$. We can also use the same value iteration algorithm to compute the ~~optimal~~ value function associated with any other, potentially stochastic, policy $\pi$. We again initialize $V^\pi_0(s)$ to some arbitrary values for all states s. Then we perform the iteration at step k,

$
V_{k+1}^\pi(s) = \sum_{a\in A} \pi(a|s) [
    r(s,a) + \gamma \sum_{s' \in S}P(s'|s,a) V_k^\pi(s')
]
$
for all $s \in S$.

This algorithm is know as policy evaluation and is useful to compute the value function given any policy. Again as before, as $k \rightarrow \inf$, $V_k^\pi(s)$ converges to the correct value function for any initialization.

# [Stanford DeepRL Course](http://rail.eecs.berkeley.edu/deeprlcourse/)
Since the material at d2l.ai was not complete, for RL switched to the stanford course.


