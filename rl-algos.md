# 🚀 Reinforcement Learning Algorithm Cheatsheet (Full Technical Edition)

A complete guide to RL algorithms, update rules, and David Silver RL Course references.

---

## 🟢 1. Model-Free Prediction (Value Estimation)
*Goal: Estimate $v_\pi$ or $q_\pi$ for a fixed policy.*

| Algorithm | Setting | Update Timing | Update Rule (Equation) |
| :--- | :--- | :--- | :--- |
| **MC Prediction** | Episodic | End of Episode | $V(S_t) \leftarrow V(S_t) + \alpha [G_t - V(S_t)]$ |
| **TD(0)** | Continuing | Every Step | $V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$ |
| **Semi-Grad TD** | Function Appx | Every Step | $\mathbf{w} \leftarrow \mathbf{w} + \alpha [R + \gamma \hat{v}(S', \mathbf{w}) - \hat{v}(S, \mathbf{w})] \nabla \hat{v}(S, \mathbf{w})$ |



---

## 🔵 2. Model-Free Control (Policy Optimization)
*Goal: Find the optimal policy $\pi_*$.*

| Algorithm | Policy Type | Update Rule (Target Calculation) | David Silver |
| :--- | :--- | :--- | :--- |
| **Sarsa** | On-Policy | $Q(S,A) \leftarrow Q(S,A) + \alpha [R + \gamma Q(S',A') - Q(S,A)]$ | L5 |
| **Q-Learning** | Off-Policy | $Q(S,A) \leftarrow Q(S,A) + \alpha [R + \gamma \max_{a} Q(S',a) - Q(S,A)]$ | L5 |
| **Expected Sarsa**| On-Policy | $Q(S,A) \leftarrow Q(S,A) + \alpha [R + \gamma \sum_{a} \pi(a|S') Q(S',a) - Q(S,A)]$ | L5 |



---

## 🟡 3. Model-Based & Planning
*Goal: Use $P(s'|s,a)$ and $R(s,a)$ to compute values.*

| Algorithm | Bellman Equation Used | Purpose |
| :--- | :--- | :--- |
| **Value Iteration** | $v_{k+1}(s) = \max_{a} \sum_{s', r} p(s',r|s,a) [r + \gamma v_k(s')]$ | Optimal Control |
| **Policy Eval** | $v_{k+1}(s) = \sum_{a} \pi(a|s) \sum_{s', r} p(s',r|s,a) [r + \gamma v_k(s')]$ | Prediction |



---

## 🟠 4. The Bridge: Eligibility Traces ($TD(\lambda)$)
*Goal: Unify TD and MC. Updates states based on how recently/frequently they were visited.*

**Trace Update:** $E_t(s) = \gamma \lambda E_{t-1}(s) + \mathbb{1}(S_t = s)$
**Value Update:** $V(S_t) \leftarrow V(S_t) + \alpha \delta_t E_t(S_t)$

| Algorithm | $\lambda$ Value | Behavior | David Silver |
| :--- | :--- | :--- | :--- |
| **TD(0)** | $\lambda = 0$ | Standard TD (Immediate bootstrap) | L4 |
| **MC** | $\lambda = 1$ | Standard Monte Carlo (Full return) | L4 |



---

## 🟣 5. Continuing Tasks (Average Reward)
*Goal: Maximize reward per time step where $G_t$ is not discounted.*

**Average Reward TD Error:**
$$\delta_t = R_{t+1} - \bar{R}_t + \hat{v}(S_{t+1}, \mathbf{w}) - \hat{v}(S_t, \mathbf{w})$$

| Algorithm | Action Space | Policy Gradient Update ($\nabla_\theta J(\theta)$) |
| :--- | :--- | :--- |
| **Softmax A-C** | Discrete | $\delta_t \nabla_\theta \ln \pi(A_t|S_t, \theta)$ |
| **Gaussian A-C** | Continuous | $\delta_t \nabla_\theta \ln \mathcal{N}(A_t | \mu(S_t), \sigma(S_t))$ |



---

## 🧠 Core Definitions
* **On-Policy:** Updates $Q(s,a)$ based on the actual next action $A'$ taken.
* **Off-Policy:** Updates $Q(s,a)$ based on the hypothetical best action $\max Q(s',a)$.
* **Bootstrapping:** Using an estimate to update an estimate ($R + \gamma \hat{V}$).
* **The Deadly Triad:** Function Approximation + Bootstrapping + Off-Policy = Risk of Divergence.

## 📉 6. Convergence Properties & Stability
*Goal: Understanding where and why algorithms succeed or fail.*

| Algorithm Group | Setting | Convergence Guarantee | Notes |
| :--- | :--- | :--- | :--- |
| **Dynamic Programming** | Tabular / Model Known | **Optimal ($v_*$)** | Guaranteed to find the unique optimal solution. |
| **Monte Carlo** | Tabular / Model Free | **Optimal ($v_*$)** | Unbiased but high variance; slow to converge. |
| **TD(0) / Sarsa** | Tabular / Model Free | **Optimal ($v_*$)** | Converges under Robbins-Monro conditions. |
| **Linear TD / Sarsa** | Function Appx | **TD Fixed Point** | Converges near optimal, but limited by feature quality. |
| **Deep RL (DQN/AC)** | Non-Linear Appx | **None** | Can diverge or oscillate; requires stability tricks (Experience Replay). |

---

### ⚠️ The Convergence "Safe Zone"
To avoid divergence, always check if your setup falls into the **"Deadly Triad."** If all three are present, convergence is not guaranteed:
1. **Function Approximation** (Neural Networks / Linear features)
2. **Bootstrapping** (TD, Sarsa, DP)
3. **Off-Policy Learning** (Q-Learning, Gradient-TD)

---

### 🎓 David Silver Lecture Mapping Recap
* **L1 & L2:** Intro & MDPs (Foundations)
* **L3:** Planning by Dynamic Programming (Value/Policy Iteration)
* **L4:** Model-Free Prediction (MC & TD)
* **L5:** Model-Free Control (Sarsa & Q-Learning)
* **L6:** Value Function Approximation (Linear/Neural Nets)
* **L7:** Policy Gradient Methods (Actor-Critic)
* **L8:** Integrating Learning and Planning (Dyna-Q)
* **L9:** Exploration and Exploitation (Multi-armed Bandits)
* **L10:** Case Study: AlphaGo (RL in Practice)
