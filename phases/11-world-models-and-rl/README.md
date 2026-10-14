# Phase 11: World Models and RL

> Learn to act from reward, then learn how the world works so you can practice in your head.

This phase teaches reinforcement learning from the ground up and then moves to the idea that drives much of current research: agents that learn a model of their environment and use it to plan or to train inside imagination. You start with exact dynamic programming on a gridworld, move to model-free learning with Q-learning, DQN and policy gradients on a cart-pole you simulate yourself, and then switch to model-based RL: a learned dynamics model with a CEM planner, Ha and Schmidhuber's VAE + RNN world model trained in a dream, Dreamer's actor-critic that backpropagates through imagined trajectories, and JEPA, which predicts the future in embedding space instead of pixels. Every environment (gridworld, cart-pole, ball worlds, bouncing-ball videos) is written from scratch in the scripts, so nothing needs to be installed or downloaded.

**Prerequisites:** Phase 01 (Neural Networks), Phase 02 (Training Deep Nets), Phase 04 (Sequence Models), Phase 06 (Autoencoders and Self-Supervised)

| # | Lesson | What you build |
|---|--------|----------------|
| 01 | [RL Basics: MDPs](01-rl-basics-mdps/) | A stochastic gridworld as tensors, policy evaluation, value iteration and policy iteration checked by Monte Carlo |
| 02 | [Q-Learning and DQN](02-q-learning-and-dqn/) | Tabular Q-learning, a cart-pole simulator, and DQN with a replay buffer, target network and Huber loss |
| 03 | [Policy Gradients](03-policy-gradients/) | REINFORCE on 32 batched cart-poles, with reward-to-go and a value baseline, and a gradient-variance measurement |
| 04 | [Model-Based RL](04-model-based-rl/) | A neural dynamics model, CEM model predictive control, a data-size sweep and a vmap ensemble |
| 05 | [World Models (VAE + RNN)](05-world-models-vae-rnn/) | V (VAE on frames), M (MDN-RNN) and C (linear controller evolved entirely in the dream) |
| 06 | [Dreamer Intuition](06-dreamer-intuition/) | An RSSM with prior and posterior, lambda-returns, and an actor trained by backprop through imagination |
| 07 | [JEPA and Video Prediction](07-jepa-and-video-prediction/) | Pixel vs latent video prediction, collapse and its EMA fix, and linear probes of the learned state |

Run the first lesson:

```bash
uv run python phases/11-world-models-and-rl/01-rl-basics-mdps/code/main.py
```
