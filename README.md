# Breakout-with-A3C

Following along the final section of this course: https://www.udemy.com/course/artificial-intelligence-az/

In this section the A3C algorithm is explained and applied to the breakout game in gym. The algorithm was introduced in 2016, so it has since been surpassed by newer approaches. I had partly finished this course back in 2017, but then forgot about it. So here I am completing the course. 

## A3C algorithm
A3C stands for Asynchronous Advantage Actor Critic and is a reinforcement learning algorithm that keeps track of both the policy and the value function. In practice this is typically implemented as a neural network with a common backbone and two different heads, one for the actor learning the policy and another for the critic learning the value function. 

Where does the name come from?
- Asynchronous: The algorithm uses multiple agents to train. Through a global network the learnings of these agents get synchronized periodically.
- Advantage: The advantage tells the agent how much better its reward was compared to what it expected. With this extra insight, the learning process of the agent is improved.
- Actor-Critic: Combines value-iteration and policy gradient methods by both predicting the value of the current state and the optimal policy. The former is predicted by the critic and the latter by the actor. 

The method is faster and more robust than the older RL algorithms, but has since been surpassed by more modern approaches.
