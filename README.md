# rl-wds
Source code for Deep Reinforcement Learning for Real-Time Optimization of Pumps in Water Distribution Systems paper.

### Random seeds to replicate results
Random seed for the data generation: 67.

Random seeds for the trainings: 7, 11, 45, 67.

### Building demo in a Docker image
1. Open a terminal.
2. Clone the repository.
2. Move to the root of the local repository.
3. Invoke: `docker build -f Dockerfile --tag rlwds-demo .`
4. Invoke: `docker run -it -p 8080:8080 --rm rlwds-demo`
