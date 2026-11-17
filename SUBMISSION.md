# HW6 submission

## AI-use declaration

I used ChatGPT to help me understand the assignment, check my calculations, write and debug Python code, and organize this report.

I completed Part A calculations on paper and ran Parts B and C in Jupyter Notebook. The experimental results are from my own runs.

## Setup and reproduction

I used Python in Jupyter Notebook.

**Libraries:**
- Gymnasium 1.4.0
- NumPy 1.26.4
- Pandas (installed in my environment)

**Installation:**

```bash
pip install gymnasium==1.4.0 numpy==1.26.4 pandas
```

**Run:**

```bash
jupyter notebook
```

Open `hw6.ipynb` and run the cells from top to bottom.

The notebook contains Parts B and C. Results are saved in `outputs/`.

No API key is required.

## Part A — Paper calculations

My handwritten calculations are saved in [Part_A.pdf](Part_A.pdf). I uploaded them to GitHub before the programming work.

I used the Bellman update with discount factor 0.9 and initial values V0 = 0.

**V1 map:**

```text
 -1   -1   10    G
 -1    #   -1    H
 -1   -1   -1   -1
```

The terminal values for G and H are zero.

**V2 calculations:**

At (1,2), the four action values are:

- Up = -3.9
- Right = 8
- Down = -3.9
- Left = -1.9

V2(1,2) = 8, best action: Right.

At (1,3):

- Up = 8
- Right = 10
- Down = -1.9
- Left = -3.9

V2(1,3) = 10, best action: Right.

At (2,3), movement is slippery:

- Up = 0.8(8) + 0.1(-1.9) + 0.1(-10) = 5.21
- Right = 0.8(-10) + 0.1(8) + 0.1(-1.9) = -7.39
- Down = 0.8(-1.9) + 0.1(-10) + 0.1(8) = -1.72
- Left = 0.8(-1.9) + 0.1(8) + 0.1(-1.9) = -0.91

V2(2,3) = 5.21, best action: Up.

**Best-action map:**

```text
 .   →   →   G
 .   #   ↑   H
 S   .   .   .
```

There are no tied best actions at these three positions.

Two iterations are not enough because state values can still change. More iterations are needed until convergence.

## Part B — FrozenLake

### Fixed policy, written before the experiments

I used the default FrozenLake 4x4 map.

Actions: 0 = Left, 1 = Down, 2 = Right, 3 = Up.

My fixed policy was:

| State | Action |
|---|---|
| 0 | Down |
| 1 | Right |
| 2 | Down |
| 3 | Left |
| 4 | Down |
| 6 | Down |
| 8 | Right |
| 9 | Down |
| 10 | Down |
| 13 | Right |
| 14 | Right |

I chose these actions by looking at the map and trying to avoid holes.

Without slipping, the successful path is:

`0 → 4 → 8 → 9 → 13 → 14 → 15`

The agent reaches the goal in 6 steps.

I kept the fixed policy unchanged during all experiments.

### Results

I tested random and fixed policies with and without slipping. Each condition had three batches of 50 episodes, using batch seeds 11, 22 and 33.

| Slippery? | Policy | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Success % | Mean steps |
|---|---|---|---|---|---|---|---|
| False | Fixed | 11 | 50 | 0 | 0 | 100 | 6.00 |
| False | Fixed | 22 | 50 | 0 | 0 | 100 | 6.00 |
| False | Fixed | 33 | 50 | 0 | 0 | 100 | 6.00 |
| False | Random | 11 | 0 | 50 | 0 | 0 | 7.94 |
| False | Random | 22 | 1 | 49 | 0 | 2 | 7.16 |
| False | Random | 33 | 2 | 48 | 0 | 4 | 6.66 |
| True | Fixed | 11 | 1 | 49 | 0 | 2 | 4.24 |
| True | Fixed | 22 | 2 | 48 | 0 | 4 | 5.48 |
| True | Fixed | 33 | 1 | 49 | 0 | 2 | 4.44 |
| True | Random | 11 | 1 | 49 | 0 | 2 | 7.98 |
| True | Random | 22 | 3 | 47 | 0 | 6 | 7.66 |
| True | Random | 33 | 1 | 49 | 0 | 2 | 7.16 |

**Saved results:**

- [Episode results](outputs/part_b_episodes.csv)
- [Batch summary](outputs/part_b_summary.csv)
- [Successful and failed traces](outputs/part_b_traces.txt)

**Explanation:**

Without slipping, the fixed policy reached the goal in all 150 episodes. With slipping, it succeeded only 4 times out of 150.

The random policy had low success rates in both environments.

The batches had some differences, but the general results were similar. Slipping made the fixed path much less reliable.

One successful episode is not enough because random movement can produce different outcomes.

These simulations are not training because the policies do not learn or change. I only tested their performance.

## Part C — Custom solver

### Compare code with paper

I implemented value iteration using the Part A rules.

The program calculated:

**V1:**

```text
 -1   -1   10    0
 -1    0   -1    0
 -1   -1   -1   -1
```

**Requested V2 values:**

| Position | Paper | Python |
|---|---|---|
| (1,2) | 8.00 | 8.00 |
| (1,3) | 10.00 | 10.00 |
| (2,3) | 5.21 | 5.21 |

All results matched my paper calculations. No discrepancies were found.

### Values, policies and convergence

I used value iteration with discount factor 0.9.

The stopping condition was a maximum value change below 0.000001, with a limit of 10,000 iterations.

**Original map:**

Iterations: 12

Final change: approximately 0.0000002165

Values:

```text
 4.2000   8.0000  10.0000   0
 2.7800   0        5.8242   0
 1.5358   2.8176   4.2418   2.8176
```

Policy:

```text
 →   →   →   G
 ↑   #   ↑   H
 →   →   ↑   ←
```

**Extended map:**

Iterations: 11

Final change: approximately 0.0000002406

Values:

```text
 4.2000   8.0000  10.0000   0
 2.7800   0        5.8242   0
 1.5020   2.8176   4.2418   2.8176
```

Policy:

```text
 →   →   →   G
 ↑   #   ↑   H
 ↑   →   ↑   ←
```

Both maps converged successfully.

For ties, my code uses `np.argmax()`, which selects the first maximum in the order Up, Right, Down, Left.

The value maps and policies are saved in [part_c_maps.txt](outputs/part_c_maps.txt).

### Simulation results

I simulated the computed policies on both maps, with three batches of 50 episodes each.

| Map | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Mean steps | Mean discounted return |
|---|---|---|---|---|---|---|
| Original | 11 | 44 | 6 | 0 | 4.92 | 1.5211 |
| Original | 22 | 45 | 5 | 0 | 5.04 | 1.6550 |
| Original | 33 | 47 | 3 | 0 | 5.10 | 2.1510 |
| Extended | 11 | 50 | 0 | 0 | 5.00 | 1.5020 |
| Extended | 22 | 50 | 0 | 0 | 5.00 | 1.5020 |
| Extended | 33 | 50 | 0 | 0 | 5.00 | 1.5020 |

**Saved results:**

- [Episode results](outputs/part_c_episodes.csv)
- [Batch summary](outputs/part_c_summary.csv)

### Explain your results

In the original map, the start value was 1.5358, and the best action was Right.

In the extended map, the start value changed to 1.5020, and the best action became Up.

Sand made the lower path more expensive, so the agent chose a different route.

Water had a high cost, but it did not change the selected route near the goal.

On ice, the agent selected Up because moving Right could lead to the hole.

The original map had 136 goals and 14 holes out of 150 episodes. The extended map had 150 goals and no holes.

The computed values represent expected discounted rewards over an infinite horizon. Simulations use only 100 steps and a limited number of episodes, so their average returns may be different.

### Rule checks

I tested the rules before trusting the solver.

| Check | Result |
|---|---|
| Wall behavior | PASS |
| Boundary behavior | PASS |
| Ice probabilities sum to 1 | PASS |
| Mud reward (-3) | PASS |
| Sand reward (-2) | PASS |
| Water reward (-5) | PASS |
| Goal reward (+10) | PASS |
| Hole reward (-10) | PASS |
| Terminal values are zero | PASS |
| Terminal stopping check | PASS |

When the agent tries to move into a wall or outside the map, it stays in the same state and receives the corresponding reward.

The ice probabilities are 0.8, 0.1 and 0.1, which sum to 1.

The simulation stops when the agent reaches the goal or hole. Terminal states have value zero and no additional rewards after the episode ends.

The checks are saved in [part_c_checks.txt](outputs/part_c_checks.txt).

## Conclusion

In this homework, I calculated Grid World values manually, tested FrozenLake policies and implemented value iteration.

I found that slipping increases the risk of failure. I also found that changing floor costs can change the best policy.

The results helped me understand how rewards, probabilities and discounting affect an agent's decisions.
