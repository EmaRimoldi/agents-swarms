# Comparison Summary: Parallelisation vs Swarm

## Best val_bpb achieved

| Experiment | Agent 0 best | Agent 1 best | System best |
|-----------|-------------|-------------|------------|
| Parallel (exp_20260401) | 1.113884 | 1.113130 | 1.113130 |
| Swarm 1  (exp_20260405a) | 1.047929 | 1.041477 | 1.041477 |
| Swarm 2  (exp_20260405b) | 1.044341 | 1.050358 | 1.044341 |

## Agent correlation (Pearson r, shared steps)

- Parallel: r=0.949, p=0.000
- Swarm 1: r=0.923, p=0.000
- Swarm 2: r=0.488, p=0.076

## Interpretation

- Swarm experiments achieve substantially lower val_bpb than parallelisation.
- Lower agent correlation in swarms suggests genuine divergence in search paths,
  enabled by the shared blackboard claim mechanism.
- Parallelisation agents converge to similar local minima (high correlation),
  consistent with redundant exploration without communication.