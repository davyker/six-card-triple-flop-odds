# Six-Card Triple Flop PLO Poker Simulator & Equity Calculator

Pot Limit Omaha (PLO) is a variant of Texas Hold'em where each player gets 4 cards instead of 2, and at showdown players make their best 5-card hand using exactly 2 of their cards + 3 from the board

This repo contains a Monte Carlo simulator for Six-Card Triple Flop PLO, a variant of Pot Limit Omaha where:
- Each player gets **6 cards** instead of 4
- There are **3 separate boards** of community cards (flops/turns/rivers)
- On each board, players make their best 5-card hand using exactly 2 of their cards + 3 (the two from their hand can differ between boards)
- Winner on each board gets 1 point (ties split the point), player with most points scoops the pot (ties for the most points split the pot)

This variant is popular in some cardrooms in London.

This is a refactored edition of some code I wrote first wrote in 2019, in Pascal, then in Octave/MATLAB, then in Python, then finally refactored and improved.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run games with randomly generated player cards
python main.py --num-players 3

python main.py --num-players 2 --max-iterations 100000 --threshold 0.005

python main.py --num-players 4 --seed 42

# Specify exact hands and boards
# 2 players, preflop
python main.py --cards  "As Ah Kc Qs Jh Tc, Qc Jd 8s 6h 4c 2h"

# 3 players, flop
python main.py --cards "Ks Qh 9s 7c 5d 3h, Ac Jd Th 8h 4s 2c, Kh Qc Td 9h 6s 6d" --boards "Ah 7s 2d, 9c 5h Kd, Js Tc 4h" --threshold 0.001

# 2 players, turn
python main.py --cards "Ah Kc 9c 7d 5h 3s, Qd Jh Tc 8s 4c 2d" --boards "Kd Qs 7h 3c, Ad Td 9s 8c, 5c 4d 2s 6c" --max-iterations 1000000 --threshold 0.0005

#Try this for fun
python main.py --cards  "As Ah Kc Qs Jh Tc, Th Td 2d 2s 2c 2h"
```

## Description

The simulator runs thousands of random board completions to estimate each player's equity (win probability) and uses variance-based convergence checking. With `threshold` = t, each player's estimated equity is accurate to ±2t with 95% confidence.

Each iteration:
1. Randomly complete all 3 boards
2. Evaluate each player's best hand on each board
3. Award points: 1 point per board won (split if tied)
4. Player with most points wins the pot (split if tied)

The running average converges to true equity by the law of large numbers.
The progress bars show:
1. **Iterations**: Progress toward max iterations
2. **Convergence**: How close the standard error is to your threshold (using 1/std² which grows linearly)

The hand evaluation is JIT-compiled with numba for speed. You should see:
- ~1000-3000 iterations/second depending on number of players
- Convergence in 10-30 seconds for typical games
- Standard error < 0.001 (±0.1% accuracy) in reasonable time

## Files

- `main.py` - Command-line interface
- `simulator.py` - Monte Carlo simulation engine
- `hand_evaluator.py` - Fast poker hand evaluation (JIT-compiled with Numba)
- `card_types.py` - Data structures and enums

## Disclaimer

Do whatever you want with this code. Use at your own risk, don't blame me if you lose money playing this ridiculous poker variant.
