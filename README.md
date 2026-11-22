# Optimal Make–Take Contracts between an Exchange and a Market Maker

Deep-learning solver for the principal–agent problem of Baldacci, Manziuk,
Mastrolia & Rosenbaum (2019): an exchange designs make–take incentives for a
market maker that quotes in both a lit order book and a dark pool.

The market maker (agent) chooses bid/ask quotes in both venues to maximise
expected PnL with an inventory penalty; the exchange (principal) chooses the
incentive contract paid on executed volumes to maximise fee revenue net of
incentive payments, anticipating the market maker's best response.

**Project report:**
[Optimal Make–Take Contracts via Deep HJB Solvers (PDF)](https://christopher-dybdahl.github.io/assets/pdfs/market_maker_contracts.pdf)
— Market Microstructure, Imperial College London, with T. Delforge and
J. Jareño García.

## Method

The nested stochastic control problem is solved in two neural stages:

1. **Market-maker best response** (`src/.../agent.py`): a network `L(z, q)`
   maps incentive `z` and inventory `q` to quotes, trained to maximise the
   continuous-time Hamiltonian pointwise.
2. **Exchange HJB** (`src/.../principal.py`): the principal's value function
   is solved backward in time with an actor–critic scheme per time step — the
   critic approximates the value function, the actor outputs the incentive
   schedule, and the critic target is advanced with a Heun
   predictor–corrector step. A stochastic exploration loss stabilises the
   actor. Learning-rate schedulers and gradient clipping are used throughout.

### Implementation notes

Two solver variants exist for the backward pass:

- The **sequential backward sweep** (used for the headline runs in the
  report, with `N = 50` time steps): each time step is trained to
  convergence against the already-converged next step.
- The **parallel (Jacobi-style) update** currently on `main` (with `N = 10`):
  all time steps update simultaneously against their neighbours' current
  estimates, which trades per-step accuracy for wall-clock speed.

Figures and trained checkpoints for the kept runs are under
`saved_models/*/keep/` (e.g. learned incentive schedules in
`saved_models/exchange/keep/0.04_model/incentives.png` and optimal quoted
volumes in `optimal_volumes.png`).

## Results

- Learned incentive schedules are monotone in inventory and steeper for the
  dark pool: the exchange pays most for liquidity where adverse selection is
  highest.
- The market maker's optimal quotes develop the expected inventory skew, and
  the contract shifts quoting toward the side that reduces imbalance.
- Three parameter regimes (tick size, dark-pool intensity, inventory
  aversion) are compared in the report, which also covers Hawkes-driven
  order flow and sketches a deep BSDE alternative for the principal's
  equation.

## Repository layout

```
src/optimal_market_maker_exchange_contract_drl/
  agent.py       # market-maker Hamiltonian maximiser
  principal.py   # exchange actor–critic backward HJB solver
  dynamics.py    # order-flow and inventory dynamics
  nnets.py       # network architectures
  plotting.py    # figures
scripts/
  train_mm.py        # stage 1: best-response training
  train_exchange.py  # stage 2: contract optimisation
config/              # market, agent and training configs (JSON)
notebooks/           # training and plotting notebooks
saved_models/        # kept checkpoints and figures
```

## Setup

Uses [Poetry](https://python-poetry.org/) (Python 3.12+):

```bash
poetry install
```

or with pip/conda:

```bash
pip install -r requirements.txt
pip install -e .
```

Training: `poetry run python scripts/train_mm.py` then
`poetry run python scripts/train_exchange.py` (configs in `config/`).

## Reference

Baldacci, B., Manziuk, I., Mastrolia, T., & Rosenbaum, M. (2019).
*Market making and incentives design in the presence of a dark pool: a deep
reinforcement learning approach.* [arXiv:1912.01129](https://arxiv.org/abs/1912.01129).
