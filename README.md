# Optimal Make-Take Policy Between Exchange and Market Maker

This repository implements the optimal make-take policy between an exchange and a market maker having lit and dark pools from Baldacci et al. (2019).

## Setup and Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.

1.  **Prerequisites**: Ensure you have Python 3.12+ and Poetry installed.

2.  **Install Dependencies**:
    ```bash
    poetry install
    ```
    Poetry installs the project in editable mode by default.

### Alternative Installation (pip / Conda)

A `requirements.txt` file is also provided for environments where Poetry is not used (e.g., Conda).

```bash
pip install -r requirements.txt
pip install -e .  # Install the project in editable mode
```

## References

* Baldacci, B., Manziuk, I., Mastrolia, T., \& Rosenbaum, M. (2019). Market making and incentives design in the presence of a dark pool: A deep reinforcement learning approach. arXiv:1912.01129. https://arxiv.org/abs/1912.01129