from typing import Tuple

import numpy as np


def imbalance_a(L_l: Tuple[float, float]):
    # Imbalance on ask side of lit pool
    l_a_l, l_b_l = L_l
    return l_a_l / (l_a_l + l_b_l)


def imbalance_b(L_l: Tuple[float, float]):
    # Imbalance on buy side of lit pool
    l_a_l, l_b_l = L_l
    return l_b_l / (l_a_l + l_b_l)


def imbalance(i: str, j: str, L_l: Tuple[float, float]):
    # Imbalance wrapper function
    if (i == "a" and j == "l") or (i == "b" and j == "d"):
        return imbalance_a(L_l)
    elif (i == "b" and j == "l") or (i == "a" and j == "d"):
        return imbalance_b(L_l)
    else:
        raise ValueError(f"Invalid input: i = {i}, j = {j}")


class Lam:
    # Class to save coefficients and to evaluate intensities of processes
    def __init__(self, A: dict, theta: dict, sigma: float, epsilon: float):
        self.A = A
        self.theta = theta
        self.sigma = sigma
        self.epsilon = epsilon

    def eval(self, i: str, j: str, L_l: Tuple[float, float]):
        l_a_l, l_b_l = L_l
        if l_a_l != 0 and l_b_l != 0:
            return self.A[j] * np.exp(
                -(self.theta[j] / self.sigma) * imbalance(i, j, L_l)
            )
        else:
            return self.epsilon


def phi_lat(k):
    # TODO: Optimise structure for performance, but keep currently for clarity
    if k == "lat":
        return 1
    elif k == "non-lat":
        return 0
    else:
        raise ValueError(f"Invalid input: k = {k}")


def phi_d(i, k):
    # TODO: Optimise structure for performance, but keep currently for clarity
    if (i == "a" and k == "lat") or (i == "b" and k == "non-lat"):
        return 1
    elif (i == "a" and k == "non-lat") or (i == "b" and k == "lat"):
        return 0
    else:
        raise ValueError(f"Invalid input: k = {k}, i = {i}")
