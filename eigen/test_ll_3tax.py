#!/usr/bin/env python3
import numpy as np
import jax
import jax.numpy as jnp

# loglik of 3 taxon tree
#
#             5
#            / \
#       b45 /   \
#          4     \ b35
#         / \     \
#    b14 /   \ b24 \
#       /     \     \
#      1       2     3

# the formula is the mathematica expansion of
# eQ[t_] = MatrixExp[t Q];
# lik = pi.((eQ[b45].(eQ[b14].tips[[1]] * eQ[b24].tips[[2]])) * eQ[b35].tips[[3]]);


def loglik3tax(mu, pi, tips, branches):
    exp = jnp.exp
    pi1, pi2, pi3, pi4 = pi
    b14, b24, b45, b35 = branches
    (t11, t12, t13, t14), (t21, t22, t23, t24), (t31, t32, t33, t34) = tips
    return jnp.log(
        pi1
        * (
            (
                (3 + exp(b45 * mu))
                * (
                    ((3 + exp(b14 * mu)) * t11) / (4.0 * exp(b14 * mu))
                    + ((-1 + exp(b14 * mu)) * t12) / (4.0 * exp(b14 * mu))
                    + ((-1 + exp(b14 * mu)) * t13) / (4.0 * exp(b14 * mu))
                    + ((-1 + exp(b14 * mu)) * t14) / (4.0 * exp(b14 * mu))
                )
                * (
                    ((3 + exp(b24 * mu)) * t21) / (4.0 * exp(b24 * mu))
                    + ((-1 + exp(b24 * mu)) * t22) / (4.0 * exp(b24 * mu))
                    + ((-1 + exp(b24 * mu)) * t23) / (4.0 * exp(b24 * mu))
                    + ((-1 + exp(b24 * mu)) * t24) / (4.0 * exp(b24 * mu))
                )
            )
            / (4.0 * exp(b45 * mu))
            + (
                (-1 + exp(b45 * mu))
                * (
                    ((-1 + exp(b14 * mu)) * t11) / (4.0 * exp(b14 * mu))
                    + ((3 + exp(b14 * mu)) * t12) / (4.0 * exp(b14 * mu))
                    + ((-1 + exp(b14 * mu)) * t13) / (4.0 * exp(b14 * mu))
                    + ((-1 + exp(b14 * mu)) * t14) / (4.0 * exp(b14 * mu))
                )
                * (
                    ((-1 + exp(b24 * mu)) * t21) / (4.0 * exp(b24 * mu))
                    + ((3 + exp(b24 * mu)) * t22) / (4.0 * exp(b24 * mu))
                    + ((-1 + exp(b24 * mu)) * t23) / (4.0 * exp(b24 * mu))
                    + ((-1 + exp(b24 * mu)) * t24) / (4.0 * exp(b24 * mu))
                )
            )
            / (4.0 * exp(b45 * mu))
            + (
                (-1 + exp(b45 * mu))
                * (
                    ((-1 + exp(b14 * mu)) * t11) / (4.0 * exp(b14 * mu))
                    + ((-1 + exp(b14 * mu)) * t12) / (4.0 * exp(b14 * mu))
                    + ((3 + exp(b14 * mu)) * t13) / (4.0 * exp(b14 * mu))
                    + ((-1 + exp(b14 * mu)) * t14) / (4.0 * exp(b14 * mu))
                )
                * (
                    ((-1 + exp(b24 * mu)) * t21) / (4.0 * exp(b24 * mu))
                    + ((-1 + exp(b24 * mu)) * t22) / (4.0 * exp(b24 * mu))
                    + ((3 + exp(b24 * mu)) * t23) / (4.0 * exp(b24 * mu))
                    + ((-1 + exp(b24 * mu)) * t24) / (4.0 * exp(b24 * mu))
                )
            )
            / (4.0 * exp(b45 * mu))
            + (
                (-1 + exp(b45 * mu))
                * (
                    ((-1 + exp(b14 * mu)) * t11) / (4.0 * exp(b14 * mu))
                    + ((-1 + exp(b14 * mu)) * t12) / (4.0 * exp(b14 * mu))
                    + ((-1 + exp(b14 * mu)) * t13) / (4.0 * exp(b14 * mu))
                    + ((3 + exp(b14 * mu)) * t14) / (4.0 * exp(b14 * mu))
                )
                * (
                    ((-1 + exp(b24 * mu)) * t21) / (4.0 * exp(b24 * mu))
                    + ((-1 + exp(b24 * mu)) * t22) / (4.0 * exp(b24 * mu))
                    + ((-1 + exp(b24 * mu)) * t23) / (4.0 * exp(b24 * mu))
                    + ((3 + exp(b24 * mu)) * t24) / (4.0 * exp(b24 * mu))
                )
            )
            / (4.0 * exp(b45 * mu))
        )
        * (
            ((3 + exp(b35 * mu)) * t31) / (4.0 * exp(b35 * mu))
            + ((-1 + exp(b35 * mu)) * t32) / (4.0 * exp(b35 * mu))
            + ((-1 + exp(b35 * mu)) * t33) / (4.0 * exp(b35 * mu))
            + ((-1 + exp(b35 * mu)) * t34) / (4.0 * exp(b35 * mu))
        )
        + pi2
        * (
            (
                (-1 + exp(b45 * mu))
                * (
                    ((3 + exp(b14 * mu)) * t11) / (4.0 * exp(b14 * mu))
                    + ((-1 + exp(b14 * mu)) * t12) / (4.0 * exp(b14 * mu))
                    + ((-1 + exp(b14 * mu)) * t13) / (4.0 * exp(b14 * mu))
                    + ((-1 + exp(b14 * mu)) * t14) / (4.0 * exp(b14 * mu))
                )
                * (
                    ((3 + exp(b24 * mu)) * t21) / (4.0 * exp(b24 * mu))
                    + ((-1 + exp(b24 * mu)) * t22) / (4.0 * exp(b24 * mu))
                    + ((-1 + exp(b24 * mu)) * t23) / (4.0 * exp(b24 * mu))
                    + ((-1 + exp(b24 * mu)) * t24) / (4.0 * exp(b24 * mu))
                )
            )
            / (4.0 * exp(b45 * mu))
            + (
                (3 + exp(b45 * mu))
                * (
                    ((-1 + exp(b14 * mu)) * t11) / (4.0 * exp(b14 * mu))
                    + ((3 + exp(b14 * mu)) * t12) / (4.0 * exp(b14 * mu))
                    + ((-1 + exp(b14 * mu)) * t13) / (4.0 * exp(b14 * mu))
                    + ((-1 + exp(b14 * mu)) * t14) / (4.0 * exp(b14 * mu))
                )
                * (
                    ((-1 + exp(b24 * mu)) * t21) / (4.0 * exp(b24 * mu))
                    + ((3 + exp(b24 * mu)) * t22) / (4.0 * exp(b24 * mu))
                    + ((-1 + exp(b24 * mu)) * t23) / (4.0 * exp(b24 * mu))
                    + ((-1 + exp(b24 * mu)) * t24) / (4.0 * exp(b24 * mu))
                )
            )
            / (4.0 * exp(b45 * mu))
            + (
                (-1 + exp(b45 * mu))
                * (
                    ((-1 + exp(b14 * mu)) * t11) / (4.0 * exp(b14 * mu))
                    + ((-1 + exp(b14 * mu)) * t12) / (4.0 * exp(b14 * mu))
                    + ((3 + exp(b14 * mu)) * t13) / (4.0 * exp(b14 * mu))
                    + ((-1 + exp(b14 * mu)) * t14) / (4.0 * exp(b14 * mu))
                )
                * (
                    ((-1 + exp(b24 * mu)) * t21) / (4.0 * exp(b24 * mu))
                    + ((-1 + exp(b24 * mu)) * t22) / (4.0 * exp(b24 * mu))
                    + ((3 + exp(b24 * mu)) * t23) / (4.0 * exp(b24 * mu))
                    + ((-1 + exp(b24 * mu)) * t24) / (4.0 * exp(b24 * mu))
                )
            )
            / (4.0 * exp(b45 * mu))
            + (
                (-1 + exp(b45 * mu))
                * (
                    ((-1 + exp(b14 * mu)) * t11) / (4.0 * exp(b14 * mu))
                    + ((-1 + exp(b14 * mu)) * t12) / (4.0 * exp(b14 * mu))
                    + ((-1 + exp(b14 * mu)) * t13) / (4.0 * exp(b14 * mu))
                    + ((3 + exp(b14 * mu)) * t14) / (4.0 * exp(b14 * mu))
                )
                * (
                    ((-1 + exp(b24 * mu)) * t21) / (4.0 * exp(b24 * mu))
                    + ((-1 + exp(b24 * mu)) * t22) / (4.0 * exp(b24 * mu))
                    + ((-1 + exp(b24 * mu)) * t23) / (4.0 * exp(b24 * mu))
                    + ((3 + exp(b24 * mu)) * t24) / (4.0 * exp(b24 * mu))
                )
            )
            / (4.0 * exp(b45 * mu))
        )
        * (
            ((-1 + exp(b35 * mu)) * t31) / (4.0 * exp(b35 * mu))
            + ((3 + exp(b35 * mu)) * t32) / (4.0 * exp(b35 * mu))
            + ((-1 + exp(b35 * mu)) * t33) / (4.0 * exp(b35 * mu))
            + ((-1 + exp(b35 * mu)) * t34) / (4.0 * exp(b35 * mu))
        )
        + pi3
        * (
            (
                (-1 + exp(b45 * mu))
                * (
                    ((3 + exp(b14 * mu)) * t11) / (4.0 * exp(b14 * mu))
                    + ((-1 + exp(b14 * mu)) * t12) / (4.0 * exp(b14 * mu))
                    + ((-1 + exp(b14 * mu)) * t13) / (4.0 * exp(b14 * mu))
                    + ((-1 + exp(b14 * mu)) * t14) / (4.0 * exp(b14 * mu))
                )
                * (
                    ((3 + exp(b24 * mu)) * t21) / (4.0 * exp(b24 * mu))
                    + ((-1 + exp(b24 * mu)) * t22) / (4.0 * exp(b24 * mu))
                    + ((-1 + exp(b24 * mu)) * t23) / (4.0 * exp(b24 * mu))
                    + ((-1 + exp(b24 * mu)) * t24) / (4.0 * exp(b24 * mu))
                )
            )
            / (4.0 * exp(b45 * mu))
            + (
                (-1 + exp(b45 * mu))
                * (
                    ((-1 + exp(b14 * mu)) * t11) / (4.0 * exp(b14 * mu))
                    + ((3 + exp(b14 * mu)) * t12) / (4.0 * exp(b14 * mu))
                    + ((-1 + exp(b14 * mu)) * t13) / (4.0 * exp(b14 * mu))
                    + ((-1 + exp(b14 * mu)) * t14) / (4.0 * exp(b14 * mu))
                )
                * (
                    ((-1 + exp(b24 * mu)) * t21) / (4.0 * exp(b24 * mu))
                    + ((3 + exp(b24 * mu)) * t22) / (4.0 * exp(b24 * mu))
                    + ((-1 + exp(b24 * mu)) * t23) / (4.0 * exp(b24 * mu))
                    + ((-1 + exp(b24 * mu)) * t24) / (4.0 * exp(b24 * mu))
                )
            )
            / (4.0 * exp(b45 * mu))
            + (
                (3 + exp(b45 * mu))
                * (
                    ((-1 + exp(b14 * mu)) * t11) / (4.0 * exp(b14 * mu))
                    + ((-1 + exp(b14 * mu)) * t12) / (4.0 * exp(b14 * mu))
                    + ((3 + exp(b14 * mu)) * t13) / (4.0 * exp(b14 * mu))
                    + ((-1 + exp(b14 * mu)) * t14) / (4.0 * exp(b14 * mu))
                )
                * (
                    ((-1 + exp(b24 * mu)) * t21) / (4.0 * exp(b24 * mu))
                    + ((-1 + exp(b24 * mu)) * t22) / (4.0 * exp(b24 * mu))
                    + ((3 + exp(b24 * mu)) * t23) / (4.0 * exp(b24 * mu))
                    + ((-1 + exp(b24 * mu)) * t24) / (4.0 * exp(b24 * mu))
                )
            )
            / (4.0 * exp(b45 * mu))
            + (
                (-1 + exp(b45 * mu))
                * (
                    ((-1 + exp(b14 * mu)) * t11) / (4.0 * exp(b14 * mu))
                    + ((-1 + exp(b14 * mu)) * t12) / (4.0 * exp(b14 * mu))
                    + ((-1 + exp(b14 * mu)) * t13) / (4.0 * exp(b14 * mu))
                    + ((3 + exp(b14 * mu)) * t14) / (4.0 * exp(b14 * mu))
                )
                * (
                    ((-1 + exp(b24 * mu)) * t21) / (4.0 * exp(b24 * mu))
                    + ((-1 + exp(b24 * mu)) * t22) / (4.0 * exp(b24 * mu))
                    + ((-1 + exp(b24 * mu)) * t23) / (4.0 * exp(b24 * mu))
                    + ((3 + exp(b24 * mu)) * t24) / (4.0 * exp(b24 * mu))
                )
            )
            / (4.0 * exp(b45 * mu))
        )
        * (
            ((-1 + exp(b35 * mu)) * t31) / (4.0 * exp(b35 * mu))
            + ((-1 + exp(b35 * mu)) * t32) / (4.0 * exp(b35 * mu))
            + ((3 + exp(b35 * mu)) * t33) / (4.0 * exp(b35 * mu))
            + ((-1 + exp(b35 * mu)) * t34) / (4.0 * exp(b35 * mu))
        )
        + pi4
        * (
            (
                (-1 + exp(b45 * mu))
                * (
                    ((3 + exp(b14 * mu)) * t11) / (4.0 * exp(b14 * mu))
                    + ((-1 + exp(b14 * mu)) * t12) / (4.0 * exp(b14 * mu))
                    + ((-1 + exp(b14 * mu)) * t13) / (4.0 * exp(b14 * mu))
                    + ((-1 + exp(b14 * mu)) * t14) / (4.0 * exp(b14 * mu))
                )
                * (
                    ((3 + exp(b24 * mu)) * t21) / (4.0 * exp(b24 * mu))
                    + ((-1 + exp(b24 * mu)) * t22) / (4.0 * exp(b24 * mu))
                    + ((-1 + exp(b24 * mu)) * t23) / (4.0 * exp(b24 * mu))
                    + ((-1 + exp(b24 * mu)) * t24) / (4.0 * exp(b24 * mu))
                )
            )
            / (4.0 * exp(b45 * mu))
            + (
                (-1 + exp(b45 * mu))
                * (
                    ((-1 + exp(b14 * mu)) * t11) / (4.0 * exp(b14 * mu))
                    + ((3 + exp(b14 * mu)) * t12) / (4.0 * exp(b14 * mu))
                    + ((-1 + exp(b14 * mu)) * t13) / (4.0 * exp(b14 * mu))
                    + ((-1 + exp(b14 * mu)) * t14) / (4.0 * exp(b14 * mu))
                )
                * (
                    ((-1 + exp(b24 * mu)) * t21) / (4.0 * exp(b24 * mu))
                    + ((3 + exp(b24 * mu)) * t22) / (4.0 * exp(b24 * mu))
                    + ((-1 + exp(b24 * mu)) * t23) / (4.0 * exp(b24 * mu))
                    + ((-1 + exp(b24 * mu)) * t24) / (4.0 * exp(b24 * mu))
                )
            )
            / (4.0 * exp(b45 * mu))
            + (
                (-1 + exp(b45 * mu))
                * (
                    ((-1 + exp(b14 * mu)) * t11) / (4.0 * exp(b14 * mu))
                    + ((-1 + exp(b14 * mu)) * t12) / (4.0 * exp(b14 * mu))
                    + ((3 + exp(b14 * mu)) * t13) / (4.0 * exp(b14 * mu))
                    + ((-1 + exp(b14 * mu)) * t14) / (4.0 * exp(b14 * mu))
                )
                * (
                    ((-1 + exp(b24 * mu)) * t21) / (4.0 * exp(b24 * mu))
                    + ((-1 + exp(b24 * mu)) * t22) / (4.0 * exp(b24 * mu))
                    + ((3 + exp(b24 * mu)) * t23) / (4.0 * exp(b24 * mu))
                    + ((-1 + exp(b24 * mu)) * t24) / (4.0 * exp(b24 * mu))
                )
            )
            / (4.0 * exp(b45 * mu))
            + (
                (3 + exp(b45 * mu))
                * (
                    ((-1 + exp(b14 * mu)) * t11) / (4.0 * exp(b14 * mu))
                    + ((-1 + exp(b14 * mu)) * t12) / (4.0 * exp(b14 * mu))
                    + ((-1 + exp(b14 * mu)) * t13) / (4.0 * exp(b14 * mu))
                    + ((3 + exp(b14 * mu)) * t14) / (4.0 * exp(b14 * mu))
                )
                * (
                    ((-1 + exp(b24 * mu)) * t21) / (4.0 * exp(b24 * mu))
                    + ((-1 + exp(b24 * mu)) * t22) / (4.0 * exp(b24 * mu))
                    + ((-1 + exp(b24 * mu)) * t23) / (4.0 * exp(b24 * mu))
                    + ((3 + exp(b24 * mu)) * t24) / (4.0 * exp(b24 * mu))
                )
            )
            / (4.0 * exp(b45 * mu))
        )
        * (
            ((-1 + exp(b35 * mu)) * t31) / (4.0 * exp(b35 * mu))
            + ((-1 + exp(b35 * mu)) * t32) / (4.0 * exp(b35 * mu))
            + ((-1 + exp(b35 * mu)) * t33) / (4.0 * exp(b35 * mu))
            + ((3 + exp(b35 * mu)) * t34) / (4.0 * exp(b35 * mu))
        )
    )


def main():
    mu = 1.0
    pi = np.ones(4) / 4
    tips = np.eye(4)[:3]  # A, C, T
    times = np.array([0.1, 0.1, 0.2, 0.3])
    times = np.ones(4)
    print(jax.value_and_grad(loglik3tax, argnums=(3,))(mu, pi, tips, times))


if __name__ == "__main__":
    main()
