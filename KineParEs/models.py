import numpy as np


"""
The `Models` class is a container for different types of reaction models.
It has two methods:

1. `__init__()`: The initializer method for this class. It accepts a
   stoichiometric matrix `N` and a string `type` that indicates the
   type of reaction model.

2. `model()`: This is the method that defines the different types of
   reaction models. It accepts the current concentrations `C`, the
   current time `t`, and the reaction rate constants `K`.
   The reaction model to use is determined by the `type` attribute set
   during initialization.

   - If `type` is `'1'`, a lipase reaction model is used. The rate
   of reaction `M` is given by the Michaelis-Menten equation.
   The rate of change of concentrations `dcdt` is then computed by
   multiplying the transpose of the stoichiometric matrix with `M`.

   - If `type` is `'2'`, a Wittig reaction model is used. The rates of
   three reactions are defined, and `dcdt` is computed similarly as above.

   - If `type` is `'custom'`, a custom reaction model is used. Here, you
   can define your own reaction model.
   Currently, it is assumed that the model is set up for a reaction system
   with two reactions. If you want to use more or fewer reactions,
   you will need to adjust the `M` array accordingly.

In each case, the method returns `dcdt` as a list. These are the rates of change
of concentration of each species, which can be used to solve the system of
ordinary differential equations (ODEs) that describe the evolution of the system
over time.
"""


class models:
    """A container for different types of reaction models."""

    def __init__(self, N, type):
        """
        Initializes the Models class.

         Args:
            N (ndarray): Stoichiometric matrix of
                        shape R (number of reactions) x S (number of species).
            model_type (str): Type of reaction model.

        Raises:
            ValueError: If N is not a 2D numpy array.
        """
        if not isinstance(N, np.ndarray) or len(N.shape) != 2:
            raise ValueError("N must be a 2D numpy array")
        self.N = N
        self.type = type

    def model(self, C, t, K):
        """
        Defines the different types of reaction models.

        Args:
            C (ndarray): Current concentrations.
            t (float): Current time.
            K (ndarray): Reaction rate constants.

        Returns:
            list: Rates of change of concentration for each species.

        Raises:
            ValueError: If model_type is invalid.
        """
        if self.type == "1":  # lipase reaction
            M = np.array([[(K[1] * C[0]) / (K[0] + C[0])]])
            dcdt = (self.N.T) @ M
            return dcdt.flatten().tolist()
        elif self.type == "2":  # wittig reaction
            M = np.array([[K[0] * C[0] * C[0]], [K[1] * C[0]], [K[2] * C[0] * C[5]]])
            dcdt = (self.N.T) @ M
            return dcdt.flatten().tolist()
        elif self.type == "custom":
            # write your
            M = np.array([[K[0] * C[0] * C[1]], [K[1] * C[2]]])
            dcdt = (self.N.T) @ M
            return dcdt.flatten().tolist()
            # pass
