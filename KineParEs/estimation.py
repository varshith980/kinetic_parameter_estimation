import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import random
from scipy.optimize import minimize
from scipy.integrate import odeint
from tqdm import tqdm
from .models import models
# from .models import *
from sklearn.linear_model import LinearRegression

"""
    par_estimate class arguments:
     spectra: Absorbance Matrix of shape - Residence time x Number of wavelengths
     c0: Initial concentration of shape - S(No of species in reaction)
     N: Stoichiometric matric of shape - R(no of reactions) x S
     tau: List of residence times
     lamdas = List of Wavelengths where we have measured (shape- No of wavelengths)

    The `par_estimate` class is a tool for parameter estimation and model selection
    in the context of reaction spectroscopic data analysis.
    The class uses several sophisticated techniques such as Principal Component
    Analysis (PCA), Singular Value Decomposition (SVD), and different kinds of
    model fitting to achieve this.
    Here's a description of some of the key methods included in this class:

- `__init__()`: The initializer method for this class. It accepts several
    parameters, including the input spectra, initial concentrations, a
    stoichiometric matrix, a list of residence times, wavelengths, and a
    function defining the model.
    It also initializes several other parameters.

- `preprocess()`: This method checks the validity of the input data and
    preprocesses the input spectra.

- `savitzky_golay()`: Applies the Savitzky-Golay filter to the input
    spectra to smooth it. This method can be used to reduce noise.

- `subset_slice()`: This method allows you to select a subset of the
    spectrum by removing a specified number of wavelengths from the start
    and end of the data.

- `MSC()`: Applies Multiplicative Scatter Correction (MSC) to the
    spectra. MSC is a technique used in spectroscopy to correct for scatter
    in the spectra.

- `spectrum_plot()`: This method provides a visualization of the spectra.

- `variance_ratio()`: This method performs Principal Component Analysis (PCA)
    to determine the number of significant components
    (i.e., number of independent reactions) in the spectra.

- `norms()`: Calculates the Frobenius norm. This is used to measure the
    'distance'/difference between the predicted and observed spectra.

- `optim()`: This method performs parameter optimization using the
    Nelder-Mead method.

- `conc_profile()`: Plots the concentration profile of the species over
    the residence time.

- `fun_arr()`: Given an array of kinetic models, this method selects the
    best one based on the residual of the optimization process.

- `forward_EFA()` and `backward_EFA()`: These methods perform forward and
    backward Evolving Factor Analysis (EFA) respectively.

- ` rmax(A)`: Computes the maximum number of reactions that can be estimated
    based on the given stoichiometric matrix A.

This class provides a comprehensive set of tools for analyzing spectroscopic data,
    identifying significant components, and fitting kinetic models to the data.

    """


class par_estimate(models):
    def __init__(
        self,
        A: np.ndarray,
        c0: np.ndarray,
        N: np.ndarray,
        tau: list,
        lamdas: list,
        n: int,
        model_type: str,
        fun: callable = None,
    ) -> None:

        """
        Initializes the par_estimate class.

        Args:
            A (ndarray): Absorbance of shape - Residence time x No of wavelengths.
            c0 (ndarray): Initial concentration shape - S (No of species in reaction).
            N (ndarray): Stoichiometric matrix of shape - R (no of reactions) x S.
            tau (list): List of residence times.
            lamdas (list): List of Wavelengths where we have measured
            (shape - No of wavelengths).
            n (int): Number of parameters to estimate.
            model_type (str): Type of kinetic model to use ('1', '2', 'custom').
            fun (function, optional): Function defining the custom model
            when model_type is 'custom'. Defaults to None.
        """
        if not isinstance(A, np.ndarray) or len(A.shape) != 2:
            raise ValueError("spectra must be a 2D numpy array")
        if not isinstance(c0, np.ndarray) or len(c0.shape) != 1:
            raise ValueError("c0 must be a 1D numpy array")
        if not isinstance(N, np.ndarray) or len(N.shape) != 2:
            raise ValueError("N must be a 2D numpy array")
        if N.shape[1] != c0.shape[0]:
            raise ValueError("Number of species in c0 and N must match")
        # if model_type not in ['1', '2', 'custom']:
        #     raise ValueError("model_type must be one of '1', '2', 'custom'")
        if not callable(fun) and model_type == "custom":
            raise ValueError("A valid function must be provided for custom model type")
        self.spectra = A
        self.lamdas = lamdas
        self.N = N
        self.tau = tau
        self.c0 = c0
        self.par = np.zeros((n))
        self.n = n
        self.model_type = model_type
        self.R = N.shape[0]  # num of reactions
        self.S = N.shape[1]  # num of species
        self.fun = fun

    # standard preprocessing of spectra
    def preprocess(self) -> None:
        """
        Checks the validity of the input data and preprocesses the input spectra.
        """
        if np.any(self.c0 < 0) is True:
            raise Exception("Initial concentration cannot be negative")
        if len(self.tau) != self.spectra.shape[0]:
            raise ValueError(
                "Dimensions mismatch. Give spectra and tau with same number of \
                    time intervals"
            )
        if self.S != self.c0.shape[0]:
            raise ValueError(
                "Number of species does not match in initial conc and N matrix"
            )
        self.spectra = np.where(self.spectra < 0, 0, self.spectra)

    def savitzky_golay(self, window_size: int = 5, order: int = 3) -> None:
        """
        Applies the Savitzky-Golay filter to the input spectra to smooth it.
        This method can be used to reduce noise.

        Args:
            window_size (int): The size of the smoothing window. Defaults to 5.
            order (int): The order of the polynomial to fit. Defaults to 3.
        """

        self.spectra = savgol_filter(
            self.spectra, window_length=window_size, polyorder=order, mode="nearest"
        )

    def subset_slice(self, a: int, b: int) -> None:
        """
        Selects a subset of the spectrum by removing a specified number of
        wavelengths from the start and end of the data.

        Args:
            a (int): Number of wavelengths to remove from the start.
            b (int): Number of wavelengths to remove from the end.
        """

        # if a > b or a < 0 or b > self.spectra.shape[1]:
        #     raise ValueError("Invalid slicing parameters")

        self.lamdas = self.lamdas[a:-b]
        self.spectra = self.spectra[:, a:-b]

    # Multiplicative scattering correction
    def MSC(self, dref: np.ndarray = None) -> None:
        """
        Applies Multiplicative Scatter Correction (MSC) to the spectra.
        MSC is a technique used to correct for scatter in the spectra.

        Args:
            dref (ndarray, optional): Reference spectrum to use for MSC.
            Defaults to None.
        """
        if dref is None:
            dref = np.mean(self.spectra, axis=0)
        X = list(dref.reshape(-1, 1))
        coeffs = []
        for i in range(len(self.tau)):
            y = list(self.spectra[i].reshape(-1, 1))
            reg = LinearRegression().fit(X, y)
            coeffs.append([reg.coef_, reg.intercept_])  # b,a
            b = reg.coef_
            a = reg.intercept_
            self.spectra[i] = (self.spectra[i] - a) / b
        # return spectra

    # Spectrum visualization
    def spectrum_plot(self) -> None:
        """
        Visualizes the spectra.
        """
        plt.figure(figsize=(15, 10))
        plt.plot(self.lamdas, np.transpose(self.spectra))
        plt.ylabel("Absorbance", fontsize=25)
        plt.xlabel("λ (nm)", fontsize=25)
        plt.title("Spectra", fontsize=30)
        plt.show()

    # Finding number of significant components (For number of reactions)
    def variance_ratio(self) -> None:
        """
        Performs Principal Component Analysis (PCA) to determine the number of
        significant components (i.e., number of independent reactions) in the spectra.
        """
        u, s, vT = np.linalg.svd(self.spectra)
        cumsum = []
        for i in range(1, len(s) + 1):
            cumsum.append(sum(s[0:i]) / sum(s))
        count = 0
        for i in cumsum:
            if i <= 0.98:
                count = count + 1
        # print("No of independent reactions: ", count)
        plt.figure(figsize=(15, 10))
        plt.plot(
            [i for i in range(1, len(s) + 1)], cumsum, **{"color": "red", "marker": "o"}
        )
        plt.axhline(y=0.98, color="b", linestyle="-")
        plt.ylabel("Cumulative % variance captured by PC", fontsize=25)
        plt.xlabel("Number of principal components", fontsize=25)
        plt.show()

    # Frobenius norm function
    def norms(self, par: np.ndarray) -> float:
        """
        Calculates the Frobenius norm to measure the difference between
        the predicted and observed spectra.

        Args:
            par (ndarray): Array of parameters.

        Returns:
            float: Frobenius norm value.
        """
        if self.model_type != "custom":
            obj = models(self.N, self.model_type)
            C = odeint(obj.model, self.c0, self.tau, args=(par,))
            len_tau = self.tau.shape[0]
            val = (np.identity(len_tau) - C @ (np.linalg.pinv(C))) @ self.spectra
            return np.linalg.norm(val, "fro") ** 2
        else:
            C = odeint(self.fun, self.c0, self.tau, args=(par,))
            len_tau = self.tau.shape[0]
            val = (np.identity(len_tau) - C @ (np.linalg.pinv(C))) @ self.spectra
            return np.linalg.norm(val, "fro") ** 2

    # Maximum number of reactions
    def rmax(self, A: np.ndarray) -> int:
        """
        Computes the maximum number of reactions that can be estimated based on
        the given atomic matrix A.

        Args:
            A (ndarray): atomic matrix.

        Returns:
            int: Maximum number of reactions.
        """
        rank = np.linalg.matrix_rank(A)
        Rmax = len(A[0]) - rank
        return Rmax

    # Objective function with
    # bounds arg to give LB and UB of all parameters
    def optim(self, bounds=None, niters=500, algo='Nelder-Mead'):
        """
        Performs parameter optimization using the specified optimization algorithm.

        Args:
            bounds (list, optional): Bounds for the parameters. Defaults to None.
            niters (int, optional): Number of iterations. Defaults to 500.
            algo (str, optional): Optimization algorithm to use.
                                Defaults to "Nelder-Mead".

        Returns:
            tuple: Optimized parameters and residual value.
        """
        bnds = bounds

        if bounds is not None:
            for lb, ub in bounds:
                if lb > ub:
                    raise ValueError("Lower bound cannot be more than upper bound")
        if niters < 1 or not isinstance(niters, int):
            raise ValueError("niters must be a positive integer")
        fun1 = lambda x: self.norms(x)
        res1 = float("inf")
        ans = []
        for i in tqdm(range(niters)):
            params = []
            for j in range(self.n):
                params.append(random.uniform(bnds[j][0], bnds[j][1]))
            curr = minimize(
                fun1,
                tuple(params),
                method=algo,
                tol=1e-6,
                bounds=bnds,
                constraints=None,
            )
            ans1 = curr.x
            if self.norms(ans1) < res1:
                ans = curr.x
                res1 = self.norms(ans)
                # print(k_guess,v_guess,ans)
                # print(params,res1)
        return ans, res1

    def conc_profile(self, pars: np.ndarray) -> None:
        """
        Plots the concentration profile of the species over the residence time.

        Args:
            pars (ndarray): Parameters.

        Returns:
            None
        """
        C = None
        if self.model_type != "custom":
            obj = models(self.N, self.model_type)
            C = odeint(obj.model, self.c0, self.tau, args=(pars,))
        else:
            C = odeint(self.fun, self.c0, self.tau, args=(pars,))
        plt.figure(figsize=(15, 10))
        plt.plot(self.tau, C, **{"marker": "*"})
        plt.xlabel("τ (s)", fontsize=25)
        plt.ylabel("Concentration (mol/L)", fontsize=25)
        plt.show()

    def fun_arr(
        self,
        arr: list,
        bounds: list = None,
        niters: int = 500,
        algo: str = "Nelder-Mead",
    ) -> np.ndarray:
        """
        Given an array of kinetic models, selects the best one based on the
          residual of the optimization process.

        Args:
            arr (list): Array of kinetic models.
            bounds (list, optional): Bounds for the parameters. Defaults to None.
            niters (int, optional): Number of iterations. Defaults to 500.
            algo (str, optional): Optimization algorithm to use.
                                    Defaults to "Nelder-Mead".

        Returns:
            ndarray: Optimized parameters.
        """
        global_min = float("inf")
        eq = 0
        for k in range(len(arr)):
            par = par_estimate(
                self.spectra,
                self.c0,
                self.N,
                self.tau,
                self.lamdas,
                self.n,
                "custom",
                arr[k],
            )
            optims, residual = par.optim(bounds=bounds, niters=niters, algo=algo)
            # ans=optims
            if residual < global_min:
                eq = k
                global_min = residual
                ans = optims
        print("Best fit kinetic model : ", eq + 1)
        return ans

    # Forward EFA
    def forward_EFA(self, step: int, comp: int = 3) -> None:
        """
        Performs forward Evolving Factor Analysis (EFA).

        Args:
            step (int): Step size.
            comp (int, optional): Number of components. Defaults to 3.

        Returns:
            None
        """
        start = comp
        ans = []
        for f in range(start, self.spectra.shape[0], step):
            Yf = self.spectra[:f, :]
            u, s, vT = np.linalg.svd(Yf)
            # ans.append([s[0]*s[0],s[1]*s[1],s[2]*s[2]])
            ans.append([s[i] ** 2 for i in range(comp)])
        ans1 = np.log10(ans)
        x = [i for i in range(start, self.spectra.shape[0], step)]
        plt.figure(figsize=(15, 10))
        plt.plot(x, ans1, **{"color": "red", "marker": "o"})
        plt.ylabel("Log of eigen values", fontsize=25)
        plt.xlabel("Number of intervals", fontsize=25)

    # Backward EFA
    def backward_EFA(self, step: int, comp: int = 3) -> None:
        """
        Performs backward Evolving Factor Analysis (EFA).

        Args:
            step (int): Step size.
            comp (int, optional): Number of components. Defaults to 3.

        Returns:
            None
        """
        start = comp
        ans2 = []
        for f in range(self.spectra.shape[0] - start, 0, -step):
            Yf = self.spectra[f:, :]
            u, s, vT = np.linalg.svd(Yf)
            ans2.append([s[i] ** 2 for i in range(comp)])
        ans2 = np.log10(ans2)
        x = [i for i in range(self.spectra.shape[0] - start, 0, -step)]
        plt.figure(figsize=(15, 10))
        plt.plot(x, ans2, **{"color": "red", "marker": "o"})
        plt.ylabel("Log of eigen values", fontsize=25)
        plt.xlabel("Number of intervals", fontsize=25)
