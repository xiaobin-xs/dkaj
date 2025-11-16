import pandas as pd
import numpy as np
from typing import Sequence, Optional, Tuple

# Import rpy2 for R integration
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects import numpy2ri

class FineGrayCRR:
    """
    A Python wrapper for the R 'cmprsk' package to fit a competing risks
    regression model based on the method by Fine and Gray (1999).

    This class fits a separate proportional subdistribution hazards model for
    each specified cause of failure.
    """

    def __init__(self, 
                 n_causes: int,
                 cencode: int = 0,
                 gtol: float = 1e-6,
                 maxiter: int = 10000,
                 variance: bool = True,
                 interpolate: str = 'linear'):
        """
        Initializes the CRModel wrapper by setting up the rpy2 environment
        and importing the R 'cmprsk' package.

        Args:
            n_causes (int): The total number of competing event types (e.g., if events are 1 and 2, n_causes=2).
            cencode (int): The numerical code for censored observations.
            gtol (float): Tolerance for convergence.
            maxiter (int): Maximum number of iterations for the fitter.
            variance (bool): Whether to compute the variance-covariance matrix.
            interpolate (str): The method for interpolation, either 'linear' or 'step'.
        """
        # Activate converters for pandas and numpy objects
        pandas2ri.activate()
        numpy2ri.activate()
        
        # Import the R 'cmprsk' package 
        try:
            self.cmprsk = importr('cmprsk')
        except ro.rinterface_lib.embedded.RRuntimeError:
            raise ImportError(
                "R package 'cmprsk' not found. "
                "Please install it in your R environment using: install.packages('cmprsk')"
            )
        self.n_causes = n_causes
        self.cencode = int(cencode)
        self.gtol = float(gtol)
        self.maxiter = int(maxiter)
        self.variance = bool(variance)
        self.interpolate = interpolate
        self.models = {}
        self._fitted = False

    def fit(self, cov: pd.DataFrame, ftime: pd.Series, fstatus: pd.Series):
        """
        Fits a proportional subdistribution hazards model for each cause of failure.

        Args:
            cov (pd.DataFrame): DataFrame of fixed covariates.
            ftime (pd.Series): Series of failure or censoring times.
            fstatus (pd.Series): Series indicating the cause of failure or censoring.
        """
        # Convert inputs if they are numpy arrays
        if isinstance(cov, np.ndarray):
            cov = pd.DataFrame(cov, columns=[f'var_{i}' for i in range(cov.shape[1])])
        if isinstance(ftime, np.ndarray):
            ftime = pd.Series(ftime, name='ftime')
        if isinstance(fstatus, np.ndarray):
            fstatus = pd.Series(fstatus, name='fstatus')

        # Convert pandas objects to R objects for the crr function
        r_ftime = ro.conversion.py2rpy(ftime)
        r_fstatus = ro.conversion.py2rpy(fstatus)
        r_cov = ro.conversion.py2rpy(cov)
        
        # Fit one model per cause of failure
        for cause in range(1, self.n_causes + 1):
            print(f"Fitting model for cause: {cause}...")
            try:
                # Call the crr function from the cmprsk package 
                fitted_model = self.cmprsk.crr(
                    ftime=r_ftime,
                    fstatus=r_fstatus,
                    cov1=r_cov,
                    failcode=cause,  # Set the failure type of interest 
                    cencode=ro.IntVector([self.cencode])[0],
                    gtol=self.gtol,
                    maxiter=self.maxiter,
                    variance=self.variance,
                )
                self.models[cause] = fitted_model
                print(f"Successfully fitted model for cause: {cause}.")
            except Exception as e:
                print(f"An error occurred while fitting the model for cause {cause}: {e}")

        self._fitted = True
                
        # You can inspect a summary of a fitted model, for example:
        # r_summary = ro.r['summary']
        # print(r_summary(self.models[self.causes[0]]))


    def predict_cumulative_incidence(self, cov: pd.DataFrame, times: Sequence[float]) -> dict:
        """
        Generates predicted cumulative incidence probabilities for new data.

        This method uses the fitted models to predict the subdistribution function
        for each cause of failure based on a new set of covariates.

        Args:
            cov (pd.DataFrame): DataFrame with new covariate data.
            times (Sequence[float]): A sequence of time points at which to predict the CIFs.

        Returns:
            np.ndarray: A NumPy array of shape (n_samples, n_causes, n_times) containing the predicted CIFs.
        """

        if not self._fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        # Get the generic R 'predict' function
        r_predict = ro.r['predict']
        
        # Convert the new covariate DataFrame to an R matrix
        r_cov_new = ro.conversion.py2rpy(cov)

        times_used = np.asarray(times, dtype=float).reshape(-1)
        predictions = np.empty((cov.shape[0], self.n_causes, len(times_used)), dtype=float)

        for cause, model in self.models.items():
            # 1. Get the native prediction from R's predict.crr
            # This returns a matrix where column 0 is time, and others are CIFs
            pred_matrix_r = r_predict(model, cov1=r_cov_new)
            pred_np = np.array(pred_matrix_r)

            base_times = pred_np[:, 0]
            cif_matrix = pred_np[:, 1:]  # Shape: (n_base_times, n_samples)

            # 2. Interpolate to the desired time grid
            if self.interpolate == 'linear':
                # Result shape: (n_times, n_samples)
                interpolated_cifs = self._interp_linear(base_times, cif_matrix, times_used)
            else: # 'step'
                interpolated_cifs = self._interp_step(base_times, cif_matrix, times_used)

            # 3. Transpose to get (n_samples, n_times) and store
            predictions[:, cause-1, :] = interpolated_cifs.T

        return predictions
    

    @staticmethod
    def _interp_linear(base_times: np.ndarray, cif_mat: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Linear interpolation for each subject column."""
        out = np.empty((times.size, cif_mat.shape[1]), dtype=float)
        for j in range(cif_mat.shape[1]):
            # np.interp requires strictly increasing x-points
            t, uniq_idx = np.unique(base_times, return_index=True)
            y = cif_mat[uniq_idx, j]
            # Use the last known value for extrapolation
            out[:, j] = np.interp(times, t, y)
        return out # Shape: (n_times, n_subjects)

    @staticmethod
    def _interp_step(base_times: np.ndarray, cif_mat: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Right-continuous step function evaluation."""
        # Find the index of the last time in base_times that is less than or equal to each time in `times`
        indices = np.searchsorted(base_times, times, side="right") - 1
        
        # Clip indices to be within the valid range [0, len(base_times)-1]
        # For times before the first event, index will be -1, which we'll handle
        indices = np.clip(indices, 0, len(base_times) - 1)
        
        # Use advanced indexing to get the CIF values
        out = cif_mat[indices, :]
        
        # For any query time before the first event time, the CIF should be 0
        out[times < base_times[0]] = 0.0
        
        return out # Shape: (n_times, n_subjects)
    


class CauseSpecificCox:
    """
    A Python wrapper for the R 'riskRegression' package to fit a
    cause-specific Cox proportional hazards model.

    This class uses the CSC (Cause-Specific Cox) function to fit a model for
    each cause and then predicts the cumulative incidence function.
    """

    def __init__(self, n_causes, surv_type: str = "hazard", fitter: str = "glmnet"):
        """
        Initializes the model wrapper.

        Args:
            [cite_start]surv_type (str): Either "hazard" or "survival".
                - "hazard": Fits cause-specific Cox models for all causes.
                - "survival": Fits a model for the cause of interest and a model for the overall event-free survival.
            [cite_start]fitter (str): The R routine to fit the Cox models (e.g., "coxph", "cph", "glmnet").
        """
        # Activate converters for pandas and numpy
        pandas2ri.activate()
        numpy2ri.activate()

        try:
            self.riskRegression = importr('riskRegression')
            self.prodlim = importr('prodlim')
            self.survival = importr('survival')
        except ro.rinterface_lib.embedded.RRuntimeError as e:
            raise ImportError(
                f"A required R package is missing. Please install 'riskRegression', 'prodlim', and 'survival'. Details: {e}"
            )

        if surv_type not in ["hazard", "survival"]:
            raise ValueError("surv_type must be either 'hazard' or 'survival'")

        self.n_causes = n_causes
        self.surv_type = surv_type
        self.fitter = fitter
        self.model = None
        self.causes = []
        self._fitted = False

    def fit(self, cov: pd.DataFrame, ftime: pd.Series, fstatus: pd.Series):
        """
        Fits cause-specific Cox models.

        Args:
            cov (pd.DataFrame): DataFrame of covariates.
            ftime (pd.Series): Series of failure or censoring times.
            fstatus (pd.Series): Series indicating the cause of failure.
        """
        # Ensure inputs are pandas objects and combine them for the R formula
        if not isinstance(cov, pd.DataFrame):
            cov = pd.DataFrame(cov, columns=[f'var_{i}' for i in range(cov.shape[1])])
        if not isinstance(ftime, pd.Series):
            ftime = pd.Series(ftime, name='ftime')
        if not isinstance(fstatus, pd.Series):
            fstatus = pd.Series(fstatus, name='fstatus')

        # Combine data into a single DataFrame
        data_df = pd.concat([ftime, fstatus, cov], axis=1)
        # remove ftime == 0 entries as it is not permitted for Cox family
        data_df = data_df[ftime > 0]
        ftime = data_df[ftime.name]
        fstatus = data_df[fstatus.name]
        self.causes = sorted([c for c in fstatus.unique() if c != 0])
        self.causes = [int(c) for c in self.causes]
        if self.causes != list(range(1, self.n_causes + 1)):
            print(f"Warning: Expected causes 1 to {self.n_causes}, but found {self.causes} in fstatus.")

        # Create the R formula
        formula_str = f"Hist({ftime.name}, {fstatus.name}) ~ " + " + ".join(cov.columns)
        r_formula = ro.Formula(formula_str)

        print("Fitting models with R formula:", formula_str)

        # Convert the combined DataFrame to an R object
        r_data = ro.conversion.py2rpy(data_df)

        try:
            # Call the CSC function
            self.model = self.riskRegression.CSC(
                formula=r_formula,
                data=r_data,
                surv_type=self.surv_type,
                fitter=self.fitter,
                nfolds=5,  # For cross-validation if using glmnet
            )
            self._fitted = True
            print("Successfully fitted all cause-specific models.")
        except Exception as e:
            print(f"An error occurred while fitting the CSC model: {e}")

    def predict_cumulative_incidence(self, cov: pd.DataFrame, times: Sequence[float]) -> dict:
        """
        Predicts cumulative incidence functions (CIFs) for new data.

        Args:
            cov (pd.DataFrame): DataFrame with new covariate data.
            times (Sequence[float]): A sequence of time points for prediction.

        Returns:
            np.ndarray: A NumPy array of shape (n_samples, n_causes, n_times) containing the predicted CIFs.
        """
        if not self._fitted:
            raise RuntimeError("Model is not fitted. Please call fit() first.")

        if not isinstance(cov, pd.DataFrame):
            cov = pd.DataFrame(cov, columns=[f'var_{i}' for i in range(cov.shape[1])])

        # Convert new data and times to R objects
        r_cov_new = ro.conversion.py2rpy(cov)
        r_times = ro.FloatVector(times)

        predictions = np.empty((cov.shape[0], self.n_causes, len(times)), dtype=float)
        for cause in self.causes:
            try:
                # Use the predictRisk function from the riskRegression package
                pred_matrix_r = self.riskRegression.predictRisk(
                    self.model,
                    newdata=r_cov_new,
                    times=r_times,
                    cause=cause,
                    # product_limit=False,
                    # truncate=True
                )
                # The result is already in (n_samples, n_times) format
                predictions[:, cause-1, :] = np.array(pred_matrix_r)
            except Exception as e:
                print(f"Could not generate predictions for cause {cause}. Error: {e}")
                # If prediction fails, return an array of NaNs
                predictions[:, cause-1, :] = np.full((cov.shape[0], len(times)), np.nan)

        return predictions
    


class CompetingRiskRSF:
    """
    A Python wrapper for the R 'randomforestsrc' package to fit a
    Random Survival Forest model for competing risks.

    This class fits a single ensemble model that handles all causes simultaneously
    and can predict the cumulativ
    e incidence function (CIF) for each cause.
    """

    def __init__(self, n_causes,
                 ntree: Optional[int] = None,
                 mtry: Optional[int] = None,
                 nodesize: int = None,
                 nodedepth: Optional[int] = None,
                 ntime: Optional[int] = None,
                 seed: Optional[int] = 2025):
        """
        Initializes the model wrapper with specified hyperparameters.

        Args:
            ntree (int): Number of trees in the forest.
            mtry (int, optional): Number of variables randomly selected at each split.
                                  Defaults to sqrt(p) where p is the number of variables.
            nodesize (int): Minimum size of terminal nodes.
            nodedepth (int, optional): Maximum depth of a tree. Defaults to None (no limit).
            ntime (int, optional): Number of time points used to calculate the ensemble CIF.
                                   Defaults to a sequence of unique event times.
            seed (int, optional): Random seed for reproducibility. 
        """
        # Activate rpy2 converters
        pandas2ri.activate()
        numpy2ri.activate()

        try:
            self.rfsrc = importr('randomForestSRC')
        except ro.rinterface_lib.embedded.RRuntimeError as e:
            raise ImportError(
                "R package 'randomforestsrc' not found. Please install it. "
                f"Details: {e}"
            )

        # Store hyperparameters
        self.params = {
            'perf.type': ro.StrVector(["none"]), # Disable performance metric calculation for speed
            'seed': ro.IntVector([seed]) if seed is not None else None,
        }
        if ntree is not None:
            self.params['ntree'] = ro.IntVector([ntree])
        if nodesize is not None:
            self.params['nodesize'] = ro.IntVector([nodesize])
        if mtry is not None:
            self.params['mtry'] = ro.IntVector([mtry])
        if nodedepth is not None:
            self.params['nodedepth'] = ro.IntVector([nodedepth])
        if ntime is not None:
            self.params['ntime'] = ro.IntVector([ntime])

        self.n_causes = n_causes
        self.seed = seed
        self.model = None
        self.causes = []
        self._fitted = False

    def fit(self, cov: pd.DataFrame, ftime: pd.Series, fstatus: pd.Series):
        """
        Fits the Random Survival Forest for competing risks.

        Args:
            cov (pd.DataFrame): DataFrame of covariates.
            ftime (pd.Series): Series of failure or censoring times.
            fstatus (pd.Series): Series indicating the cause of failure (0 for censoring).
        """
        # Ensure inputs are pandas objects
        if not isinstance(cov, pd.DataFrame):
            cov = pd.DataFrame(cov, columns=[f'var_{i}' for i in range(cov.shape[1])])
        if not isinstance(ftime, pd.Series):
            ftime = pd.Series(ftime, name='ftime')
        if not isinstance(fstatus, pd.Series):
            fstatus = pd.Series(fstatus, name='fstatus')

        # Combine into a single DataFrame for the R formula
        data_df = pd.concat([ftime, fstatus, cov], axis=1)
        self.causes = sorted([c for c in fstatus.unique() if c != 0])
        if not self.causes:
            raise ValueError("No failure events found in 'fstatus'.")

        # Create the R formula (e.g., "Surv(ftime, fstatus) ~ x1 + x2")
        formula_str = f"Surv({ftime.name}, {fstatus.name}) ~ " + " + ".join(cov.columns)
        r_formula = ro.Formula(formula_str)

        # print("Fitting RSF model with R formula:", formula_str)
        # print("Using hyperparameters:", {k: v[0] for k, v in self.params.items()})

        # Convert data to an R object
        r_data = ro.conversion.py2rpy(data_df)

        try:
            ro.r(f'set.seed({self.seed})')
            # Call the rfsrc function with stored parameters
            self.model = self.rfsrc.rfsrc(
                formula=r_formula,
                data=r_data,
                **self.params
            )
            self._fitted = True
            # print("Successfully fitted the RSF model.")
        except Exception as e:
            print(f"An error occurred while fitting the model: {e}")

    def predict_cumulative_incidence(self, cov: pd.DataFrame, times: Sequence[float]) -> dict:
        """
        Predicts cumulative incidence functions (CIFs) with linear interpolation.

        Args:
            cov (pd.DataFrame): DataFrame with new covariate data.
            times (Sequence[float]): A sequence of time points for which to predict the CIFs.

        Returns:
            dict: A dictionary where keys are failure causes and values are NumPy
                  arrays of shape (n_samples, n_times) with the predicted CIFs.
        """
        if not self._fitted:
            raise RuntimeError("Model is not fitted. Please call fit() first.")

        if not isinstance(cov, pd.DataFrame):
            cov = pd.DataFrame(cov, columns=[f'var_{i}' for i in range(cov.shape[1])])

        # Get the generic R predict function
        r_predict = ro.r['predict']
        r_cov_new = ro.conversion.py2rpy(cov)

        # Generate predictions from the fitted model
        pred_obj = r_predict(self.model, newdata=r_cov_new)

        # Extract the base time grid and the 3D CIF array from the prediction object
        base_times = np.array(pred_obj.rx2('time.interest'))
        # CIF array shape: (n_samples, n_base_times, n_causes)
        cif_3d_array = np.array(pred_obj.rx2('cif'))

        # Prepare for interpolation
        times_used = np.asarray(times, dtype=float).reshape(-1)
        predictions = np.empty((cov.shape[0], self.n_causes, len(times)), dtype=float)

        for i, cause in enumerate(self.causes):
            # Extract CIF matrix for the current cause: (n_samples, n_base_times)
            cif_matrix_cause = cif_3d_array[:, :, i]
            
            # Interpolate. Note: input to _interp_linear needs (n_times, n_samples)
            interpolated_cifs = self._interp_linear(
                base_times, cif_matrix_cause.T, times_used
            ) # Returns (n_user_times, n_samples)
            
            # Transpose to final desired shape (n_samples, n_user_times)
            predictions[:, int(cause)-1, :] = interpolated_cifs.T

        return predictions

    @staticmethod
    def _interp_linear(base_times: np.ndarray, cif_mat: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Linear interpolation for each subject column."""
        out = np.empty((times.size, cif_mat.shape[1]), dtype=float)
        for j in range(cif_mat.shape[1]):
            t, uniq_idx = np.unique(base_times, return_index=True)
            y = cif_mat[uniq_idx, j]
            # Interpolate, setting left boundary to 0 and extrapolating with the last known value
            out[:, j] = np.interp(times, t, y, left=0.0, right=y[-1] if len(y) > 0 else 0.0)
        return out # Shape: (n_times, n_subjects)