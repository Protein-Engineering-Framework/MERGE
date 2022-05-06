# version         Beta
# date            27/01/2022
# author          Alexander-Maurice Illig,
# affilation      Institute of Biotechnology, RWTH Aachen
# email           a.illig@biotec.rwth-aachen.de

import pickle
import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from scipy.optimize import differential_evolution

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV,train_test_split

# how about new predictions? function that converts variant name to encoded sequence needed

class Hybrid_Model:

    alphas=np.logspace(-6,6,100) # Grid for the parameter 'alpha'.
    parameter_range=[(0,1), (0,1)] # Paramter range of 'beta_1' and 'beta_2' with lb <= x <= ub

    def __init__(self,
        wild_type_file:str,
        encoding_file:str,
        alphas=alphas,
        parameter_range=parameter_range
        ):
        self.x_wild_type=np.load(wild_type_file)
        self.df_encoding=pd.read_csv(encoding_file,sep=';',comment='#')
        self._alphas=alphas
        self._parameter_range=parameter_range

        self.variants,self.X,self.y=self._process_df_encoding()
        self._spearmanr_dca=self._spearmanr_dca()

    @staticmethod
    def _spearmanr(
        x1:np.ndarray,
        x2:np.ndarray
        ) -> float:
        """
        Parameters
        ----------
        x1 : np.ndarray
            Array of target fitness values.
        x2 : np.ndarray 
            Array of predicted fitness values.

        Returns
        -------
        Spearman's rank correlation coefficient.
        """
        return spearmanr(x1,x2)[0]

    def _process_df_encoding(self) -> tuple:
        """
        Extracts the array of names, encoded sequences, and fitness values
        of the variants from the dataframe 'self.df_encoding'.

        It is mandatory that 'self.df_encoding' contains the names of the
        variants in the first column, the associated fitness value in the
        second column, and the encoded sequence starting from the third
        column.

        Returns
        -------
        Tuple of variant names, encoded sequences, and fitness values.
        """
        return (
            self.df_encoding.iloc[:,0].to_numpy(),
            self.df_encoding.iloc[:,2:].to_numpy(),
            self.df_encoding.iloc[:,1].to_numpy()
            )

    def _delta_X(self,
        X:np.ndarray
        ) -> np.ndarray:
        """
        Subtracts for each variant the encoded wild-type sequence
        from its encoded sequence.
        
        Parameters
        ----------
        X : np.ndarray
            Array of encoded variant sequences.

        Returns
        -------
        Array of encoded variant sequences with subtracted encoded
        wild-type sequence.
        """
        return np.subtract(X,self.x_wild_type)

    def _delta_E(self,
        X:np.ndarray
        ) -> np.ndarray:
        """
        Calculates the difference of the statistical energy 'dE'
        of the variant and wild-type sequence.

        dE = E (variant) - E (wild-type)
        with E = \sum_{i} h_i (o_i) + \sum_{i<j} J_{ij} (o_i, o_j)

        Parameters
        ----------
        X : np.ndarray
            Array of the encoded variant sequences.

        Returns
        -------
        Difference of the statistical energy between variant 
        and wild-type.
        """
        return np.sum(self._delta_X(X),axis=1)

    def _spearmanr_dca(self) -> float:
        """
        Returns
        -------
        Spearman's rank correlation coefficient of the (full)
        data and the DCA predictions (difference of statistical
        energies).
        """
        y_dca=self._delta_E(self.X)
        return self._spearmanr(self.y,y_dca)

    def ridge_predictor(self,
        X_train:np.ndarray,
        y_train:np.ndarray,
        ) -> object:
        """
        Sets the parameter 'alpha' for ridge regression.

        Parameters
        ----------
        X_train : np.ndarray
            Array of the encoded sequences for training.
        y_train : np.ndarray
            Associated fitness values to the sequences present
            in 'X_train'.

        Returns
        -------
        Ridge object trained on 'X_train' and 'y_train' (cv=5)
        with optimized 'alpha'. 
        """
        grid=GridSearchCV(Ridge(fit_intercept=True),{'alpha':self._alphas},cv=5)
        grid.fit(X_train,y_train)
        return Ridge(**grid.best_params_).fit(X_train,y_train)
        
    def _y_hybrid(self,
        y_dca:np.ndarray,
        y_ridge:np.ndarray,
        beta_1:float,
        beta_2:float
        ) -> np.ndarray:
        """
        Chooses sign for connecting the parts of the hybrid model.

        Parameters
        ----------
        y_dca : np.ndarray
            Difference of the statistical energies of variants
            and wild-type.
        y_ridge : np.ndarray
            (Ridge) predicted fitness values of the variants.
        b1 : float
            Float between [0,1] coefficient for regulating DCA 
            model contribution.
        b2 : float
            Float between [0,1] coefficient for regulating ML 
            model contribution.

        Returns
        -------
        The predicted fitness value-representatives of the hybrid
        model.
        """
        if self._spearmanr_dca >= 0:
            return beta_1*y_dca + beta_2*y_ridge 
        
        else:
            return beta_1*y_dca - beta_2*y_ridge

    def _adjust_betas(self,
        y:np.ndarray,
        y_dca:np.ndarray,
        y_ridge:np.ndarray
        ) -> np.ndarray:
        """
        Find parameters that maximize the absolut Spearman rank
        correlation coefficient using differential evolution.

        Parameters
        ----------
        y : np.ndarray
            Array of fitness values.
        y_dca : np.ndarray
            Difference of the statistical energies of variants
            and wild-type.
        y_ridge : np.ndarray
            (Ridge) predicted fitness values of the variants.

        Returns
        -------
        'beta_1' and 'beta_2' that maximize the absolut Spearman rank correlation
        coefficient.
        """
        loss=lambda b: -np.abs(self._spearmanr(y, b[0]*y_dca + b[1]*y_ridge))

        minimizer=differential_evolution(loss, bounds=self.parameter_range, tol=1e-4)
        return minimizer.x

    def _settings(self,
        X_train:np.ndarray,
        y_train:np.ndarray,
        train_size_train=0.66,
        random_state=224
        ) -> tuple:
        """
        Get the adjusted parameters 'beta_1', 'beta_2', and the
        tuned regressor of the hybrid model.

        Parameters
        ----------
        X_train : np.ndarray
            Encoded sequences of the variants in the training set.
        y_train : np.ndarray
            Fitness values of the variants in the training set.
        train_size_train : float [0,1] (default 0.66)
            Fraction to split training set into another
            training and testing set.
        random_state : int (default=224)
            Random state used to split.

        Returns
        -------
        Tuple containing the adjusted parameters 'beta_1' and 'beta_2',
        as well as the tuned regressor of the hybrid model.
        """
        try:
            X_ttrain,X_ttest,y_ttrain,y_ttest=train_test_split(
                X_train,y_train,
                train_size=train_size_train,
                random_state=random_state
                )

        except ValueError:
            """
            Not enough sequences to construct a sub-training and sub-testing 
            set when splitting the training set.

            Machine learning/adjusting the parameters 'beta_1' and 'beta_2' not 
            possible -> return parameter setting for 'EVmutation' model.
            """
            return (1.0, 0.0, None)

        """
        The sub-training set 'y_ttrain' is subjected to a five-fold cross 
        validation. This leadss to the constraint that at least two sequences
        need to be in the 20 % of that set in order to allow a ranking. 

        If this is not given -> return parameter setting for 'EVmutation' model.
        """
        y_ttrain_min_cv=int(0.2*len(y_ttrain)) # 0.2 because of five-fold cross validation (1/5)
        if y_ttrain_min_cv < 2:
            return (1.0, 0.0, None)

        y_dca_ttest=self._delta_E(X_ttest)
        
        ridge=self.ridge_predictor(X_ttrain,y_ttrain)
        y_ridge_ttest=ridge.predict(X_ttest)

        beta1,beta2=self._adjust_betas(y_ttest,y_dca_ttest,y_ridge_ttest)
        return (beta1, beta2, ridge)

    def predict(self,
        X:np.ndarray,
        reg:object,
        beta_1:float,
        beta_2:float
        ) -> np.ndarray:
        """
        Use the regressor 'reg' and the parameters 'beta_1'
        and 'beta_2' for constructing a hybrid model and
        predicting the fitness associates of 'X'.

        Parameters
        ----------
        X : np.ndarray
            Encoded sequences used for prediction.
        reg : object
            Tuned ridge regressor for the hybrid model.
        beta_1 : float
            Float for regulating EVmutation model contribution.
        beta_2 : float
            Float for regulating Ridge regressor contribution.

        Returns
        -------
        Predicted fitness associates of 'X' using the
        hybrid model.
        """
        y_dca=self._delta_E(X)
        if reg == None:
            y_ridge=np.random.random(y_dca.size) # in order to suppress error
        else:
            y_ridge=reg.predict(X)
        return self._y_hybrid(y_dca,y_ridge,beta_1,beta_2)

    def performance(self,
        train_size=0.8,
        n_runs=10,
        seed=224,
        verbose=False,
        save_model=False
        ) -> np.ndarray:
        """
        Estimates performance of the model.

        Parameters
        ----------
        train_size : int or float (default=0.8)
            Number of samples in the training dataset
            or fraction of full dataset used for training.
        n_runs : int (default=10)
            Number of different splits to perform.
        seed : int (default=224)
            Seed for random generator.
        verbose : bool (default=False)
            Controls information content to be returned. If 'False'
            only Spearman's rank correlation coefficient is returned.
        save_model : bool (default=False)
            If True, model is saved using pickle, else not.

        Returns
        -------
        Returns array of spearman rank correlation coefficients
        if verbose=False, otherwise returns array of spearman
        rank correlation coefficients, cs, alphas, number of 
        samples in the training and testing set, respectively.
        """
        data=[]
        np.random.seed(seed)
        for random_state in np.random.randint(100,size=n_runs):
            X_train,X_test,y_train,y_test=train_test_split(self.X,self.y,train_size=train_size,random_state=random_state)
            beta_1,beta_2,reg=self._settings(X_train,y_train)
            if beta_2==0.0:
                alpha=np.nan
            else:
                if save_model:
                    pickle.dumps(reg)

                alpha=reg.alpha
            data.append([
                self._spearmanr(y_test,self.predict(X_test,reg,beta_1,beta_2)),
                alpha,beta_1,beta_2,y_train.size,y_test.size
                ])

        if verbose:
            return np.array(data).T
        else:
            return np.array(data).T[0]

    def _get_train_sizes(self
        ) -> np.ndarray:
        """
        Generates a list of train sizes to perform low-n with.

        Returns
        -------
        Numpy array of train sizes up to (and inlcuding) 80%.
        """
        eighty_percent=int(self.y.size*0.8)

        train_sizes=np.sort(np.concatenate([
            np.arange(15,50,5), np.arange(50,100,10),
            np.arange(100,150,20), [160,200,250,300,eighty_percent],
            np.arange(400,1100,100)
        ]))

        idx_max=np.where(train_sizes>=eighty_percent)[0][0]+1
        return train_sizes[:idx_max]

    def run(self,
        train_sizes:np.ndarray,
        data:list,
        n_runs=10,
        verbose=True,
        ) -> list:
        """
        Function for doing performance measurement of different
        'train_sizes' in a parallel manner.

        Parameters
        ----------
        train_sizes : np.ndarray
            Array of integers or floats for generating a training size.
        data : list
            Performance results are appended to this list.
        n_runs : int (default=10)
            Number of different splits to perform.
        verbose : bool (default=False)
            Controls information content to be returned. If 'False'
            only Spearman's rank correlation coefficient is returned.

        Returns
        -------
        data : list
            List containing the results of the performance study.
        """
        for train_size in train_sizes:
            data.append(self.performance(train_size=train_size,n_runs=n_runs,verbose=verbose))
        return data