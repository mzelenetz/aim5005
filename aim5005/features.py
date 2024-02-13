import numpy as np
from typing import List, Tuple
### YOU MANY NOT ADD ANY MORE IMPORTS (you may add more typing imports)

class MinMaxScaler:
    def __init__(self):
        self.minimum = None
        self.maximum = None
        
    def _check_is_array(self, x:np.ndarray) -> np.ndarray:
        """
        Try to convert x to a np.ndarray if it'a not a np.ndarray and return. If it can't be cast raise an error
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            
        assert isinstance(x, np.ndarray), "Expected the input to be a list"
        return x
        
    
    def fit(self, x:np.ndarray) -> None:   
        x = self._check_is_array(x)
        self.minimum=x.min(axis=0)
        self.maximum=x.max(axis=0)
        
    def transform(self, x:np.ndarray) -> list:
        """
        MinMax Scale the given vector
        """
        x = self._check_is_array(x)
        diff_max_min = self.maximum - self.minimum
        
        # TODO: There is a bug here... Look carefully! 
        return (x-self.minimum) / diff_max_min
    
    def fit_transform(self, x:list) -> np.ndarray:
        x = self._check_is_array(x)
        self.fit(x)
        return self.transform(x)
    
    
class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None
    
    def _check_is_array(self, x:np.ndarray) -> np.ndarray:
        """
        Try to convert x to a np.ndarray if it'a not a np.ndarray and return. If it can't be cast raise an error
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            
        assert isinstance(x, np.ndarray), "Expected the input to be a list"
        return x
        
    
    def fit(self, x:np.ndarray) -> None:   
        x = self._check_is_array(x)
        self.mean=x.mean(axis=0)
        self.std=x.std(axis=0)
    
    def transform(self, x:np.ndarray) -> list:
        x = self._check_is_array(x)

        return (x - self.mean) / self.std
    
    def fit_transform(self, x:list) -> np.ndarray:
        x = self._check_is_array(x)
        self.fit(x)
        return self.transform(x)