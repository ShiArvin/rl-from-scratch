import time 
import numpy as np 
import torch 
import warnings 
class Result:
    def __init__(self, head:str):
        self.metrics = {}
        self.start_time = None
        self.end_time = None
        self.head = head    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.metrics[f"{self.head}/time"] = self.end_time - self.start_time

    
    def add_metric(self, name, value):
        if name in self.metrics:
            raise ValueError(f"Metric {name} already exists in the result, you are overwriting it.")
        
        if isinstance(value, np.ndarray) or isinstance(value, np.floating):
            if len(value.shape)>1:
                warnings.warn(f"Metric {name} is a numpy array with shape {value.shape}, we take the mean of it as the metric value.")
                value = np.mean(value) # if the value is a numpy array, we take the mean of it as the metric value
                
            value = value.item() # convert numpy scalar to python scalar
        elif isinstance(value, torch.Tensor):
            if len(value.shape)>1:
                warnings.warn(f"Metric {name} is a torch tensor with shape {value.shape}, we take the mean of it as the metric value.")
                value = value.mean() # if the value is a torch tensor, we take the mean of it as the metric value
            
            value = value.item() # convert torch scalar to python scalar
        elif isinstance(value, float):
            pass 
        else:
            raise ValueError(f"Metric value must be anint/float, numpy array or a torch tensor, but got {type(value)}")
        self.metrics[name] = value
    
    def add(self, result):
        """group other result into the current result"""
        for k,v in result.metrics.items():
            self.add_metric(k,v)

