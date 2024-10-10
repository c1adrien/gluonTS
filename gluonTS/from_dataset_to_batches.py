from typing import List, Optional, Callable, Iterable
from itertools import islice

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from gluonts.dataset.repository import get_dataset
from gluonts.dataset.pandas import PandasDataset
from gluonts.transform import (
    Transformation,
    Chain,
    RemoveFields,
    SetField,
    AsNumpyArray,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AddAgeFeature,
    VstackFeatures,
    InstanceSplitter,
    ValidationSplitSampler,
    TestSplitSampler,
    ExpectedNumInstanceSampler,
    MissingValueImputation,
    DummyValueImputation,
)
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.common import Dataset
from gluonts.itertools import Cyclic
import torch 
from gluonts.dataset.loader import as_stacked_batches


PREDICTION_INPUT_NAMES = [
    "feat_static_cat",
    "feat_static_real",
    "past_time_feat",
    "past_target",
    "past_observed_values",
    "future_time_feat",
]

TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
    "future_target",
    "future_observed_values",
]
TRAINING_INPUT_NAMES = ["feat_static_cat"]

class LoadingSimpleDatasetAndCreatingBatches():
    """a demo class to understanding 
        1. Transformations
        2. Generation/creation of a dataset 
        3. creation of a loading dataclass to pass through a model"""
    
    def __init__(self):
        #1. We create a dataframe with high dimension, and with many dynamics features. 
        url = (
            "https://gist.githubusercontent.com/rsnirwan/a8b424085c9f44ef2598da74ce43e7a3"
            "/raw/b6fdef21fe1f654787fa0493846c546b7f9c4df2/ts_long.csv"
        )
        df = pd.read_csv(url, index_col=0, parse_dates=True)
        df["dynamic_item1"] = np.random.normal(0,1,len(df))
        df["dynamic_item2"] = np.random.normal(0,1,len(df))
        df["dynamic_item3"] = np.random.normal(0,1,len(df))
        

        #2. We convert it to a PandasDataset 
        self.dataset = PandasDataset.from_long_dataframe(df,
                                                         item_id="item_id",
                                                         target="target",
                                                         feat_dynamic_real=["dynamic_item1","dynamic_item2","dynamic_item3"]) 
        """plusieurs choses à dire sur le PandasDataset:
            1. on donne un argumment le dataset, la position de target, le nom des features dynamics, et le nom de item_id qui sert à distringuer les différentes colonnes
            2. L'objet renvoyé est un PandasDataset, qui est un iétrable. On a un dictionnaire par dimension, qui contient grosso-modo :    
                                                                                            a. les caractéristiques de base de la time serie i.e. freq, 
                                                                                            b. le numpy array avec le target 
                                                                                            c. un numpy array multidimensionnel avec les differentes dynamic_features.
                                                                                            
        On a donc ce format, pour chaque  item_id present.                                                                
        {'start': Period('2021-01-01 00:00', 'H'), 'target': array([-1.3378e+00, -1.6111e+00, -1.9259e+00, -1.9184e+00, -1.9168e+00,
       -1.968...1, -1.2090e-01, -5.0720e-01, -6.6610e-01]), 'item_id': 'A', 'feat_dynamic_real': array([[ 0.00874503, -0.18895264,  0.19976967,  0.88544543, -0.60249421,
        -0.6..., -0.35331075,  1.46276465,  0.80923735]])}
                                                                                                                            """
        self.instance = self.create_instance_splitter("training").apply(self.dataset, is_train=True) #application : on a un TransformedDataset

        self.dataloader = self.create_training_data_loader(self.dataset)


    def create_instance_splitter(self,mode: str):
        assert mode in ["training", "validation", "test"]

        instance_sampler = {
            "training":  ExpectedNumInstanceSampler(
                num_instances=1.0, min_future=12
            ),
            "validation":  ExpectedNumInstanceSampler(
                num_instances=1.0, min_future=12
            ),
            "test": TestSplitSampler(),}[mode]

        return InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=12, #pas vraiment besoin du module ici. Plus une version soft. 
            future_length=10,
            #time_series_fields=[
            #    FieldName.FEAT_TIME,
            #    FieldName.OBSERVED_VALUES,
            #],
        # dummy_value=self.distr_output.value_in_support,
        )


    def create_training_data_loader(self,data: Dataset,shuffle_buffer_length: Optional[int] = None,**kwargs) -> Iterable:
        data = Cyclic(data).stream() #permet de construire un itérable 
        
        return as_stacked_batches(
            self.instance,
            batch_size=40,
            shuffle_buffer_length=shuffle_buffer_length,
            field_names=["feat_dynamic_real","future_target","past_target"], #on donne ici tous les noms de tous les featutres dans le dataset. i.e.
            output_type=torch.tensor,
            num_batches_per_epoch=20,
        )

        


        


if __name__ == "__main__":
    loader = LoadingSimpleDatasetAndCreatingBatches()
    #create_training_data_loader(loader.dataset)
    loader
