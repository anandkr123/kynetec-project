import pytest

from kedro.framework.project import settings
from kedro.config import ConfigLoader
from kedro.framework.context import KedroContext
from kedro.framework.hooks import _create_hook_manager

from kynetec_project.pipelines.data_engineering.nodes import filter_dataset, fix_datatypes
import pandas as pd
import numpy as np


def test_filter_dataset():
    basic_data = pd.DataFrame({
        'capital_city': ['delhi', 'washington', 'stockholm',
                         'berlin', 'tokyo', 'sofia',
                         'london', 'moscow', 'prague'],
        'country': ['india', 'usa', 'sweden',
                    'germany', 'japan', 'bulgaria',
                    'uk', 'russia', 'czech republic']
    })

    filters = {
        'columns': ['capital_city'],
        'values': ['delhi', 'washington', 'tokyo', 'sofia',
                   'london', 'moscow'],
        'columns_exclude': ['country'],
        'values_exclude': ['russia']
    }
    output = filter_dataset(basic_data, filters)
    assert output.equals(pd.DataFrame({
        'capital_city': ['delhi', 'washington', 'tokyo', 'sofia', 'london'],
        'country': ['india', 'usa', 'japan', 'bulgaria', 'uk']
    }))


def test_fix_datatypes():
    basic_data = pd.DataFrame({
        'pincode': ['20346', '20345', '24114', '235144', '23213']
    })
    datatypes = {
        'columns': ['pincode'],
        'datatypes': ['float64']
    }
    output = fix_datatypes(basic_data, datatypes)
    assert output.dtypes[0] is np.dtype('float64')

