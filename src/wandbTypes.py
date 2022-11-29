from typing import List, TypeVar, TypedDict, Union, Literal, Dict


class sweepMetric(TypedDict):
    name: str
    goal: Union[Literal["minimize"], Literal["maximize"]]


T = TypeVar('T')


class parameterWithValue(TypedDict):
    values: List[T]


class parameterWithMaxMin(TypedDict):
    max: float
    min: float


class sweepConfig(TypedDict):
    method: Union[Literal["random"], Literal["grid"], Literal["bayes"]]
    name: Union[None, str]
    metric: sweepMetric
    parameters: Dict[str, Union[parameterWithValue, parameterWithMaxMin]]


class defaultParameters(TypedDict):
    batch_size: parameterWithValue
    n_epoch: parameterWithMaxMin
    lr: parameterWithMaxMin


def seep_config_with_default(method: str = "random", name: str = "sweep", metric: sweepMetric = {"goal": "minimize", "name": "g_loss"}, parameters: defaultParameters = {"batch_size": {"values": [32]}, "lr": {"max": 0.1, "min": 0.000001}, "n_epoch": {"max": 500, "min": 100}}) -> sweepConfig:
    config: sweepConfig = {
        "method": method,
        "name": name,
        "metric": metric,
        "parameters": parameters,
    }
    return config
