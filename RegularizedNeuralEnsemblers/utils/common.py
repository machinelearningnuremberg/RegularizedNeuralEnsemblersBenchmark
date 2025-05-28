from __future__ import annotations

import inspect
import random
from functools import partial, wraps
from typing import Any

import numpy as np
import torch


def move_to_device(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        results = method(self, *args, **kwargs)
        if self.device and isinstance(results, (tuple, list)):
            return tuple(x.to(self.device) if hasattr(x, "to") else x for x in results)
        return results

    return wrapper


def has_instance(collection, *types):
    return any([isinstance(el, typ) for el in collection for typ in types])


def filter_instances(collection, *types):
    return [el for el in collection if any([isinstance(el, typ) for typ in types])]


def get_rnd_state() -> dict:
    np_state = list(np.random.get_state())
    np_state[1] = np_state[1].tolist()
    state = {
        "random_state": random.getstate(),
        "np_seed_state": np_state,
        "torch_seed_state": torch.random.get_rng_state().tolist(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_seed_state"] = [
            dev.tolist() for dev in torch.cuda.get_rng_state_all()
        ]
    return state


def set_rnd_state(state: dict):
    # rnd_s1, rnd_s2, rnd_s3 = state["random_state"]
    random.setstate(
        tuple(
            tuple(rnd_s) if isinstance(rnd_s, list) else rnd_s
            for rnd_s in state["random_state"]
        )
    )
    np.random.set_state(tuple(state["np_seed_state"]))
    torch.random.set_rng_state(torch.ByteTensor(state["torch_seed_state"]))
    if torch.cuda.is_available() and "torch_cuda_seed_state" in state:
        torch.cuda.set_rng_state_all(
            [torch.ByteTensor(dev) for dev in state["torch_cuda_seed_state"]]
        )


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def is_partial_class(obj):
    """Check if the object is a (partial) class, or an instance"""
    if isinstance(obj, partial):
        obj = obj.func
    return inspect.isclass(obj)


# Taken from https://github.com/automl/neps/blob/master/src/metahyper/utils.p
def instance_from_map(
    mapping: dict[str, Any],
    request: str | list | tuple | Any,
    name: str = "mapping",
    allow_any: bool = True,
    as_class: bool = False,
    kwargs: dict | None = None,
):
    """Get an instance of an class from a mapping.

    Arguments:
        mapping: Mapping from string keys to classes or instances
        request: A key from the mapping. If allow_any is True, could also be an
            object or a class, to use a custom object.
        name: Name of the mapping used in error messages
        allow_any: If set to True, allows using custom classes/objects.
        as_class: If the class should be returned without beeing instanciated
        kwargs: Arguments used for the new instance, if created. Its purpose is
            to serve at default arguments if the user doesn't built the object.

    Raises:
        ValueError: if the request is invalid (not a string if allow_any is False),
            or invalid key.
    """

    # Split arguments of the form (request, kwargs)
    args_dict = kwargs or {}
    if isinstance(request, tuple) or isinstance(request, list):
        if len(request) != 2:
            raise ValueError(
                "When building an instance and specifying arguments, "
                "you should give a pair (class, arguments)"
            )
        request, req_args_dict = request
        if not isinstance(req_args_dict, dict):
            raise ValueError("The arguments should be given as a dictionary")
        args_dict = {**args_dict, **req_args_dict}

    # Then, get the class/instance from the request
    if isinstance(request, str):
        if request in mapping:
            instance = mapping[request]
        else:
            raise ValueError(f"{request} doesn't exists for {name}")
    elif allow_any:
        instance = request
    else:
        raise ValueError(f"Object {request} invalid key for {name}")

    # Check if the request is a class if it is mandatory
    if (args_dict or as_class) and not is_partial_class(instance):
        raise ValueError(
            f"{instance} is not a class and can't be used with additional arguments"
        )

    # Give the arguments to the class
    if args_dict:
        instance = partial(instance, **args_dict)

    # Return the class / instance
    if as_class:
        return instance
    if is_partial_class(instance):
        try:
            instance = instance()
        except TypeError as e:
            raise TypeError(f"{e} when calling {instance} with {args_dict}") from e
    return instance
