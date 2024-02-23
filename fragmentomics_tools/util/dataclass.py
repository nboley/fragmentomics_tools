import collections
import json
from copy import deepcopy
from dataclasses import asdict, fields, replace, MISSING
from typing import List, Tuple

import numpy
import numpy as np
import pandas
import typing
from dacite import from_dict, Config
from smart_open import open as smart_open

"""
from fbio.util.aws_utils import path_exists_s3_or_local
from fbio.util.iter_utils import only_one
from fbio.util.misc_utils import classproperty, isinstance_typing_union
"""


################# fbio.util.misc_utils ##################################
class ClassPropertyDescriptor(object):
    """
    used by classproperty() to make @classproperty decorator
    """

    def __init__(self, fget, fset=None):
        self.fget = fget
        self.fset = fset

    def __get__(self, obj, klass=None):
        if klass is None:
            klass = type(obj)
        return self.fget.__get__(obj, klass)()

    def __set__(self, obj, value):
        if not self.fset:
            raise AttributeError("can't set attribute")
        type_ = type(obj)
        return self.fset.__get__(obj, type_)(value)

    def setter(self, func):
        if not isinstance(func, (classmethod, staticmethod)):
            func = classmethod(func)
        self.fset = func
        return self


def classproperty(func):
    """
    @classproperty decorator
    """
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)

    return ClassPropertyDescriptor(func)


def isinstance_typing_union(x):
    """
    returns True if x is an instance of typing.Union (isinstance does not work)

    >>> isinstance_typing_union(Union[str, int])
    True
    >>> isinstance_typing_union(1)
    False
    >>> isinstance_typing_union(Union[str]) # Union[str] resolves to str
    False
    """
    return hasattr(x, "__origin__") and x.__origin__ == typing.Union


############## fbio.util.iter_utils
def only_one(itrbl, error_message=None):
    """
    >>> only_one(iter([1]))
    1
    >>> only_one(iter([]))
    Traceback (most recent call last):
    ...
    StopIteration: iterable had 0 items.
    >>> only_one(iter([]), 'custom message.')
    Traceback (most recent call last):
    ...
    StopIteration: iterable had 0 items. custom message.
    >>> only_one(iter([0, 1]))
    Traceback (most recent call last):
    ...
    ValueError: iterable had > 1 items. The first two items were 0 and 1.
    """
    if error_message is None:
        error_message = ""
    itrbl = iter(itrbl)
    try:
        a = next(itrbl)
    except StopIteration:
        raise StopIteration(
            f"iterable had 0 items.{' ' if error_message else ''}{error_message}"
        )

    try:
        b = next(itrbl)
        raise ValueError(
            f"iterable had > 1 items. The first two items were {a} and {b}."
            f"{' ' if error_message else ''}{error_message}"
        )
    except StopIteration:
        return a


##################################################################################################


def to_ndarray(x):
    return numpy.array(x)


class JSONEncodePlusNumpy(json.JSONEncoder):
    def default(self, obj):
        if obj.__class__.__module__ == numpy.__name__:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif obj.sum().__class__.__name__.startswith("int"):
                return int(obj)
            elif obj.sum().__class__.__name__.startswith("float"):
                return float(obj)
            else:
                raise TypeError(f"cannot convert {obj} to json")
        else:
            return json.JSONEncoder.default(self, obj)


class DataClassMixin:
    @classmethod
    def from_dict(cls, d, strict=True, check_types=True):
        if isinstance(d, cls):
            return deepcopy(d)

        # convert lists to tuples.
        # FIXME this doesn't handle converting lists nested inside another dict
        for field in cls.fields:
            if field.type in [
                Tuple,
                tuple,
                typing.Union[tuple, type(None)],
                typing.Union[type(None), tuple],
            ]:
                if isinstance(d[field.name], list):
                    d[field.name] = tuple(d[field.name])

        return from_dict(
            cls,
            d,
            config=Config(
                strict=strict,
                check_types=check_types,
                type_hooks={np.ndarray: to_ndarray},
            ),
        )

    def validate_types(self):
        # check types by serializing to json and unserializing
        if self.from_dict(self.to_dict()) != self:
            raise TypeError(self)
        return self

    def to_dict(self):
        return asdict(self)

    @classproperty
    def fields(self):
        return fields(self)

    @classproperty
    def field_names(self):
        return [field.name for field in self.fields]

    @classproperty
    def field_types(self):
        return [field.type for field in self.fields]

    @classproperty
    def field_name_to_type(self):
        return {field.name: field.type for field in self.fields}

    def replace(self, **changes):
        return replace(self, **changes)

    def save_to_json_s3_or_local(
        self, out_fname, validate_reload=False, allow_overwrite=False
    ):
        """
        :param out_fname: path to write to
        :param validate_reload: validate that we can reload the we saved
        :param allow_overwrite: allow overwriting out_fname
        :return:
        """
        if not allow_overwrite and path_exists_s3_or_local(out_fname):
            raise FileExistsError(f"'{out_fname}' already exists")
        with smart_open(out_fname, "w") as fp:
            json.dump(self.to_dict(), fp=fp, cls=JSONEncodePlusNumpy, indent=2)

        if validate_reload:
            type(self).from_json_s3_or_local(out_fname)

    @classproperty
    def names(cls):
        return [f.name for f in fields(cls)]

    @classproperty
    def dtypes(cls):
        """
        a dtypes dictionary of every field.name->field.type that can be used with pandas or numpy.
        An attribute must be a single type, or a Union[type, None] for this to work.

        ex: {'field1': str, 'field2': bool}
        """
        d = dict()
        for field in cls.fields:
            if isinstance_typing_union(field.type):
                # type is an instance of Union
                # we only support Union[typeA, None], in which case we set the field to `typeA`, otherwise raise
                # an eception
                union = field.type
                non_nones = [t for t in union.__args__ if t not in [None, type(None)]]
                if len(non_nones) == 0:
                    raise TypeError(
                        f"{field.name} does not have a valid type: {field.type}"
                    )
                elif len(non_nones) == 1:
                    # only one non None element in the typing.Union, add it to dtypes
                    d[field.name] = only_one(non_nones)
                elif len(non_nones) > 1:
                    raise TypeError(
                        f"{field.name} is has more than two types {field.type}, cannot select one"
                    )
                else:
                    raise AssertionError(
                        "impossible, there must be two or more __args__s"
                    )
            else:
                if isinstance(field.type, collections.Container):
                    raise TypeError(f"Cannot return a single dtype for field: {field}")
                d[field.name] = field.type

        return d

    @classmethod
    def from_json_s3_or_local(
        cls, fname, strict=True, check_types=True, force_values=None
    ):
        """
        :param fname: json file path
        :param strict: all keys must exist, there can be no extra keys
        :param check_types: check types match
        :param force_values: overrides values
        :return:
        """
        with smart_open(fname) as fp:
            d = json.load(fp)
        if force_values:
            for key, val in force_values.items():
                d[key] = val

        return cls.from_dict(d, strict=strict, check_types=check_types)

    @classmethod
    def load_dataframe(cls, in_tsv):
        pandas.read_table(in_tsv, names=cls.names, dtypes=cls.dtypes)

    @classmethod
    def dataframe_from_instances(cls, records):
        return pandas.DataFrame.from_dict(rec.to_dict() for rec in records)

    @classmethod
    def add_to_argparse(
        cls, parameter_group, group_name=None, underscores_to_dashes=True
    ):
        required_arguments = []
        optional_arguments = []

        for f in cls.fields:
            if f.type == bool:
                d = dict(action="store_true")
            elif f.type == dict:
                d = dict(type=json.loads)
            elif isinstance(f.type, typing._GenericAlias):
                if f.type.__origin__ in (list, tuple, List, Tuple):
                    # accept List[type] or Tuple[type]
                    assert (
                        len(f.type.__args__) == 1
                    ), f"{f.type} can only have a single type"
                    d = dict(type=only_one(f.type.__args__), nargs="*")
                elif f.type.__origin__ == typing.Union:
                    # accept Union[type, None] for optional arguments
                    args = [arg for arg in f.type.__args__ if arg is not type(None)]
                    assert len(args) == 1, f"{f.type} can only have a single type"
                    d = dict(type=only_one(args))
                else:
                    raise ValueError(f"cannot handle {f}")
            else:
                d = dict(type=f.type)

            if f.default_factory == MISSING:
                if f.default == MISSING:
                    # no default was set, this is a required field
                    d["required"] = True
                else:
                    d["default"] = f.default
            else:
                d["default"] = f.default_factory()

            name = f.name.replace("_", "-") if underscores_to_dashes else f.name

            if f.metadata.get("choices") is not None:
                d["choices"] = set(f.metadata.get("choices"))

            def get_help():
                help = f.metadata.get("help", "")
                if help.strip() != "" and not help.endswith("."):
                    help += "."

                if d.get("required"):
                    help += "  Required."

                if d.get("default", None) is not None:
                    help += f'  Default=`{d.get("default")}`.'

                if d.get("choices", None) is not None:
                    help += f'  choices={d.get("choices")}'

                return help

            d["help"] = get_help()

            if d.get("required"):
                required_arguments.append((name, d))
            else:
                optional_arguments.append((name, d))

        group_name_str = f" {group_name}" if group_name is not None else ""
        if len(required_arguments):
            required_args = parameter_group.add_argument_group(
                f"Required{group_name_str}"
            )

            for name, d in required_arguments:
                required_args.add_argument(f"--{name}", **d)

        if len(optional_arguments):
            optional_args = parameter_group.add_argument_group(
                f"Optional{group_name_str}"
            )

            for name, d in optional_arguments:
                optional_args.add_argument(f"--{name}", **d)

    @classmethod
    def from_argparse(cls, args):
        return cls(**{f.name: getattr(args, f.name) for f in cls.fields})
