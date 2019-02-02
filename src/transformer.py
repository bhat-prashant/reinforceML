#!/usr/bin/env python
__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"


class Output_Array(object):
    """Output data type of pipelines."""
    pass


class ARGType(object):
    """Base class for parameter specifications."""
    pass


class BaseTransformer(object):
    # class variables
    root = False
    arg_types = None


# create argument class dynamically
def ArgClassGenerator(name, range, BaseClass=ARGType):
    return type(name, (BaseClass,), {'values': range})


# create transformer class dynamically
def TransformerClassGenerator(name, transformerdict, BaseClass=BaseTransformer, ArgClass=ARGType):
    arg_types = []
    for arg in transformerdict['params']:
        arg_types.append(ArgClassGenerator(arg, transformerdict['params'][arg], ArgClass))
    profile = {'root': transformerdict['root'], 'arg_types': arg_types}
    return type(name, (BaseClass,), profile)
