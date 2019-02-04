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
def ArgClassGenerator(transformer_name, arg_name, range, BaseClass=ARGType):
    return type(transformer_name + '_' + arg_name, (BaseClass,), {'values': range, 'name':arg_name})


# create transformer class dynamically
def TransformerClassGenerator(name, transformerdict, BaseClass=BaseTransformer, ArgClass=ARGType):
    arg_types = []
    for arg in transformerdict['params']:
        arg_types.append(ArgClassGenerator(name, arg, transformerdict['params'][arg], ArgClass))
    # build class attributes
    profile = {'root': transformerdict['root'], 'package': transformerdict['package'],
               'transformer': transformerdict['transformer'], 'arg_types': arg_types
               }
    return type(name, (BaseClass,), profile)
