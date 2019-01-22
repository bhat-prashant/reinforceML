import logging


class BaseTransformer():
    def __init__(self, transformer=None, param_count=None):
        self.transformer = transformer
        self.param_count = param_count
        pass

    def set_transformer(self, transformer):
        if transformer is not None:
            self.transformer = transformer

    def get_transformer(self):
        return self.transformer

    def get_param_count(self):
        return self.param_count

    def set_param_count(self, param_count=None):
        self.param_count = param_count

    def transform(self, *args):
        if len(args) == self.param_count:
            return self.transformer(*args)
        else:
            logging.error("Error! Expecting %d parameters, instead got %d".format(self.param_count, len(*args)))
