models = {}


def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def make(name, config):
    print('name', name)
    model = models[name](config)
    return model

def se_make(se_name, config):
    print('se_name', se_name)
    model = models[se_name](config)
    # print('model', model)    # SeVolumeDensity
    return model
# name nerf
# name volume-density
# name volume-radiance


from . import nerf, neus, geometry, texture
