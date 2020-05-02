CONFIGS = {}


def register_config(config_name):
    """
    Decorator for registering config class.
    :param config_name:
    :return:
    """
    def decorator(f):
        CONFIGS[config_name] = f
        return f

    return decorator


def get_config(config):
    """
    Returns config class if registered.
    :param config:
    :return:
    """
    if config in CONFIGS:
        config = CONFIGS[config]

        return config
    else:
        raise ValueError("Config not found: %s", config)
