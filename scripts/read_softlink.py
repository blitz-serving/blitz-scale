import os


def get_abspath(path) -> str:
    if os.path.islink(path) and os.path.isdir(os.readlink(path)):
        return os.path.abspath(os.readlink(path))
    elif os.path.isdir(path):
        return os.path.abspath(path)
    else:
        raise FileNotFoundError(f"{path} is not a valid directory")


print(get_abspath("log_home"))
print(get_abspath("log"))
