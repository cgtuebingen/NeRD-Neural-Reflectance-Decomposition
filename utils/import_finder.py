import os


def is_hidden(filepath):
    name = os.path.relpath(filepath)
    return name.startswith(".")


def crawl_directories(path):
    cur_dir = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(f)]
    modules = [
        os.path.relpath(f) for f in cur_dir if os.path.isfile(f) and not is_hidden(f)
    ]
    for root, dirs, files in os.walk(path):
        if is_hidden(root) or "__pycache__" in root:
            continue
        else:
            modules.append(os.path.relpath(root))

            for f in files:
                if "__init__" in f or "__main__" in f:
                    continue
                modules.append(os.path.relpath(os.path.join(root, f)))

    return modules


def to_module(path):
    return os.path.splitext(path)[0].replace("/", ".")


def to_path(module):
    path_str = os.path.join(*[p for p in module.split(".")])
    if os.path.exists(path_str + ".py"):
        return path_str + ".py"
    return os.path.join(path_str, "__init__.py")


def find_files_to_copy(toCheck, base_path="./"):
    """
    Given a filename, returns a list of all source files in the current directory being
    used.
    """
    importedItems = [toCheck]

    if base_path == "":
        base_path = "./"

    src_files = crawl_directories(base_path)
    modulify = [to_module(p) for p in src_files]

    with open(toCheck, "r") as pyFile:
        for line in pyFile:
            # ignore comments
            line = line.strip().partition("#")[0].partition("as")[0].split(" ")

            if line[0] == "import" or line[0] == "from":
                if line[1] in modulify:
                    importedItems.append(to_path(line[1]))
                    importedItems += find_files_to_copy(to_path(line[1]))

    return sorted(list(set(importedItems)))
