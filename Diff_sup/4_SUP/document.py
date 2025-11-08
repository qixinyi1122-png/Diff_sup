import os

def print_tree(root, prefix=""):
    """递归打印目录树"""
    files = sorted(os.listdir(root))
    for i, name in enumerate(files):
        path = os.path.join(root, name)
        connector = "└── " if i == len(files) - 1 else "├── "
        print(prefix + connector + name)
        if os.path.isdir(path):
            extension = "    " if i == len(files) - 1 else "│   "
            print_tree(path, prefix + extension)

if __name__ == "__main__":
    folder = "DiffusionForensics"  # 换成你要展示的目录路径
    print(folder + "/")
    print_tree(folder)