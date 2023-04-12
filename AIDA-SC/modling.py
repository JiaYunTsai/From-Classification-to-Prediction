from scipy import sparse


def main():
    X = sparse.load_npz("tmp/limit_model.npz") #讀入稀疏矩陣
    X

if __name__ == '__main__':
    main()