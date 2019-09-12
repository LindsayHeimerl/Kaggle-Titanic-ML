import numpy as np
import sys
import random
import pandas as pd
from sklearn import tree
from sklearn import ensemble
from sklearn.model_selection import train_test_split as tts
import argparse

def parse_args():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("train", type=str)
    parser.add_argument("-dev", type=float, default=0.3)
    parser.add_argument("-t", type=int, default=5)
    parser.add_argument("-d", type = int, default = 4)
    parser.add_argument("-f", type = int, default = 4)
    return parser.parse_args()

def main():
    args = parse_args()
    train = pd.read_csv(args.train)
    X = train.loc[:,train.columns!='Survived']
    Y = train['Survived']
    x_train, x_dev, y_train, y_dev = tts(X, Y, test_size=args.dev)

    rfc = ensemble.RandomForestClassifier(criterion='entropy', max_depth=args.d, max_features=args.f, random_state=2)
    model = rfc.fit(x_train, y_train)
    print(model.score(x_train, y_train))
    print(model.score(x_dev, y_dev))

    '''
    clfs = []
    for i in range(0,args.t):
        clfs.append(tree.DecisionTreeClassifier(criterion='entropy', splitter = "best", max_depth = args.d, max_features = 4, random_state=random.randint(0,100)))

    for clf in clfs:
        model = clf.fit(x_train, y_train)
        print(model.score(x_train, y_train))
        print(model.score(x_dev, y_dev))
    '''
    

if __name__ == "__main__":
    main()