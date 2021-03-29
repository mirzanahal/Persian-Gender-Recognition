from classifier import Classifier

def train(X, y, model):
    classifier = Classifier(model)
    classifier.fit(X, y)
    return classifier

