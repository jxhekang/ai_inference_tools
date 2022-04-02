
indices = (-predictions[0]).argsort()[:5]
print("Class | Likelihood")
print(list(zip(indices, predictions[0][indices])))