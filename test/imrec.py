from sklearn import datasets
olivetti = datasets.fetch_olivetti_faces()
X, y = olivetti.data, olivetti.target
