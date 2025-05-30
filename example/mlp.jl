using Pkg
cd("example")
Pkg.activate("..")
using Revise

using JLD2
X_train = load("data/imdb_dataset_prepared.jld2", "X_train")
y_train = load("data/imdb_dataset_prepared.jld2", "y_train")
X_test = load("data/imdb_dataset_prepared.jld2", "X_test")
y_test = load("data/imdb_dataset_prepared.jld2", "y_test")

using NNLib

model = Chain([Dense(size(X_train, 1), 32, σ=ReLU), Dense(32, 1, σ=Sigmoid)])

loss(m, x, y) = mse_loss(m(x), y)
opt = Adam()

train!(model, X_train, y_train, X_test, y_test, loss, opt)