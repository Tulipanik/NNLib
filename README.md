# NNLib - Train your own Neural Network!
NNLib is a library for creating your own neural network. It allows users to create network of their desires and modify it as they like.

### For now library supports some elements like:
- `Chain` structure for building neural model
- Dense layers and it's parameters like:
> - setting inputs count
> - setting output count
> - setting activation function like ReLU or Sigmoid
- Loss functions like:
> - Mean Square Loss `mse_loss`
> - Binary Cross entrophy `binary_cross_entrophy`
- Using Adam Optimiser
- `train!` function that allows easy train process

## Installation
To install please use block of code below:

```julia
using Pkg
Pkg.add("NNLib")
```

Now you have access for various functions for Building NN :).

## Example usage
First you have to upload your jld2 data. To do it use package `JLD2`. Upload training X and Y data and testing X and Y data.

Then write some neural parameters like so:

```julia
model = Chain([Dense(size(X_train, 1), 32, σ=ReLU), Dense(32, 1, σ=Sigmoid)])

loss(m, x, y) = mse_loss(m(x), y)
opt = Adam()
```

Now you can superpass these parameters to `train!` function:

```julia
train!(model, X_train, y_train, X_test, y_test, loss, opt)
```

Now your network is trained :).

Also `train!` function supports parameters like:
- accuracy - which is function that counts accuracy of your model,
- epoch - which is count of epochs

if you wanna be even more custom :).

## Licence
Project is under licence. Please do not reuse and repost without consent.