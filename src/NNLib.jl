module NNLib

# Write your package code here.
include("Layers/Dense.jl")
# include("Layers/Activation.jl")
include("Losses.jl")
include("Model.jl")
include("Optimizers.jl")
include("Training.jl")
include("DataLoader.jl")
include("Utils.jl")
include("Gradient.jl")

using .DenseLayer: Dense
export Dense

using .Model: Chain
export Chain

# export mse_loss
using .Losses: binarycrossentropy
export binarycrossentropy

using .Optimizer: Adam, apply!
export Adam, apply!

using .Train: train!
export train!

using .DataLoaderModule: DataLoader
export DataLoader

using .TrainingUtils: setup
export setup

using .Gradients: gradient
export gradient

using AutoDiffLib: ReLU, Sigmoid, Variable
export ReLU, Sigmoid, Variable

end
