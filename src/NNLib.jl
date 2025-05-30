module NNLib

include("Layers/Dense.jl")
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

using .Losses: binarycrossentropy, mse_loss
export binarycrossentropy, mse_loss

using .Optimizer: Adam, apply!, update!
export Adam, apply!, update!

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
