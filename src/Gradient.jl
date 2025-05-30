module Gradients

using Debugger
using AutoDiffLib
using Statistics

export gradient

function gradient(f, model)
    loss = f(model)

    z = topological_sort(loss)

    backward!(z)
    # @show z[1].grad
    # exit()
    # @show Statistics.mean(loss.grad)

    grads = [(layer.W.grad, layer.b.grad) for layer in model.layers]
    # @show Statistics.mean(grads[2][1])
    # @show Statistics.mean(grads[2][2])

    return grads
end

function gradient(f::Function)
    return model -> gradient(f, model)
end

end