module DataLoaderModule

using Random

export DataLoader

struct DataLoader{T}
    data::T
    batchsize::Int
    shuffle::Bool
end

"""
    DataLoader(data; batchsize=1, shuffle=false)

Tworzy prosty DataLoader jak w Flux. Obsługuje macierze, krotki i namedtuple.
"""
function DataLoader(data; batchsize::Int=1, shuffle::Bool=false)
    DataLoader(data, batchsize, shuffle)
end

Base.IteratorSize(::Type{<:DataLoader}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{<:DataLoader}) = Base.EltypeUnknown()

function Base.iterate(dl::DataLoader, state=1)
    n = numobs(dl.data)
    idxs = dl.shuffle ? randperm(n) : 1:n
    _iterate(dl, idxs, state)
end

function _iterate(dl::DataLoader, idxs, state)
    n = numobs(dl.data)
    bs = dl.batchsize
    if state > n
        return nothing
    end
    inds = state:min(state+bs-1, n)
    batch = getbatch(dl.data, idxs[inds])
    return batch, state+bs
end

numobs(x::AbstractArray) = size(x, ndims(x))
numobs(x::Tuple) = numobs(x[1])
numobs(x::NamedTuple) = numobs(values(x)[1])

getbatch(x::AbstractArray, idxs) = selectdim(x, ndims(x), idxs)
getbatch(x::Tuple, idxs) = map(d -> getbatch(d, idxs), x)
getbatch(x::NamedTuple, idxs) = NamedTuple{keys(x)}(getbatch.(values(x), Ref(idxs)))

end