module TrainingUtils

export setup

function setup(opt, model)
    return _setup(opt, model)
end

function _setup(opt, m::T) where T
    fields = fieldnames(T)
    values = [getfield(m, f) for f in fields]
    state = Dict()
    for (f, v) in zip(fields, values)
        if isnumericarray(v)
            state[f] = init_state(opt, v)
        elseif isstruct(v)
            state[f] = _setup(opt, v)
        end
    end
    return (optimizer = opt, state = state)
end

isnumericarray(x) = x isa AbstractArray{<:Number}
isstruct(x) = x isa NamedTuple || x isa StructTypes.StructType || (typeof(x).name.wrapper isa DataType && ismutable(x))

function init_state(opt, param)
    zero_mt = zero(param)
    zero_vt = zero(param)
    βp = [opt.beta[1], opt.beta[2]]
    return (mt = zero_mt, vt = zero_vt, βp = βp)
end

end
