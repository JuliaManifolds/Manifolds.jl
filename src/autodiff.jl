# Wrapper for current backend as `Symbol`
mutable struct CurrentADBackend
    backend::Symbol
end

const _current_adbackend = CurrentADBackend(:none)
const _adbackends = Symbol[]

"""
    adbackend(backend::Symbol = :default) -> Symbol

Check if autodiff `backend` is valid (see [`adbackends`](@ref)) and return
formatted. Note that `backend=:default` returns the current backend.
"""
adbackend() = _current_adbackend.backend
adbackend(backend::Symbol) = adbackend(Val(backend))
adbackend(backend::Val{:default}) = adbackend()

function adbackend(backend::Val{T}) where {T}
    T in _adbackends && return T
    return throw(ArgumentError("Invalid AD backend $(T). Valid options are $(adbackends())."))
end

"""
    adbackend!(backend::Symbol)

Set current backend for autodiff to `backend`.
"""
function adbackend!(backend)
    backend = adbackend(backend)
    _current_adbackend.backend = backend
    return backend
end

"""
    adbackends() -> Vector{Symbol}

Get vector of currently valid AD backends.
"""
adbackends() = _adbackends

"""
    _gradient(f, p::Number, backend = adbackend()) -> Number
    _gradient(f, p::Array, backend = adbackend()) -> Array

Compute gradient of function `f` at `x` using AD backend `backend`.
"""
_gradient(f, p) = _gradient(f, p, Val(adbackend()))
_gradient(f, p, backend::Symbol) = _gradient(f, p, Val(adbackend(backend)))

"""
    _jacobian(f, p, backend = adbackend()) -> Array

Compute Jacobian matrix of function `f` at `p` using AD backend `backend`.
Inputs and outputs of `f` are vectorized.
"""
_jacobian(f, p) = _jacobian(f, p, Val(adbackend()))
_jacobian(f, p, backend::Symbol) = _jacobian(f, p, Val(adbackend(backend)))

# Finite differences

push!(_adbackends, :finitedifferences)
adbackend!(:finitedifferences)

_default_fdm() = central_fdm(5, 1)

_gradient(f, p, backend::Val{:finitedifferences}) = _gradient(f, p, _default_fdm())

function _gradient(f, p, fdm::FiniteDifferences.FiniteDifferenceMethod)
    return FiniteDifferences.grad(fdm, f, p)[1]
end

_jacobian(f, p, backend::Val{:finitedifferences}) = _jacobian(f, p, _default_fdm())

function _jacobian(f, p, fdm::FiniteDifferences.FiniteDifferenceMethod)
    return FiniteDifferences.jacobian(fdm, f, p)[1]
end
