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
    T in _adbackends ||
    throw(ArgumentError("Invalid AD backend $(T). Valid options are $(adbackends())."))
    return T
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
    _gradient(f, x::Number, backend = adbackend()) -> Number
    _gradient(f, x::Array, backend = adbackend()) -> Array

Compute gradient of function `f` at point `x` using AD backend `backend`.
"""
_gradient(f, x) = _gradient(f, x, Val(adbackend()))
_gradient(f, x, backend::Symbol) = _gradient(f, x, Val(adbackend(backend)))

"""
    _jacobian(f, x::Array, backend = adbackend()) -> Array

Compute Jacobian matrix of function `f` at point `x` using AD backend `backend`.
Inputs and outputs of `f` are vectorized.
"""
_jacobian(f, x) = _jacobian(f, x, Val(adbackend()))
_jacobian(f, x, backend::Symbol) = _jacobian(f, x, Val(adbackend(backend)))

# Finite differences

push!(_adbackends, :finitedifferences)
adbackend!(:finitedifferences)

_default_fdm() = central_fdm(5, 1)

_gradient(f, x, backend::Val{:finitedifferences}) = _gradient(f, x, _default_fdm())

function _gradient(f, x, fdm::FiniteDifferences.FiniteDifferenceMethod)
    return FiniteDifferences.grad(fdm, f, x)[1]
end

_jacobian(f, x, backend::Val{:finitedifferences}) = _jacobian(f, x, _default_fdm())

function _jacobian(f, x, fdm::FiniteDifferences.FiniteDifferenceMethod)
    return FiniteDifferences.jacobian(fdm, f, x)[1]
end
