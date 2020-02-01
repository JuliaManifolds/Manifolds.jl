
abstract type AbstractDiffBackend end

struct NoneDiffBackend <: AbstractDiffBackend end

function _derivative(f, x, backend::AbstractDiffBackend)
    error("_derivative not implemented for curve $(typeof(f)), point $(typeof(x)) and " *
          "backend $(typeof(backend))")
end

function _gradient(f, x, backend::AbstractDiffBackend)
    error("_gradient not implemented for field $(typeof(f)), point $(typeof(x)) and " *
          "backend $(typeof(backend))")
end

function _jacobian(f, x, backend::AbstractDiffBackend)
    error("_jacobian not implemented for map $(typeof(f)), point $(typeof(x)) and " *
          "backend $(typeof(backend))")
end

"""
    CurrentDiffBackend(backend::AbstractDiffBackend)

A mutable struct for storing the current differentiation backend in a global
constant [`_current_diff_backend`](@ref).

# See also

[`AbstractDiffBackend`](@ref), [`diff_backend`](@ref), [`diff_backend!`]
"""
mutable struct CurrentDiffBackend
    backend::AbstractDiffBackend
end

"""
    _current_diff_backend

The instance of [`CurrentDiffBackend`] that stores the globally default differentiation
backend.
"""
const _current_diff_backend = CurrentDiffBackend(NoneDiffBackend())

"""
    _diff_backends

A vector of valid [`AbstractDiffBackend`](@ref).
"""
const _diff_backends = AbstractDiffBackend[]

"""
    diff_backend() -> AbstractDiffBackend

Get the current differentiation backend.
"""
diff_backend() = _current_diff_backend.backend

"""
    diff_backend!(backend::AbstractDiffBackend)

Set current backend for differentiation to `backend`.
"""
function diff_backend!(backend::AbstractDiffBackend)
    _current_diff_backend.backend = backend
    return backend
end

"""
    diff_backends() -> Vector{AbstractDiffBackend}

Get vector of currently valid differentiation backends.
"""
diff_backends() = _diff_backends

_derivative(f, x) = _derivative(f, x, diff_backend())

_gradient(f, x) = _gradient(f, x, diff_backend())

_jacobian(f, x) = _jacobian(f, x, diff_backend())

# Finite differences

"""
    FiniteDifferencesBackend(method::FiniteDifferenceMethod = central_fdm(5, 1))

Differentiation backend based on the FiniteDifferences package.
"""
struct FiniteDifferencesBackend{TM<:FiniteDifferenceMethod} <: AbstractDiffBackend
    method::TM
end

function FiniteDifferencesBackend()
    return FiniteDifferencesBackend(central_fdm(5, 1))
end

push!(_diff_backends, FiniteDifferencesBackend())

diff_backend!(_diff_backends[end])

function _derivative(f, x, backend::FiniteDifferencesBackend)
    return FiniteDifferences.grad(backend.method, f, x)[1]
end

function _gradient(f, x, backend::FiniteDifferencesBackend)
    return FiniteDifferences.grad(backend.method, f, x)[1]
end

function _jacobian(f, x, backend::FiniteDifferencesBackend)
    return FiniteDifferences.jacobian(backend.method, f, x)[1]
end
