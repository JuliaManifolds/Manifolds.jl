
abstract type AbstractDiffBackend end

struct NoneDiffBackend <: AbstractDiffBackend end

function _derivative(f, x, ::NoneDiffBackend)
    error("NoneDiffBackend does not provide _derivative")
end

function _gradient(f, x, ::NoneDiffBackend)
    error("NoneDiffBackend does not provide _gradient")
end

function _jacobian(f, x, ::NoneDiffBackend)
    error("NoneDiffBackend does not provide _jacobian")
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
    adbackend!(backend::AbstractDiffBackend)

Set current backend for autodiff to `backend`.
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

"""
    _derivative(f::AbstractCurve, x, backend = diff_backend())

Compute derivative of the curve `f` at point `x` using differentiation backend `backend`.
"""
_derivative(f::AbstractCurve, x) = _derivative(f, x, diff_backend())

"""
    _gradient(f::AbstractRealField, x, backend = diff_backend())

Compute gradient of function `f` at point `x` using differentiation backend `backend`.
"""
_gradient(f::AbstractRealField, x) = _gradient(f, x, diff_backend())

"""
    _jacobian(f::AbstractMap, x::AbstractArray, backend = diff_backend()) -> AbstractArray

Compute Jacobian matrix of function `f` at point `x` using differentiation backend `backend`.
Inputs and outputs of `f` are vectorized.
"""
_jacobian(f::AbstractMap, x) = _jacobian(f::AbstractMap, x, diff_backend())

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

function _derivative(f::AbstractCurve, x, backend::FiniteDifferencesBackend)
    return FiniteDifferences.grad(backend.method, f, x)[1]
end

function _gradient(f::AbstractRealField, x, backend::FiniteDifferencesBackend)
    return FiniteDifferences.grad(backend.method, f, x)[1]
end

function _jacobian(f::AbstractMap, x, backend::FiniteDifferencesBackend)
    return FiniteDifferences.jacobian(backend.method, f, x)[1]
end
