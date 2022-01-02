
"""
    AbstractDiffBackend

An abstract type for diff backends. See [`FiniteDifferencesBackend`](@ref) for
an example.
"""
abstract type AbstractDiffBackend end

struct NoneDiffBackend <: AbstractDiffBackend end

"""
    _derivative(f, t[, backend::AbstractDiffBackend])

Compute the derivative of a callable `f` at time `t` computed using the given `backend`,
an object of type [`Manifolds.AbstractDiffBackend`](@ref). If the backend is not explicitly
specified, it is obtained using the function [`default_differential_backend`](@ref).

This function calculates plain Euclidean derivatives, for Riemannian differentiation see
for example [`differential`](@ref Manifolds.differential(::AbstractManifold, ::Any, ::Real, ::AbstractRiemannianDiffBackend)).

!!! note

    Not specifying the backend explicitly will usually result in a type instability
    and decreased performance.
"""
function _derivative end

_derivative(f, t) = _derivative(f, t, default_differential_backend())

function _derivative!(f, X, t, backend::AbstractDiffBackend=default_differential_backend())
    return copyto!(X, _derivative(f, t, backend))
end

"""
    _gradient(f, p[, backend::AbstractDiffBackend])

Compute the gradient of a callable `f` at point `p` computed using the given `backend`,
an object of type [`AbstractDiffBackend`](@ref). If the backend is not explicitly
specified, it is obtained using the function [`default_differential_backend`](@ref).

This function calculates plain Euclidean gradients, for Riemannian gradient calculation see
for example [`gradient`](@ref Manifolds.gradient(::AbstractManifold, ::Any, ::Any, ::AbstractRiemannianDiffBackend)).

!!! note

    Not specifying the backend explicitly will usually result in a type instability
    and decreased performance.
"""
function _gradient end

_gradient(f, p) = _gradient(f, p, default_differential_backend())

function _gradient!(f, X, p, backend::AbstractDiffBackend=default_differential_backend())
    return copyto!(X, _gradient(f, p, backend))
end

"""
    _jacobian(f, p[, backend::AbstractDiffBackend])

Compute the jacobian of a callable `f` at point `p` computed using the given `backend`,
an object of type [`AbstractDiffBackend`](@ref). If the backend is not explicitly
specified, it is obtained using the function [`default_differential_backend`](@ref).

This function calculates plain Euclidean gradients, for Riemannian gradient calculation see
for example [`gradient`](@ref Manifolds.gradient(::AbstractManifold, ::Any, ::Any, ::AbstractRiemannianDiffBackend)).

!!! note

    Not specifying the backend explicitly will usually result in a type instability
    and decreased performance.
"""
function _jacobian end

_jacobian(f, p) = _jacobian(f, p, default_differential_backend())

function _jacobian!(f, X, p, backend::AbstractDiffBackend=default_differential_backend())
    return copyto!(X, _jacobian(f, p, backend))
end

"""
    CurrentDiffBackend(backend::AbstractDiffBackend)

A mutable struct for storing the current differentiation backend in a global
constant [`_current_default_differential_backend`](@ref).

# See also

[`AbstractDiffBackend`](@ref), [`default_differential_backend`](@ref), [`set_default_differential_backend!`](@ref)
"""
mutable struct CurrentDiffBackend
    backend::AbstractDiffBackend
end

"""
    _current_default_differential_backend

The instance of [`Manifolds.CurrentDiffBackend`](@ref) that stores the globally default
differentiation backend.
"""
const _current_default_differential_backend = CurrentDiffBackend(NoneDiffBackend())
"""
    default_differential_backend() -> AbstractDiffBackend

Get the default differentiation backend.
"""
default_differential_backend() = _current_default_differential_backend.backend

"""
    set_default_differential_backend!(backend::AbstractDiffBackend)

Set current backend for differentiation to `backend`.
"""
function set_default_differential_backend!(backend::AbstractDiffBackend)
    _current_default_differential_backend.backend = backend
    return backend
end