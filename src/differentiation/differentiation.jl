
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

@doc raw"""
    ODEExponentialRetraction{T<:AbstractRetractionMethod, B<:AbstractBasis} <: AbstractRetractionMethod

Approximate the exponential map on the manifold by evaluating the ODE descripting the geodesic at 1,
assuming the default connection of the given manifold by solving the ordinary differential
equation

```math
\frac{d^2}{dt^2} p^k + Γ^k_{ij} \frac{d}{dt} p_i \frac{d}{dt} p_j = 0,
```

where ``Γ^k_{ij}`` are the Christoffel symbols of the second kind, and
the Einstein summation convention is assumed.

See [`solve_exp_ode`](@ref) for further details.

# Constructor

    ODEExponentialRetraction(
        r::AbstractRetractionMethod,
        b::AbstractBasis=DefaultOrthogonalBasis(),
    )

Generate the retraction with a retraction to use internally (for some approaches)
and a basis for the tangent space(s).
"""
struct ODEExponentialRetraction{T<:AbstractRetractionMethod,B<:AbstractBasis} <:
       AbstractRetractionMethod
    retraction::T
    basis::B
end
function ODEExponentialRetraction(r::T) where {T<:AbstractRetractionMethod}
    return ODEExponentialRetraction(r, DefaultOrthonormalBasis())
end
function ODEExponentialRetraction(::T, b::CachedBasis) where {T<:AbstractRetractionMethod}
    return throw(
        DomainError(
            b,
            "Cached Bases are currently not supported, since the basis has to be implemented in a surrounding of the start point as well.",
        ),
    )
end
function ODEExponentialRetraction(r::ExponentialRetraction, ::AbstractBasis)
    return throw(
        DomainError(
            r,
            "You can not use the exponential map as an inner method to solve the ode for the exponential map.",
        ),
    )
end
function ODEExponentialRetraction(r::ExponentialRetraction, ::CachedBasis)
    return throw(
        DomainError(
            r,
            "Neither the exponential map nor a Cached Basis can be used with this retraction type.",
        ),
    )
end
