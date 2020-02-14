#
# General functions for decorators
#

"""
    AbstractDecoratorManifold <: Manifold

An `AbstractDecoratorManifold` indicates that to some extend a manifold subtype
decorates another manifold in the sense that

* it extends the functionality of a manifold with further features
* it defines a new manifold that internally uses functions from another manifold

with the main intend that several or most functions of [`Manifold`](@ref) are transparently
passed thrugh to the manifold that is decorated.
"""
abstract type AbstractDecoratorManifold <: Manifold end

"""
    is_default_decorator(M)

For any manifold that is a subtype of [`AbstractDecoratorManifold`](@ref), this function
indicates whether a certain manifold `M` acts as a default decorator.

This yields that _all_ functions are passed through to the [`Manifold`](@ref) that is
decorated independently of [`is_decorator_transparent`](@ref) for single functions.

The idea is that a set of decorators has a default decorator that is already covered
by the orginial implementation.
"""
is_default_decorator(M::Manifold) = _extract_val(default_decorator_dispatch(M))

default_decorator_dispatch(M::Manifold) = Val(false)

"""
    is_decorator_transparent(f, M, args...)

Given a [`Manifold`](@ref) `M` and a function `f`, indicate, whether a
[`AbstractDecoratorManifold`](@ref) acts transparent for `f`. This means, it
just passes through down to the internally stored manifold.
Only decorator manifolds can be transparent and their default is, to be transparent.
A function that is affected by the decorator hence has to set this to `Val(false)`
actively.
If a decorator manifold is not in general transparent, it might still pass down
for the case that a decorator is the default decorator, see [`is_default_decorator`](@ref).
"""
function is_decorator_transparent(f, M::Manifold, args...)
    return _extract_val(decorator_transparent_dispatch(f, M, args...))
end

decorator_transparent_dispatch(f, M::Manifold, args...) = Val(:transparent)

function _acts_transparently(f, M::Manifold, args...)
    return _val_or(default_decorator_dispatch(M), decorator_transparent_dispatch(f, M, args...))
end

_val_or(::Val{true}, ::Val{T}) where {T} = Val(:transparent)
_val_or(::Val{false}, ::Val{T}) where {T} = Val(T)

#
# Functions overwritten with decorators
#

function base_manifold(M::AbstractDecoratorManifold, depth::Val{N} = Val(-1)) where {N}
    return (N != 0) ? base_manifold(M.manifold, (N > 0) ? Val(N-1) : depth) : M
end

@decorator_transparent_function check_manifold_point(
    M::AbstractDecoratorManifold,
    p;
    kwargs...,
)

@decorator_transparent_function check_tangent_vector(
    M::AbstractDecoratorManifold,
    p,
    X;
    kwargs...,
)

@decorator_transparent_function distance(
    M::AbstractDecoratorManifold,
    p,
    q,
)

@decorator_transparent_function exp!(
    M::AbstractDecoratorManifold,
    q,
    p,
    X,
)


@decorator_transparent_function exp!(
    M::AbstractDecoratorManifold,
    q,
    p,
    X,
    T
)

@decorator_transparent_function flat!(
    M::AbstractDecoratorManifold,
    ξ::CoTFVector,
    p,
    X::TFVector,
)

@decorator_transparent_function get_basis(
    M::AbstractDecoratorManifold,
    p,
    B::AbstractBasis,
)

@decorator_transparent_function get_basis(
    M::AbstractDecoratorManifold,
    p,
    B::AbstractOrthonormalBasis,
)
@decorator_transparent_function get_basis(
    M::AbstractDecoratorManifold,
    p,
    B::ProjectedOrthonormalBasis{:svd,ℝ},
)
@decorator_transparent_function get_basis(
    M::AbstractDecoratorManifold,
    p,
    B::AbstractPrecomputedOrthonormalBasis,
)
@decorator_transparent_function get_basis(
    M::AbstractDecoratorManifold,
    p,
    B::ArbitraryOrthonormalBasis,
)

@decorator_transparent_function get_coordinates(
    M::AbstractDecoratorManifold,
    p,
    X,
    B::AbstractBasis,
)
@decorator_transparent_function get_coordinates(
    M::AbstractDecoratorManifold,
    p,
    X,
    B::AbstractPrecomputedOrthonormalBasis,
)
@decorator_transparent_function get_coordinates(
    M::AbstractDecoratorManifold,
    p,
    X,
    B::AbstractPrecomputedOrthonormalBasis{ℝ},
)

@decorator_transparent_function get_vector(
    M::AbstractDecoratorManifold,
    p,
    X,
    B::AbstractBasis,
)
@decorator_transparent_function get_vector(
    M::AbstractDecoratorManifold,
    p,
    X,
    B::AbstractPrecomputedOrthonormalBasis,
)

@decorator_transparent_function hat!(M::AbstractDecoratorManifold, X, p, Xⁱ)

@decorator_transparent_function injectivity_radius(M::AbstractDecoratorManifold)
@decorator_transparent_function injectivity_radius(M::AbstractDecoratorManifold, p)

@decorator_transparent_function inner(M::AbstractDecoratorManifold, p, X, Y)

@decorator_transparent_function inverse_retract!(
    M::AbstractDecoratorManifold,
    X,
    p,
    q,
    m::AbstractInverseRetractionMethod,
)

@decorator_transparent_function inverse_retract!(
    M::AbstractDecoratorManifold,
    X,
    p,
    q,
    m::LogarithmicInverseRetraction,
)

@decorator_transparent_function isapprox(M::AbstractDecoratorManifold, p, q; kwargs...)
@decorator_transparent_function isapprox(M::AbstractDecoratorManifold, p, X, Y; kwargs...)

@decorator_transparent_function log!(M::AbstractDecoratorManifold, X, p, q)

@decorator_transparent_function manifold_dimension(M::AbstractDecoratorManifold)

@decorator_transparent_function mean!(
    M::AbstractDecoratorManifold,
    x::AbstractVector,
    w::AbstractVector;
    kwargs...,
)

@decorator_transparent_function median!(
    M::AbstractDecoratorManifold,
    x::AbstractVector,
    w::AbstractVector;
    kwargs...,
)

@decorator_transparent_function normal_tvector_distribution(
    M::AbstractDecoratorManifold, p, σ
)

@decorator_transparent_function project_point!(M::AbstractDecoratorManifold, p)

@decorator_transparent_function project_tangent!(M::AbstractDecoratorManifold, Y, p, X)

@decorator_transparent_function projected_distribution(M::AbstractDecoratorManifold, d, p)

@decorator_transparent_function projected_distribution(M::AbstractDecoratorManifold, d)

@decorator_transparent_function representation_size(M::AbstractDecoratorManifold)

@decorator_transparent_function retract!(
    M::AbstractDecoratorManifold,
    q,
    p,
    X,
    m::AbstractRetractionMethod
)

@decorator_transparent_function retract!(
    M::AbstractDecoratorManifold,
    q,
    p,
    X,
    m::ExponentialRetraction
)

@decorator_transparent_function sharp!(
    M::AbstractDecoratorManifold,
    X::TFVector,
    p,
    ξ::CoTFVector,
)

@decorator_transparent_function vector_transport_along!(
    M::AbstractDecoratorManifold,
    Y,
    p,
    X,
    c,
)

@decorator_transparent_function vector_transport_direction!(
    M::AbstractDecoratorManifold,
    Y,
    p,
    X,
    d,
)

@decorator_transparent_function vector_transport_to!(
    M::AbstractDecoratorManifold,
    Y,
    p,
    X,
    q,
    m::AbstractVectorTransportMethod,
)

@decorator_transparent_function vee!(M::AbstractDecoratorManifold, Xⁱ, p, X)

@decorator_transparent_function zero_tangent_vector!(M::AbstractDecoratorManifold, X, p)
