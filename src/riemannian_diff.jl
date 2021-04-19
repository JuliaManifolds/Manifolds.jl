
"""
    AbstractRiemannianDiffBackend

An abstract type for diff backends. See [`RiemannianONBDiffBackend`](@ref) for
an example.
"""
abstract type AbstractRiemannianDiffBackend end

@doc raw"""
    differential(M::AbstractManifold, f, t::Real, backend::AbstractDiffBackend = rdifferential_backend())

Compute the Riemannian differential of a curve $f: ℝ\to M$ on a manifold `M`
represented by function `f` at time `t` using the given backend.
It is calculated as the tangent vector equal to $\mathrm{d}f_t(t)[1]$.
"""
differential(::AbstractManifold, ::Any, ::Real, ::AbstractRiemannianDiffBackend)

@doc raw"""
    gradient(M::AbstractManifold, f, p, backend::AbstractRiemannianDiffBackend = rgradient_backend())

Compute the Riemannian gradient $∇f(p)$ of a real field on manifold `M` represented by
function `f` at point `p` using the given backend.
"""
gradient(::AbstractManifold, ::Any, ::Any, ::AbstractRiemannianDiffBackend)

function differential!(
    M::AbstractManifold,
    f::Any,
    X,
    t,
    backend::AbstractRiemannianDiffBackend,
)
    return copyto!(X, differential(M, f, t, backend))
end

function gradient!(M::AbstractManifold, f, X, p, backend::AbstractRiemannianDiffBackend)
    return copyto!(X, gradient(M, f, p, backend))
end

differential(M::AbstractManifold, f, p) = differential(M, f, p, rdifferential_backend())

function differential!(M::AbstractManifold, f, X, p)
    return differential!(M, f, X, p, rdifferential_backend())
end

gradient(M::AbstractManifold, f, p) = gradient(M, f, p, rgradient_backend())

gradient!(M::AbstractManifold, f, X, p) = gradient!(M, f, X, p, rgradient_backend())

"""
    RiemannianONBDiffBackend(
        diff_backend::AbstractDiffBackend
        retraction::AbstractRetractionMethod
        inverse_retraction::AbstractInverseRetractionMethod
        basis::Union{AbstractOrthonormalBasis,CachedBasis{<:AbstractOrthonormalBasis}},
    ) <: AbstractRiemannianDiffBackend

Riemannian differentiation based on differentiation in an [`AbstractOrthonormalBasis`](@ref)
`basis` using specified `retraction`, `inverse_retraction` and using backend `diff_backend`.
"""
struct RiemannianONBDiffBackend{
    TADBackend<:AbstractDiffBackend,
    TRetr<:AbstractRetractionMethod,
    TInvRetr<:AbstractInverseRetractionMethod,
    TBasis<:Union{
        AbstractOrthonormalBasis,
        CachedBasis{𝔽,<:AbstractOrthonormalBasis{𝔽}} where {𝔽},
    },
} <: AbstractRiemannianDiffBackend
    diff_backend::TADBackend
    retraction::TRetr
    inverse_retraction::TInvRetr
    basis::TBasis
end

function differential(M::AbstractManifold, f, t::Real, backend::RiemannianONBDiffBackend)
    p = f(t)
    onb_coords = _derivative(zero(number_eltype(p)), backend.diff_backend) do h
        return get_coordinates(
            M,
            p,
            inverse_retract(M, p, f(t + h), backend.inverse_retraction),
            backend.basis,
        )
    end
    return get_vector(M, p, onb_coords, backend.basis)
end

function differential!(
    M::AbstractManifold,
    f,
    X,
    t::Real,
    backend::RiemannianONBDiffBackend,
)
    p = f(t)
    onb_coords = _derivative(zero(number_eltype(p)), backend.diff_backend) do h
        return get_coordinates(
            M,
            p,
            inverse_retract(M, p, f(t + h), backend.inverse_retraction),
            backend.basis,
        )
    end
    return get_vector!(M, X, p, onb_coords, backend.basis)
end

function gradient(M::AbstractManifold, f, p, backend::RiemannianONBDiffBackend)
    X = get_coordinates(M, p, zero_vector(M, p), backend.basis)
    onb_coords = _gradient(X, backend.diff_backend) do Y
        return f(retract(M, p, get_vector(M, p, Y, backend.basis), backend.retraction))
    end
    return get_vector(M, p, onb_coords, backend.basis)
end

function gradient!(M::AbstractManifold, f, X, p, backend::RiemannianONBDiffBackend)
    X2 = get_coordinates(M, p, zero_vector(M, p), backend.basis)
    onb_coords = _gradient(X2, backend.diff_backend) do Y
        return f(retract(M, p, get_vector(M, p, Y, backend.basis), backend.retraction))
    end
    return get_vector!(M, X, p, onb_coords, backend.basis)
end

"""
    CurrentRiemannianDiffBackend(backend::AbstractRiemannianDiffBackend)

A mutable struct for storing the current Riemannian differentiation backend in global
constants [`Manifolds._current_rgradient_backend`](@ref) and
[`Manifolds._current_rdifferential_backend`](@ref).

# See also

[`AbstractRiemannianDiffBackend`](@ref), [`rdifferential_backend`](@ref),
[`rdifferential_backend!`](@ref)
"""
mutable struct CurrentRiemannianDiffBackend
    backend::AbstractRiemannianDiffBackend
end

"""
    _current_rgradient_backend

The instance of [`Manifolds.CurrentRiemannianDiffBackend`](@ref) that stores the
globally default differentiation backend for calculating gradients.

# See also

[`Manifolds.gradient(::AbstractManifold, ::Any, ::Any, ::AbstractRiemannianDiffBackend)`](@ref)
"""
const _current_rgradient_backend = CurrentRiemannianDiffBackend(
    RiemannianONBDiffBackend(
        diff_backend(),
        ExponentialRetraction(),
        LogarithmicInverseRetraction(),
        DefaultOrthonormalBasis(),
    ),
)

"""
    _current_rdifferential_backend

The instance of [`Manifolds.CurrentRiemannianDiffBackend`](@ref) that stores the
globally default differentiation backend for calculating differentials.

# See also

[`Manifolds.differential`](@ref)
"""
const _current_rdifferential_backend = CurrentRiemannianDiffBackend(
    RiemannianONBDiffBackend(
        diff_backend(),
        ExponentialRetraction(),
        LogarithmicInverseRetraction(),
        DefaultOrthonormalBasis(),
    ),
)

"""
    rgradient_backend() -> AbstractRiemannianDiffBackend

Get the current differentiation backend for Riemannian gradients.
"""
rgradient_backend() = _current_rgradient_backend.backend

"""
    rgradient_backend!(backend::AbstractRiemannianDiffBackend)

Set current Riemannian gradient backend for differentiation to `backend`.
"""
function rgradient_backend!(backend::AbstractRiemannianDiffBackend)
    _current_rgradient_backend.backend = backend
    return backend
end

"""
    rdifferential_backend() -> AbstractRiemannianDiffBackend

Get the current differentiation backend for Riemannian differentials.
"""
rdifferential_backend() = _current_rdifferential_backend.backend

"""
    rdifferential_backend!(backend::AbstractRiemannianDiffBackend)

Set current Riemannian differential backend for differentiation to `backend`.
"""
function rdifferential_backend!(backend::AbstractRiemannianDiffBackend)
    _current_rdifferential_backend.backend = backend
    return backend
end

"""
    RiemannianProjectionGradientBackend(
        diff_backend::AbstractDiffBackend
    ) <: AbstractRiemannianDiffBackend

Riemannian differentiation based on differentiation in the ambient space and projection to
the given manifold. Differentiation in the ambient space is performed using
the backend `diff_backend`.

Only valid for manifolds that are embedded in a special way in the Euclidean space.
See [^Absil2008], Section 3.6.1 for details.

[^Absil2008]:
    > Absil, P. A., et al. Optimization Algorithms on Matrix Manifolds. 2008.
"""
struct RiemannianProjectionGradientBackend{TADBackend<:AbstractDiffBackend} <:
       AbstractRiemannianDiffBackend
    diff_backend::TADBackend
end

function gradient(M::AbstractManifold, f, p, backend::RiemannianProjectionGradientBackend)
    amb_grad = _gradient(f, p, backend.diff_backend)
    return project(M, p, amb_grad)
end

function gradient!(
    M::AbstractManifold,
    f,
    X,
    p,
    backend::RiemannianProjectionGradientBackend,
)
    amb_grad = embed(M, p, X)
    _gradient!(f, amb_grad, p, backend.diff_backend)
    return project!(M, X, p, amb_grad)
end
