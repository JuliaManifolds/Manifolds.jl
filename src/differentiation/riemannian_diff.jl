
"""
    AbstractRiemannianDiffBackend

An abstract type for backends for differentiation.
"""
abstract type AbstractRiemannianDiffBackend end

@doc raw"""
    differential(M::AbstractManifold, f, t::Real, backend::AbstractDiffBackend)
    differential!(M::AbstractManifold, f, X, t::Real, backend::AbstractDiffBackend)

Compute the Riemannian differential of a curve $f: ℝ\to M$ on a manifold `M`
represented by function `f` at time `t` using the given backend.
It is calculated as the tangent vector equal to $\mathrm{d}f_t(t)[1]$.

The mutating variant computes the differential in place of `X`.
"""
differential(::AbstractManifold, ::Any, ::Real, ::AbstractRiemannianDiffBackend)

@doc raw"""
    gradient(M::AbstractManifold, f, p, backend::AbstractRiemannianDiffBackend)
    gradient!(M::AbstractManifold, f, X, p, backend::AbstractRiemannianDiffBackend)

Compute the Riemannian gradient ``∇f(p)`` of a real-valued function ``f:\mathcal M \to ℝ``
at point `p` on the manifold `M` using the specified [`AbstractRiemannianDiffBackend`](@ref).

The mutating variant computes the gradient in place of `X`.
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

@doc raw"""
    TangentDiffBackend <: AbstractRiemannianDiffBackend

A backend that uses a tangent space and a basis therein to derive an
intrinsic differentiation scheme.

Since it works in a tangent space, methods might require a retraction and an
inverse retraction as well as a basis.

In the tangent space itself, this backend then employs an (Euclidean)
[`AbstractDiffBackend`](@ref)

# Constructor

    TangentDiffBackend(diff_backend)

where `diff_backend` is an [`AbstractDiffBackend`](@ref) to be used on the tangent space.

With the keyword arguments

* `retraction` an [`AbstractRetractionMethod`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions.html#ManifoldsBase.AbstractRetractionMethod) ([`ExponentialRetraction`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions.html#ManifoldsBase.ExponentialRetraction) by default)
* `inverse_retraction` an [`AbstractInverseRetractionMethod`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions.html#ManifoldsBase.AbstractInverseRetractionMethod) ([`LogarithmicInverseRetraction`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions.html#ManifoldsBase.LogarithmicInverseRetraction) by default)
* `basis` an [`AbstractBasis`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/bases.html#ManifoldsBase.AbstractBasis) ([`DefaultOrthogonalBasis`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/bases.html#ManifoldsBase.DefaultOrthogonalBasis) by default)
"""
struct TangentDiffBackend{
    TAD<:AbstractDiffBackend,
    TR<:AbstractRetractionMethod,
    TIR<:AbstractInverseRetractionMethod,
    TB<:AbstractBasis,
} <: AbstractRiemannianDiffBackend
    diff_backend::TAD
    retraction::TR
    inverse_retraction::TIR
    basis::TB
end
function TangentDiffBackend(
    diff_backend::TAD;
    retraction::TR=ExponentialRetraction(),
    inverse_retraction::TIR=LogarithmicInverseRetraction(),
    basis::TB=DefaultOrthonormalBasis(),
) where {
    TAD<:AbstractDiffBackend,
    TR<:AbstractRetractionMethod,
    TIR<:AbstractInverseRetractionMethod,
    TB<:AbstractBasis,
}
    return TangentDiffBackend{TAD,TR,TIR,TB}(
        diff_backend,
        retraction,
        inverse_retraction,
        basis,
    )
end

function differential(M::AbstractManifold, f, t::Real, backend::TangentDiffBackend)
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

function differential!(M::AbstractManifold, f, X, t::Real, backend::TangentDiffBackend)
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

@doc raw"""
    gradient(M, f, p, backend::TangentDiffBackend)

This method uses the internal `backend.diff_backend` (Euclidean) on the function

```math
    f(\retr_p(\cdot))
```

which is given on the tangent space. In detail, the gradient can be written in
terms of the `backend.basis`. We illustrate it here for an [`AbstractOrthonormalBasis`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/bases.html#ManifoldsBase.AbstractOrthonormalBasis),
since that simplifies notations:

```math
\operatorname{grad}f(p) = \operatorname{grad}f(p) = \sum_{i=1}^{d} g_p(\operatorname{grad}f(p),X_i)X_i
	= \sum_{i=1}^{d} Df(p)[X_i]X_i
```

where the last equality is due to the definition of the gradient as the Riesz representer of the differential.

If the backend is a forward (or backward) finite difference, these coefficients in the sum can be approximates as

```math
DF(p)[Y] ≈ \frac{1}{h}\bigl( f(\exp_p(hY)) - f(p) \bigr)
```
writing ``p=\exp_p(0)`` we see that this is a finite difference of ``f\circ\exp_p``, i.e. for
a function on the tangent space, so we can also use other (Euclidean) backends
"""
function gradient(M::AbstractManifold, f, p, backend::TangentDiffBackend)
    X = get_coordinates(M, p, zero_vector(M, p), backend.basis)
    onb_coords = _gradient(X, backend.diff_backend) do Y
        return f(retract(M, p, get_vector(M, p, Y, backend.basis), backend.retraction))
    end
    return get_vector(M, p, onb_coords, backend.basis)
end

function gradient!(M::AbstractManifold, f, X, p, backend::TangentDiffBackend)
    X2 = get_coordinates(M, p, zero_vector(M, p), backend.basis)
    onb_coords = _gradient(X2, backend.diff_backend) do Y
        return f(retract(M, p, get_vector(M, p, Y, backend.basis), backend.retraction))
    end
    return get_vector!(M, X, p, onb_coords, backend.basis)
end

@doc raw"""
    RiemannianProjectionBackend <: AbstractRiemannianDiffBackend

This backend computes the differentiation in the embedding, which is currently limited
to the gradient. Let ``mathcal M`` denote a manifold embedded in some ``R^m``, where ``m``
is usually (much) larger than the manifold dimension.
Then we require three tools

* A function ``f̃: ℝ^m → ℝ`` such that its restriction to the manifold yields the cost
  function ``f`` of interest.
* A [`project`](@ref) function to project tangent vectors from the embedding (at ``T_pℝ^m``)
  back onto the tangent space ``T_p\mathcal M``. This also includes possible changes
  of the representation of the tangent vector (e.g. in the Lie algebra or in a different data format).
* A [`change_representer`](@ref) for non-isometrically embedded manifolds,
  i.e. where the tangent space ``T_p\mathcal M`` of the manifold does not inherit
  the inner product from restriction of the inner product from the tangent space ``T_pℝ^m``
  of the embedding

For more details see [^AbsilMahonySepulchre2008], Section 3.6.1 for a derivation on submanifolds.

[^AbsilMahonySepulchre2008]:
    > Absil, P.-A., Mahony, R. and Sepulchre R.,
    > _Optimization Algorithms on Matrix Manifolds_
    > Princeton University Press, 2008,
    > doi: [10.1515/9781400830244](https://doi.org/10.1515/9781400830244)
    > [open access](http://press.princeton.edu/chapters/absil/)
"""
struct RiemannianProjectionBackend{TADBackend<:AbstractDiffBackend} <:
       AbstractRiemannianDiffBackend
    diff_backend::TADBackend
end

function gradient(M::AbstractManifold, f, p, backend::RiemannianProjectionBackend)
    amb_grad = _gradient(f, p, backend.diff_backend)
    return change_representer(M, EuclideanMetric(), p, project(M, p, amb_grad))
end

function gradient!(M::AbstractManifold, f, X, p, backend::RiemannianProjectionBackend)
    amb_grad = embed(M, p, X)
    _gradient!(f, amb_grad, p, backend.diff_backend)
    project!(M, X, p, amb_grad)
    return change_representer!(M, X, EuclideanMetric(), p, X)
end
