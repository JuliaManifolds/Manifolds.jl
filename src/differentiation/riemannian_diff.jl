
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

A backend that uses tangent spaces and bases therein to derive an
intrinsic differentiation scheme.

Since it works in tangent spaces at argument and function value, methods might require a
retraction and an inverse retraction as well as a basis.

In the tangent space itself, this backend then employs an (Euclidean)
[`AbstractDiffBackend`](@ref)

# Constructor

    TangentDiffBackend(diff_backend)

where `diff_backend` is an [`AbstractDiffBackend`](@ref) to be used on the tangent space.

With the keyword arguments

* `retraction` an [AbstractRetractionMethod](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions.html) (`ExponentialRetraction` by default)
* `inverse_retraction` an [AbstractInverseRetractionMethod](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions.html) `LogarithmicInverseRetraction` by default)
* `basis_arg` an [AbstractBasis](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/bases.html) (`DefaultOrthogonalBasis` by default)
* `basis_val` an [AbstractBasis](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/bases.html) (`DefaultOrthogonalBasis` by default)
"""
struct TangentDiffBackend{
    TAD<:AbstractDiffBackend,
    TR<:AbstractRetractionMethod,
    TIR<:AbstractInverseRetractionMethod,
    TBarg<:AbstractBasis,
    TBval<:AbstractBasis,
} <: AbstractRiemannianDiffBackend
    diff_backend::TAD
    retraction::TR
    inverse_retraction::TIR
    basis_arg::TBarg
    basis_val::TBval
end
function TangentDiffBackend(
    diff_backend::TAD;
    retraction::TR=ExponentialRetraction(),
    inverse_retraction::TIR=LogarithmicInverseRetraction(),
    basis_arg::TBarg=DefaultOrthonormalBasis(),
    basis_val::TBval=DefaultOrthonormalBasis(),
) where {
    TAD<:AbstractDiffBackend,
    TR<:AbstractRetractionMethod,
    TIR<:AbstractInverseRetractionMethod,
    TBarg<:AbstractBasis,
    TBval<:AbstractBasis,
}
    return TangentDiffBackend{TAD,TR,TIR,TBarg,TBval}(
        diff_backend,
        retraction,
        inverse_retraction,
        basis_arg,
        basis_val,
    )
end

function differential(M::AbstractManifold, f, t::Real, backend::TangentDiffBackend)
    p = f(t)
    onb_coords = Manifolds._derivative(zero(number_eltype(p)), backend.diff_backend) do h
        return get_coordinates(
            M,
            p,
            inverse_retract(M, p, f(t + h), backend.inverse_retraction),
            backend.basis_val,
        )
    end
    return get_vector(M, p, onb_coords, backend.basis_val)
end

function differential!(M::AbstractManifold, f, X, t::Real, backend::TangentDiffBackend)
    p = f(t)
    onb_coords = Manifolds._derivative(zero(number_eltype(p)), backend.diff_backend) do h
        return get_coordinates(
            M,
            p,
            inverse_retract(M, p, f(t + h), backend.inverse_retraction),
            backend.basis_val,
        )
    end
    return get_vector!(M, X, p, onb_coords, backend.basis_val)
end

@doc raw"""
    gradient(M, f, p, backend::TangentDiffBackend)

This method uses the internal `backend.diff_backend` (Euclidean) on the function

```math
    f(\operatorname{retr}_p(\cdot))
```

which is given on the tangent space. In detail, the gradient can be written in
terms of the `backend.basis_arg`. We illustrate it here for an [AbstractOrthonormalBasis](https://juliamanifolds.github.io/Manifolds.jl/stable/interface.html#ManifoldsBase.AbstractOrthonormalBasis),
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
    X = get_coordinates(M, p, zero_vector(M, p), backend.basis_arg)
    onb_coords = Manifolds._gradient(X, backend.diff_backend) do Y
        return f(retract(M, p, get_vector(M, p, Y, backend.basis_arg), backend.retraction))
    end
    return get_vector(M, p, onb_coords, backend.basis_arg)
end

function gradient!(M::AbstractManifold, f, X, p, backend::TangentDiffBackend)
    X2 = get_coordinates(M, p, zero_vector(M, p), backend.basis_arg)
    onb_coords = Manifolds._gradient(X2, backend.diff_backend) do Y
        return f(retract(M, p, get_vector(M, p, Y, backend.basis_arg), backend.retraction))
    end
    return get_vector!(M, X, p, onb_coords, backend.basis_arg)
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

see also [`riemannian_gradient`](@ref) and [^AbsilMahonySepulchre2008], Section 3.6.1 for a derivation on submanifolds.

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

@doc raw"""
    riemannian_gradient(M, p, Y; embedding_metric=EuclideanMetric())
    riemannian_gradient!(M, X, p, Y; embedding_metric=EuclideanMetric())

For a given gradient ``Y = \operatorname{grad} \tilde f(p)`` in the embedding of a manifold,
this function computes the Riemannian gradient ``\operatorname{grad} f(p)`` of the function
``\tilde f`` restricted to the manifold ``M``.
This can also be done in place of `X`.

By default it uses the following computation:
Let the projection ``Z = \operatorname{proj}_{T_p\mathcal M}(Y)`` of ``Y`` onto the
tangent space at ``p`` be given, that is with respect to the inner product in the embedding.
Then

```math
⟨Z-Y, W⟩ = 0 \text{ for all } W \in T_p\mathal M,
```

or rearranged ``⟨Y,W⟩ = ⟨Z,W⟩``. We then use the definition of the Riemannian gradient

```math
⟨\operatorname{grad} f(p), W⟩_p = Df(p)[X] = ⟨\operatorname{grad}f(p), W⟩ = ⟨\operatorname{proj}_{T_p\mathcal M}(\operatorname{grad}f(p)),W⟩
\quad\text{for all } W \in T_p\mathcal M.
```
Comparing the first and the last term, the remaining computation is the function [`change_representer`](@ref change_representer(M::AbstractManifold, G2::AbstractMetric, p, X)).

This method can also be implemented directly, if a more efficient/stable version is known.

The function is inspired by `egrad2rgrad` in the [Matlab package Manopt](https://manopt.org).
"""
function riemannian_gradient(
    M::AbstractManifold,
    p,
    Y;
    embedding_metric::AbstractMetric=EuclideanMetric(),
)
    X = zero_vector(M, p)
    riemannian_gradient!(M, X, p, Y; embedding_metric=embedding_metric)
    return X
end

function riemannian_gradient!(
    M::AbstractManifold,
    X,
    p,
    Y;
    embedding_metric=EuclideanMetric(),
)
    project!(M, X, p, Y)
    change_representer!(M, X, embedding_metric, p, X)
    return X
end

function gradient(
    M::AbstractManifold,
    f,
    p,
    backend::RiemannianProjectionBackend;
    kwargs...,
)
    amb_grad = Manifolds._gradient(f, p, backend.diff_backend)
    return riemannian_gradient(M, p, amb_grad; kwargs...)
end

function gradient!(
    M::AbstractManifold,
    f,
    X,
    p,
    backend::RiemannianProjectionBackend;
    kwargs...,
)
    amb_grad = embed(M, p, X)
    Manifolds._gradient!(f, amb_grad, p, backend.diff_backend)
    riemannian_gradient!(M, X, p, amb_grad; kwargs...)
    return X
end

function jacobian(
    M_dom::AbstractManifold,
    M_codom::AbstractManifold,
    f,
    p,
    backend::TangentDiffBackend,
)
    X = get_coordinates(M_dom, p, zero_vector(M_dom, p), backend.basis_arg)
    q = f(p)
    onb_coords = Manifolds._jacobian(X, backend.diff_backend) do Y
        return get_coordinates(
            M_codom,
            q,
            inverse_retract(
                M_codom,
                q,
                f(
                    retract(
                        M_dom,
                        p,
                        get_vector(M_dom, p, Y, backend.basis_arg),
                        backend.retraction,
                    ),
                ),
                backend.inverse_retraction,
            ),
            backend.basis_val,
        )
    end
    return onb_coords
end
