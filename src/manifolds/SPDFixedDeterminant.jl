@doc raw"""
    SPDFixedDeterminant{T,D} <: AbstractDecoratorManifold{ℝ}

The manifold of symmetric positive definite matrices of fixed determinant ``d > 0``, i.e.

````math
\mathcal P_d(n) =
\bigl\{
p ∈ ℝ^{n×n} \ \big|\ a^\mathrm{T}pa > 0 \text{ for all } a ∈ ℝ^{n}\backslash\{0\}
  \text{ and } \det(p) = d
\bigr\}.
````

This manifold is modelled as an isometrically embedded manifold in [`SymmetricPositiveDefinite`](@ref)`(n)`,
see [`IsometricallyEmbeddedManifoldType`](@extref `ManifoldsBase.IsometricallyEmbeddedManifoldType`) for the implications.
Since it is a totally geodesic submanifold, the exponential and logarithmic maps,
parallel transport and the Riemann tensor are those of the embedding.

These matrices are sometimes also called [isochoric](https://en.wiktionary.org/wiki/isochoric), which refers to the interpretation of
the matrix representing an ellipsoid. All ellipsoids that represent points on this manifold have the same volume.

The tangent space is modelled the same as for [`SymmetricPositiveDefinite`](@ref)`(n)`
and consists of all symmetric matrices ``X`` whose product with ``p^{-1}`` has zero trace
```math
    T_p\mathcal P_d(n) =
    \bigl\{
        X \in \mathbb R^{n×n} \big|\ X=X^\mathrm{T} \text{ and } \operatorname{tr}(p^{-1}X) = 0
    \bigr\},
```
since for a constant determinant we require that ``0 = D\det(p)[X] = \det(p)\operatorname{tr}(p^{-1}X)`` for all tangent vectors ``X``.
At the identity matrix this is the set of symmetric matrices of zero trace.

# Constructor

    SPDFixedDeterminant(n::Int, d::Real=1.0; parameter::Symbol=:type)

Generate the manifold $\mathcal P_d(n) \subset \mathcal P(n)$ of determinant ``d``,
which defaults to `1.0`.

`parameter`: whether a type parameter should be used to store `n`. By default size
is stored in type. Value can either be `:field` or `:type`.
"""
struct SPDFixedDeterminant{T, TD <: Real} <: AbstractDecoratorManifold{ℝ}
    size::T
    d::TD
end

function SPDFixedDeterminant(n::Int, d::F = 1.0; parameter::Symbol = :type) where {F <: Real}
    @assert d > 0 "The determinant has to be positive but was provided as $d."
    n < 1 && throw(
        DomainError(
            n,
            "The matrix size n of SPD matrices with fixed determinant must be positive, but was $(n).",
        ),
    )
    size = wrap_type_parameter(parameter, (n,))
    return SPDFixedDeterminant{typeof(size), F}(size, d)
end

@doc raw"""
    check_point(M::SPDFixedDeterminant, p; kwargs...)

Check whether `p` is a valid manifold point on the [`SPDFixedDeterminant`](@ref)`(n,d)` `M`, i.e.
whether `p` is a [`SymmetricPositiveDefinite`](@ref) matrix of size `(n, n)`

with determinant ``\det(p) = ```M.d`.

The tolerance for the determinant of `p` can be set using `kwargs...`.
"""
function check_point(M::SPDFixedDeterminant, p; kwargs...)
    if !isapprox(det(p), M.d; kwargs...)
        return DomainError(
            det(p),
            "The point $(p) does not lie on $M, since it does not have a determinant $(M.d).",
        )
    end
    return nothing
end

@doc raw"""
    check_vector(M::SPDFixedDeterminant, p, X; atol, rtol, kwargs... )

Check whether `X` is a tangent vector to manifold point `p` on the
[`SPDFixedDeterminant`](@ref) `M`.
`X` has to be a tangent vector on [`SymmetricPositiveDefinite`](@ref), so a symmetric matrix,
and additionally fulfill ``\operatorname{tr}(p^{-1}X) = 0``
up to `max(atol, rtol * sqrt(n) * norm(M, p, X))`.
The relative tolerance `rtol` refers to the size of `X`; its default is the one of `isapprox`.
"""
function check_vector(
        M::SPDFixedDeterminant, p, X::T;
        atol::Real = sqrt(prod(representation_size(M))) * eps(real(float(number_eltype(T)))),
        rtol::Real = sqrt(eps(real(float(number_eltype(T))))),
        kwargs...,
    ) where {T}
    n = get_parameter(M.size)[1]
    r = abs(tr(p \ X))
    if !(r <= atol || r <= rtol * sqrt(n) * norm(M, p, X))
        return DomainError(
            r,
            "The vector $(X) is not a tangent vector to $(p) on $(M), since the trace of p^{-1}X does not vanish.",
        )
    end
    return nothing
end

embed(M::SPDFixedDeterminant, p) = copy(M, p)
embed(M::SPDFixedDeterminant, p, X) = copy(M, X)
embed!(M::SPDFixedDeterminant, q, p) = copyto!(M, q, p)
embed!(M::SPDFixedDeterminant, Y, p, X) = copyto!(M, Y, p, X)

function get_embedding(::SPDFixedDeterminant{TypeParameter{Tuple{n}}}) where {n}
    return SymmetricPositiveDefinite(n)
end
function get_embedding(M::SPDFixedDeterminant{Tuple{Int}})
    n = get_parameter(M.size)[1]
    return SymmetricPositiveDefinite(n; parameter = :field)
end

function ManifoldsBase.get_embedding_type(::SPDFixedDeterminant)
    return ManifoldsBase.IsometricallyEmbeddedManifoldType()
end

"""
    exp(M::SPDFixedDeterminant, p, X)

Compute the exponential map on the [`SPDFixedDeterminant`](@ref) `M`, which is the one of
the [`SymmetricPositiveDefinite`](@ref) matrices, since `M` is a totally geodesic submanifold.
"""
exp(::SPDFixedDeterminant, ::Any, ::Any)

exp!(M::SPDFixedDeterminant, q, p, X) = exp!(get_embedding(M), q, p, X)

"""
    injectivity_radius(M::SPDFixedDeterminant[, p])

Return the injectivity radius of the [`SPDFixedDeterminant`](@ref) `M`, which is ``∞`` as for
the [`SymmetricPositiveDefinite`](@ref) matrices, since a totally geodesic submanifold of a
Hadamard manifold is a Hadamard manifold. For ``n = 1`` the manifold consists of a single point
and the injectivity radius is ``0``.
"""
function injectivity_radius(M::SPDFixedDeterminant)
    return get_parameter(M.size)[1] == 1 ? 0.0 : Inf
end
injectivity_radius(M::SPDFixedDeterminant, p) = injectivity_radius(M)
injectivity_radius(M::SPDFixedDeterminant, ::AbstractRetractionMethod) = injectivity_radius(M)
function injectivity_radius(M::SPDFixedDeterminant, p, ::AbstractRetractionMethod)
    return injectivity_radius(M)
end

"""
    is_flat(M::SPDFixedDeterminant)

Return true if the [`SPDFixedDeterminant`](@ref) `M` is a single point, i.e. for ``n = 1``,
and false otherwise.
"""
is_flat(M::SPDFixedDeterminant) = get_parameter(M.size)[1] == 1

"""
    log(M::SPDFixedDeterminant, p, q)

Compute the logarithmic map on the [`SPDFixedDeterminant`](@ref) `M`, which is the one of
the [`SymmetricPositiveDefinite`](@ref) matrices, since `M` is a totally geodesic submanifold.
"""
log(::SPDFixedDeterminant, ::Any, ::Any)

log!(M::SPDFixedDeterminant, X, p, q) = log!(get_embedding(M), X, p, q)

@doc raw"""
    manifold_dimension(M::SPDFixedDeterminant)

Return the manifold dimension of the [`SPDFixedDeterminant`](@ref) manifold `M`
which is given by

````math
\dim \mathcal P_d(n) = \frac{n(n+1)}{2} - 1.
````
"""

function manifold_dimension(M::SPDFixedDeterminant)
    return manifold_dimension(get_embedding(M)) - 1
end

@doc raw"""
    q = project(M::SPDFixedDeterminant, p)
    project!(M::SPDFixedDeterminant, q, p)

Project the symmetric positive definite (s.p.d.) matrix `p` from the embedding onto the
(sub-)manifold of s.p.d. matrices of determinant `M.d` (in place of `q`).

The formula reads

```math
q = \Bigl(\frac{d}{\det(p)}\Bigr)^{\frac{1}{n}}p
```
"""
project(M::SPDFixedDeterminant, p)

function project!(M::SPDFixedDeterminant, q, p)
    n = get_parameter(M.size)[1]
    q .= (M.d / det(p))^(1 / n) .* p
    return q
end

@doc raw"""
    Y = project(M::SPDFixedDeterminant, p, X)
    project!(M::SPDFixedDeterminant, Y, p, X)

Project the symmetric matrix `X` onto the tangent space at `p` of the
(sub-)manifold of s.p.d. matrices of determinant `M.d` (in place of `Y`).
This subtracts the multiple of `p` that makes the trace of ``p^{-1}X`` vanish.
"""
project(M::SPDFixedDeterminant, p, X)

function project!(M::SPDFixedDeterminant, Y, p, X)
    n = get_parameter(M.size)[1]
    copyto!(M, Y, p, X)
    Y .-= (tr(p \ X) / n) .* p
    return Y
end

"""
    vector_transport_to(M::SPDFixedDeterminant, p, X, q, ::ProjectionTransport)

Transport the tangent vector `X` at `p` to `q` by projecting it onto the tangent space
at `q`, see [`project`](@ref project(::SPDFixedDeterminant, ::Any, ::Any)).
"""
vector_transport_to(::SPDFixedDeterminant, ::Any, ::Any, ::Any, ::ProjectionTransport)

vector_transport_to_project!(M::SPDFixedDeterminant, Y, p, X, q) = project!(M, Y, q, X)

"""
    parallel_transport_to(M::SPDFixedDeterminant, p, X, q)

Compute the parallel transport of `X` from `p` to `q` on the [`SPDFixedDeterminant`](@ref) `M`,
which is the one of the [`SymmetricPositiveDefinite`](@ref) matrices, since `M` is a
totally geodesic submanifold.
"""
parallel_transport_to(::SPDFixedDeterminant, ::Any, ::Any, ::Any)

function parallel_transport_to!(M::SPDFixedDeterminant, Y, p, X, q)
    return parallel_transport_to!(get_embedding(M), Y, p, X, q)
end

"""
    riemann_tensor(M::SPDFixedDeterminant, p, X, Y, Z)

Compute the Riemann tensor ``R(X,Y)Z`` at `p` on the [`SPDFixedDeterminant`](@ref) `M`,
which is the one of the [`SymmetricPositiveDefinite`](@ref) matrices, since `M` is a
totally geodesic submanifold.
"""
riemann_tensor(::SPDFixedDeterminant, ::Any, ::Any, ::Any, ::Any)

function riemann_tensor!(M::SPDFixedDeterminant, Xresult, p, X, Y, Z)
    return riemann_tensor!(get_embedding(M), Xresult, p, X, Y, Z)
end


@doc raw"""
    rand(M::SPDFixedDeterminant; vector_at=nothing, σ::Real=1.0)
    rand!(rng::AbstractRNG, M::SPDFixedDeterminant, pX; vector_at=nothing, σ::Real=1.0)

Draw a random point, or a random tangent vector at `vector_at`, on the
[`SymmetricPositiveDefinite`](@ref) matrices and project it onto `M`.
"""
function Random.rand!(
        rng::AbstractRNG, M::SPDFixedDeterminant, pX;
        vector_at = nothing, σ::Real = one(real(eltype(pX))),
    )
    if vector_at === nothing
        rand!(rng, get_embedding(M), pX; σ = σ)
        project!(M, pX, pX)
    else
        rand!(rng, get_embedding(M), pX; vector_at = vector_at, σ = σ)
        project!(M, pX, vector_at, pX)
    end
    return pX
end

function Base.show(io::IO, M::SPDFixedDeterminant{TypeParameter{Tuple{n}}}) where {n}
    return print(io, "SPDFixedDeterminant($n, $(M.d))")
end
function Base.show(io::IO, M::SPDFixedDeterminant{Tuple{Int}})
    n = get_parameter(M.size)[1]
    return print(io, "SPDFixedDeterminant($n, $(M.d); parameter=:field)")
end
