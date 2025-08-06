@doc raw"""
    DeterminantOneMatrices{𝔽,T} <: AbstractDecoratorManifold{𝔽}

The [`AbstractManifold`](@extref `ManifoldsBase.AbstractManifold`)
consisting of the real- or complex-valued (invertible) matrices od determinant one,
that is the set

```math
\bigl\{p  ∈ 𝔽^{n×n}\ \big|\ \det(p) = 1 \bigr\},
```
where the field ``𝔽 ∈ \{ ℝ, ℂ\}``.

Note that this is a subset of [`InvertibleMatrices`](@ref),
and a superset of any of the [`GeneralUnitaryMatrices`](@ref)

The tangent space at any point `p` is the set of matrices with trace 0.

# Constructor

    DeterminantOneMatrices(n::Int, field::AbstractNumbers=ℝ)

Generate the manifold of ``n×n`` matrices of determinant one.
"""
struct DeterminantOneMatrices{𝔽,T} <: AbstractDecoratorManifold{𝔽}
    size::T
end

function DeterminantOneMatrices(n::Int, field::AbstractNumbers=ℝ; parameter::Symbol=:type)
    size = wrap_type_parameter(parameter, (n,))
    return DeterminantOneMatrices{field,typeof(size)}(size)
end

@doc raw"""
    check_point(M::DeterminantOneMatrices{n,𝔽}, p; kwargs...)

Check whether `p` is a valid manifold point on the [`DeterminantOneMatrices`](@ref) `M`, i.e.
whether `p` has a determinant of ``1``.

The check is perfomed with `isapprox` and all keyword arguments are passed to this
"""
function check_point(M::DeterminantOneMatrices, p; kwargs...)
    if !isapprox(det(p), 1; kwargs...)
        return DomainError(
            det(p),
            "The point $(p) does not lie on $(M), since its determinant is $(det(p)) and not 1.",
        )
    end
    return nothing
end

"""
    check_vector(M::DeterminantOneMatrices{n,𝔽}, p, X; kwargs... )

Check whether `X` is a tangent vector to manifold point `p` on the
[`DeterminantOneMatrices`](@ref) `M`, which are all matrices of size ``n×n``
with trace 0.
"""
function check_vector(M::DeterminantOneMatrices, p, X; kwargs...)
    if !isapprox(tr(X), 0; kwargs...)
        return DomainError(
            tr(X),
            "The tangent vector $(X) does not lie in the Tangent space at $(p) of $(M), since its trace is $(tr(X)) and not zero.",
        )
    end
    return nothing
end

embed(::DeterminantOneMatrices, p) = p
embed(::DeterminantOneMatrices, p, X) = X

function get_embedding(::DeterminantOneMatrices{𝔽,TypeParameter{Tuple{n}}}) where {n,𝔽}
    return Euclidean(n, n; field=𝔽)
end
function get_embedding(M::DeterminantOneMatrices{𝔽,Tuple{Int}}) where {𝔽}
    n = get_parameter(M.size)[1]
    return Euclidean(n, n; field=𝔽, parameter=:field)
end

@doc raw"""
    manifold_dimension(M::DeterminantOneMatrices{n,𝔽})

Return the dimension of the [`DeterminantOneMatrices`](@ref) matrix `M` over the number system
`𝔽`, which is one dimension less than its embedding, the [`Euclidean`](@ref)`(n, n; field=𝔽)`.
"""
function manifold_dimension(M::DeterminantOneMatrices{<:Any,𝔽}) where {𝔽}
    return manifold_dimension(get_embedding(M)) - 1
end

@doc raw"""
    project(G::DeterminantOneMatrices, p)
    project!(G::DeterminantOneMatrices, q, p)

Project ``p ∈ \mathrm{GL}(n, 𝔽)`` to the [`DeterminantOneMatrices`](@ref)
using the singular value decomposition of ``p = U S V^\mathrm{H}``.

The formula for the projection is

````math
\operatorname{proj}(p) = U S D V^\mathrm{H},
````
where

````math
D_{ij} = δ_{ij} \begin{cases}
    1            & \text{ if } i ≠ n \\
    \det(p)^{-1} & \text{ if } i = n
\end{cases}.
````

The operation can be done in-place of `q`.
"""
project(::DeterminantOneMatrices, p)

function project!(M::DeterminantOneMatrices, q, p)
    n = get_parameter(M.size)[1]
    detp = det(p)
    isapprox(detp, 1) && return copyto!(q, p)
    F = svd(p)
    q .= F.U .* F.S'
    q[:, n] ./= detp
    mul!_safe(q, q, F.Vt)
    return q
end

@doc raw"""
    project(G::DeterminantOneMatrices, p, X)
    project!(G::DeterminantOneMatrices, Y, p, X)

Orthogonally project ``X ∈ 𝔽^{n×n}`` onto the tangent space of ``p`` to the
[`DeterminantOneMatrices`](@ref).

This first changes the representation from `X` to the trace-zero component, i.e.
computes `Y = p \ X` and then subtracts `c = tr(Y) / n` from all diagonal entries.
"""
project(::DeterminantOneMatrices, p, X)

function project!(G::DeterminantOneMatrices, Y, p, X)
    n = get_parameter(G.size)[1]
    Y .= p \ X
    c = tr(Y) / n
    Y[diagind(n, n)] .-= c
    return Y
end

@doc raw"""
    Random.rand(M::DeterminantOneMatrices; vector_at=nothing, kwargs...)

If `vector_at` is `nothing`, return a random point on the [`DeterminantOneMatrices`](@ref)
manifold `M` by using `rand` in the embedding.

If `vector_at` is not `nothing`, return a random tangent vector from the tangent space of
the point `vector_at` on the [`DeterminantOneMatrices`](@ref) by using by using `rand` in the
embedding.
"""
rand(M::DeterminantOneMatrices; kwargs...)

function Random.rand!(M::DeterminantOneMatrices, pX; kwargs...)
    return rand!(Random.default_rng(), M, pX; kwargs...)
end
function Random.rand!(
    rng::AbstractRNG,
    M::DeterminantOneMatrices,
    pX;
    vector_at=nothing,
    kwargs...,
)
    rand!(rng, get_embedding(M), pX; kwargs...)
    if vector_at === nothing # for points ensure invertibility
        n = size(pX)[1]
        _ensure_nonzero_rng_determinant!(rng, get_embedding(M), pX; kwargs...)
        det_pX = det(pX)
        pX[1, :] ./= sign(det_pX)
        pX ./= abs(det_pX)^(1 / n)
    else # tangent vectors: trace 0
        pX[diagind(pX)] .= 0
    end
    return pX
end

function Base.show(io::IO, ::DeterminantOneMatrices{𝔽,TypeParameter{Tuple{n}}}) where {n,𝔽}
    return print(io, "DeterminantOneMatrices($(n), $(𝔽))")
end
function Base.show(io::IO, M::DeterminantOneMatrices{𝔽,Tuple{Int}}) where {𝔽}
    n = get_parameter(M.size)[1]
    return print(io, "DeterminantOneMatrices($(n), $(𝔽); parameter=:field)")
end
