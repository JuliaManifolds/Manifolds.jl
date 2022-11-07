
@doc raw"""
    ShapeSpace{n,k} <: AbstractDecoratorManifold{ℝ}

Kendall's shape space, defined as quotient of a pre-shape space (equivalent to
[`ArraySphere`](@ref) represented by n×k matrices) by the action [`ColumnwiseMultiplicationAction`](@ref)

"""
struct ShapeSpace{n,k} <: AbstractDecoratorManifold{ℝ} end

ShapeSpace(n::Int, k::Int) = ShapeSpace{n,k}()

function active_traits(f, ::ShapeSpace, args...)
    return merge_traits(IsIsometricEmbeddedManifold(), IsQuotientManifold())
end

function get_orbit_action(M::ShapeSpace{n,k}) where {n,k}
    return ColumnwiseMultiplicationAction(M, SpecialOrthogonal(n))
end

@doc raw"""
    get_total_space(::Grassmann{n,k})

Return the total space of the [`ShapeSpace`](@ref) manifold, which is the corresponding
[`ArraySphere`](@ref) manifold.
"""
get_total_space(::ShapeSpace{n,k}) where {n,k} = ArraySphere(n, k)

function distance(M::ShapeSpace, p, q)
    A = get_orbit_action(M)
    a = optimal_alignment(A, p, q)
    rot_q = apply(A, a, q)
    return distance(get_embedding(M), p, rot_q)
end

"""
    exp(M::ShapeSpace, p, X)

Compute the exponential map on [`ShapeSpace`](@ref) `M`. See [^Guigui2021] for discussion
about its computation.

[^Guigui2021]:
    > N. Guigui, E. Maignant, A. Trouvé, and X. Pennec, “Parallel Transport on Kendall Shape
    > Spaces,” in Geometric Science of Information, Cham, 2021, pp. 103–110.
    > doi: 10.1007/978-3-030-80209-7_12.
"""
exp(M::ShapeSpace, p, X)

function exp!(M::ShapeSpace, q, p, X)
    return exp!(get_embedding(M), q, p, X)
end

embed(::ShapeSpace, p) = p
embed(::ShapeSpace, p, X) = X

function get_embedding(::ShapeSpace{N,K}) where {N,K}
    return ArraySphere(N, K)
end

function Base.isapprox(M::ShapeSpace, p, X, Y; atol=sqrt(max_eps(X, Y)), kwargs...)
    return isapprox(norm(M, p, X - Y), 0; atol=atol, kwargs...)
end
function Base.isapprox(M::ShapeSpace, p, q; atol=sqrt(max_eps(p, q)), kwargs...)
    return isapprox(distance(M, p, q), 0; atol=atol, kwargs...)
end

"""
    log(M::ShapeSpace, p, q)

Compute the logarithmic map on [`ShapeSpace`](@ref) `M`. See [^Guigui2021] for discussion
about its computation.

[^Guigui2021]:
    > N. Guigui, E. Maignant, A. Trouvé, and X. Pennec, “Parallel Transport on Kendall Shape
    > Spaces,” in Geometric Science of Information, Cham, 2021, pp. 103–110.
    > doi: 10.1007/978-3-030-80209-7_12.
"""
log(M::ShapeSpace, p, q)

function log!(M::ShapeSpace, X, p, q)
    A = get_orbit_action(M)
    a = optimal_alignment(A, p, q)
    rot_q = apply(A, a, q)
    return log!(get_embedding(M), X, p, rot_q)
end

@doc raw"""
    manifold_dimension(M::ShapeSpace)

Return the dimension of the [`ShapeSpace`](@ref) manifold `M`. The dimension is given by
``nk - 1 - n(n - 1)/2``

"""
manifold_dimension(::ShapeSpace{n,k}) where {n,k} = n * k - 1 - div(n * (n - 1), 2)

@doc raw"""
    rand(::ShapeSpace; vector_at=nothing)

When `vector_at` is `nothing`, return a random point `x` on the [`ShapeSpace`](@ref)
manifold `M` by generating a random point in the embedding.

When `vector_at` is not `nothing`, return a random vector from the tangent space
with mean zero and standard deviation `σ`.
"""
rand(::ShapeSpace; σ::Real=1.0)

function Random.rand!(M::ShapeSpace{n,k}, pX; vector_at=nothing) where {n,k}
    rand!(get_embedding(M), pX; vector_at=vector_at)
    return pX
end
function Random.rand!(
    rng::AbstractRNG,
    M::ShapeSpace{n,k},
    pX;
    vector_at=nothing,
    σ::Real=one(eltype(pX)),
) where {n,k}
    rand!(rng, get_embedding(M), pX; vector_at=vector_at, σ=σ)
    return pX
end
