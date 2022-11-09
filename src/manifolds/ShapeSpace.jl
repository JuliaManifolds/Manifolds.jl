
@doc raw"""
    KendallsPreShapeSpace{n,k} <: AbstractSphere{ℝ}

Kendall's pre-shape space of ``k`` landmarks in $ℝ^n$ represented by n×k matrices.

# Constructor 

    KendallsPreShapeSpace(n::Int, k::Int)
"""
struct KendallsPreShapeSpace{n,k} <: AbstractSphere{ℝ} end

KendallsPreShapeSpace(n::Int, k::Int) = KendallsPreShapeSpace{n,k}()

function active_traits(f, ::KendallsPreShapeSpace, args...)
    return merge_traits(IsEmbeddedSubmanifold())
end

representation_size(::KendallsPreShapeSpace{n,k}) where {n,k} = (n, k)

"""
    check_point(M::KendallsPreShapeSpace, p; atol=sqrt(max_eps(X, Y)), kwargs...)

Check whether `p` is a valid point on [`KendallsPreShapeSpace`](@ref), i.e. whether
each row has zero mean. Other conditions are checked via embedding in [`ArraySphere`](@ref).
"""
function check_point(M::KendallsPreShapeSpace, p; atol=sqrt(eps(eltype(p))), kwargs...)
    for p_row in eachrow(p)
        if !isapprox(mean(p_row), 0; atol, kwargs...)
            return DomainError(
                mean(p_row),
                "The point $(p) does not lie on the $(M) since one of the rows does not have zero mean.",
            )
        end
    end
    return nothing
end

"""
    check_vector(M::KendallsPreShapeSpace, p, X; kwargs... )

Check whether `X` is a valid tangent vector on [`KendallsPreShapeSpace`](@ref), i.e. whether
each row has zero mean. Other conditions are checked via embedding in [`ArraySphere`](@ref).
"""
function check_vector(M::KendallsPreShapeSpace, p, X; atol=sqrt(max_eps(X, Y)), kwargs...)
    for X_row in eachrow(X)
        if !isapprox(mean(X_row), 0; atol, kwargs...)
            return DomainError(
                mean(X_row),
                "The vector $(X) is not a tangent vector to $(p) on $(M), since one of the rows does not have zero mean.",
            )
        end
    end
    return nothing
end

embed(::KendallsPreShapeSpace, p) = p
embed(::KendallsPreShapeSpace, p, X) = X

function get_embedding(::KendallsPreShapeSpace{N,K}) where {N,K}
    return ArraySphere(N, K)
end

manifold_dimension(::KendallsPreShapeSpace{n,k}) where {n,k} = n * (k - 1) - 1

function project!(::KendallsPreShapeSpace, q, p)
    q .= p .- mean(p, dims=2)
    q ./= norm(q)
    return q
end

function project!(::KendallsPreShapeSpace, Y, p, X)
    Y .= X .- mean(X, dims=2)
    Y .-= dot(p, Y) .* p
    return Y
end

function Random.rand!(M::KendallsPreShapeSpace, pX; vector_at=nothing, σ=one(eltype(pX)))
    if vector_at === nothing
        project!(M, pX, randn(representation_size(M)))
    else
        n = σ * randn(size(pX)) # Gaussian in embedding
        project!(M, pX, vector_at, n) # project to TpM (keeps Gaussianness)
    end
    return pX
end
function Random.rand!(
    rng::AbstractRNG,
    M::KendallsPreShapeSpace,
    pX;
    vector_at=nothing,
    σ=one(eltype(pX)),
)
    if vector_at === nothing
        project!(M, pX, randn(rng, representation_size(M)))
    else
        n = σ * randn(rng, size(pX)) # Gaussian in embedding
        project!(M, pX, vector_at, n) #project to TpM (keeps Gaussianness)
    end
    return pX
end

###

@doc raw"""
    KendallsShapeSpace{n,k} <: AbstractDecoratorManifold{ℝ}

Kendall's shape space, defined as quotient of a [`KendallsPreShapeSpace`](@ref)
(represented by n×k matrices) by the action [`ColumnwiseMultiplicationAction`](@ref).

# Constructor 

    KendallsShapeSpace(n::Int, k::Int)
"""
struct KendallsShapeSpace{n,k} <: AbstractDecoratorManifold{ℝ} end

KendallsShapeSpace(n::Int, k::Int) = KendallsShapeSpace{n,k}()

function active_traits(f, ::KendallsShapeSpace, args...)
    return merge_traits(IsIsometricEmbeddedManifold(), IsQuotientManifold())
end

function get_orbit_action(M::KendallsShapeSpace{n,k}) where {n,k}
    return ColumnwiseMultiplicationAction(M, SpecialOrthogonal(n))
end

@doc raw"""
    get_total_space(::Grassmann{n,k})

Return the total space of the [`KendallsShapeSpace`](@ref) manifold, which is the
[`KendallsPreShapeSpace`](@ref) manifold.
"""
get_total_space(::KendallsShapeSpace{n,k}) where {n,k} = KendallsPreShapeSpace(n, k)

function distance(M::KendallsShapeSpace, p, q)
    A = get_orbit_action(M)
    a = optimal_alignment(A, p, q)
    rot_q = apply(A, a, q)
    return distance(get_embedding(M), p, rot_q)
end

"""
    exp(M::KendallsShapeSpace, p, X)

Compute the exponential map on [`KendallsShapeSpace`](@ref) `M`. See [^Guigui2021] for discussion
about its computation.

[^Guigui2021]:
    > N. Guigui, E. Maignant, A. Trouvé, and X. Pennec, “Parallel Transport on Kendall Shape
    > Spaces,” in Geometric Science of Information, Cham, 2021, pp. 103–110.
    > doi: 10.1007/978-3-030-80209-7_12.
"""
exp(M::KendallsShapeSpace, p, X)

function exp!(M::KendallsShapeSpace, q, p, X)
    return exp!(get_embedding(M), q, p, X)
end

embed(::KendallsShapeSpace, p) = p
embed(::KendallsShapeSpace, p, X) = X

function get_embedding(::KendallsShapeSpace{N,K}) where {N,K}
    return KendallsPreShapeSpace(N, K)
end

function Base.isapprox(M::KendallsShapeSpace, p, X, Y; atol=sqrt(max_eps(X, Y)), kwargs...)
    return isapprox(norm(M, p, X - Y), 0; atol=atol, kwargs...)
end
function Base.isapprox(M::KendallsShapeSpace, p, q; atol=sqrt(max_eps(p, q)), kwargs...)
    return isapprox(distance(M, p, q), 0; atol=atol, kwargs...)
end

"""
    log(M::KendallsShapeSpace, p, q)

Compute the logarithmic map on [`KendallsShapeSpace`](@ref) `M`. See [^Guigui2021] for discussion
about its computation.

[^Guigui2021]:
    > N. Guigui, E. Maignant, A. Trouvé, and X. Pennec, “Parallel Transport on Kendall Shape
    > Spaces,” in Geometric Science of Information, Cham, 2021, pp. 103–110.
    > doi: 10.1007/978-3-030-80209-7_12.
"""
log(M::KendallsShapeSpace, p, q)

function log!(M::KendallsShapeSpace, X, p, q)
    A = get_orbit_action(M)
    a = optimal_alignment(A, p, q)
    rot_q = apply(A, a, q)
    return log!(get_embedding(M), X, p, rot_q)
end

@doc raw"""
    manifold_dimension(M::KendallsShapeSpace)

Return the dimension of the [`KendallsShapeSpace`](@ref) manifold `M`. The dimension is given by
``nk - 1 - n(n - 1)/2``

"""
function manifold_dimension(::KendallsShapeSpace{n,k}) where {n,k}
    return n * (k - 1) - 1 - div(n * (n - 1), 2)
end

function project!(M::KendallsShapeSpace, q, p)
    return project!(get_embedding(M), q, p)
end

function project!(M::KendallsShapeSpace, Y, p, X)
    return project!(get_embedding(M), Y, p, X)
end

@doc raw"""
    rand(::KendallsShapeSpace; vector_at=nothing)

When `vector_at` is `nothing`, return a random point `x` on the [`KendallsShapeSpace`](@ref)
manifold `M` by generating a random point in the embedding.

When `vector_at` is not `nothing`, return a random vector from the tangent space
with mean zero and standard deviation `σ`.
"""
rand(::KendallsShapeSpace; σ::Real=1.0)

function Random.rand!(M::KendallsShapeSpace{n,k}, pX; vector_at=nothing) where {n,k}
    rand!(get_embedding(M), pX; vector_at=vector_at)
    return pX
end
function Random.rand!(
    rng::AbstractRNG,
    M::KendallsShapeSpace{n,k},
    pX;
    vector_at=nothing,
    σ::Real=one(eltype(pX)),
) where {n,k}
    rand!(rng, get_embedding(M), pX; vector_at=vector_at, σ=σ)
    return pX
end
