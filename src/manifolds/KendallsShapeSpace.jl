
@doc raw"""
    KendallsShapeSpace{n,k} <: AbstractDecoratorManifold{ℝ}

Kendall's shape space, defined as quotient of a [`KendallsPreShapeSpace`](@ref)
(represented by n×k matrices) by the action [`ColumnwiseMultiplicationAction`](@ref).

The space can be interpreted as tuples of ``k`` points in ``ℝ^n`` up to simultaneous
translation and scaling and rotation of all points [^Kendall1984][^Kendall1989].

This manifold possesses the [`IsQuotientManifold`](@ref) trait.

# Constructor

    KendallsShapeSpace(n::Int, k::Int)

# References

[^Kendall1989]:
    > D. G. Kendall, “A Survey of the Statistical Theory of Shape,” Statist. Sci., vol. 4,
    > no. 2, pp. 87–99, May 1989
    > doi: [10.1214/ss/1177012582](https://doi.org/10.1214/ss/1177012582).
[^Kendall1984]:
    > D. G. Kendall, “Shape Manifolds, Procrustean Metrics, and Complex Projective Spaces,”
    > Bull. London Math. Soc., vol. 16, no. 2, pp. 81–121, Mar. 1984
    > doi: [10.1112/blms/16.2.81](https://doi.org/10.1112/blms/16.2.81).
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

Compute the exponential map on [`KendallsShapeSpace`](@ref) `M`.
See [^Guigui2021] for discussion about its computation.

[^Guigui2021]:
    > N. Guigui, E. Maignant, A. Trouvé, and X. Pennec, “Parallel Transport on Kendall Shape
    > Spaces,” in Geometric Science of Information, Cham, 2021, pp. 103–110.
    > doi: [10.1007/978-3-030-80209-7_12](https://doi.org/10.1007/978-3-030-80209-7_12).
"""
exp(M::KendallsShapeSpace, p, X)

function exp!(M::KendallsShapeSpace, q, p, X)
    return exp!(get_embedding(M), q, p, X)
end

embed(::KendallsShapeSpace, p) = p
embed(::KendallsShapeSpace, p, X) = X

"""
    get_embedding(M::KendallsShapeSpace)

Get the manifold in which [`KendallsShapeSpace`](@ref) `M` is embedded, i.e.
[`KendallsPreShapeSpace`](@ref) of matrices of the same shape.
"""
function get_embedding(::KendallsShapeSpace{N,K}) where {N,K}
    return KendallsPreShapeSpace(N, K)
end

function horizontal_component!(::KendallsShapeSpace, Y, p, X)
    B = p * transpose(p)
    C = X * transpose(p) - p * transpose(X)
    A = sylvc(B, B, C)
    Y .= X .- A * p
    return Y
end

function inner(M::KendallsShapeSpace, p, X, Y)
    Xh = horizontal_component(M, p, X)
    Yh = horizontal_component(M, p, Y)
    return inner(get_embedding(M), p, Xh, Yh)
end

function _isapprox(M::KendallsShapeSpace, p, X, Y; atol=sqrt(max_eps(X, Y)), kwargs...)
    return isapprox(norm(M, p, X - Y), 0; atol=atol, kwargs...)
end
function _isapprox(M::KendallsShapeSpace, p, q; atol=sqrt(max_eps(p, q)), kwargs...)
    return isapprox(distance(M, p, q), 0; atol=atol, kwargs...)
end

"""
    is_flat(::KendallsShapeSpace)

Return false. [`KendallsShapeSpace`](@ref) is not a flat manifold.
"""
is_flat(M::KendallsShapeSpace) = false

"""
    log(M::KendallsShapeSpace, p, q)

Compute the logarithmic map on [`KendallsShapeSpace`](@ref) `M`.
See the [`exp`](@ref exp(::KendallsShapeSpace, ::Any, ::Any)onential map for more details
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

Return the dimension of the [`KendallsShapeSpace`](@ref) manifold `M`. The dimension is
given by ``n(k - 1) - 1 - n(n - 1)/2`` in the typical case where ``k \geq n+1``, and
``(k + 1)(k - 2) / 2`` otherwise, unless ``k`` is equal to 1, in which case the dimension
is 0. See [^Kendall1984] for a discussion of the over-dimensioned case.
"""
function manifold_dimension(::KendallsShapeSpace{n,k}) where {n,k}
    if k < n + 1 # over-dimensioned case
        if k == 1
            return 0
        else
            return div((k + 1) * (k - 2), 2)
        end
    else
        return n * (k - 1) - 1 - div(n * (n - 1), 2)
    end
end

function norm(M::KendallsShapeSpace, p, X)
    Xh = horizontal_component(M, p, X)
    return norm(get_embedding(M), p, Xh)
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
