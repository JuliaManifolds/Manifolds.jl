
@doc raw"""
    KendallsPreShapeSpace{n,k} <: AbstractSphere{ℝ}

Kendall's pre-shape space of ``k`` landmarks in ``ℝ^n`` represented by n×k matrices.
In each row the sum of elements of a matrix is equal to 0. The Frobenius norm of the matrix
is equal to 1 [^Kendall1984][^Kendall1989].

The space can be interpreted as tuples of ``k`` points in ``ℝ^n`` up to simultaneous
translation and scaling of all points, so this can be thought of as a quotient manifold.

# Constructor 

    KendallsPreShapeSpace(n::Int, k::Int)

# References

[^Kendall1989]:
    > D. G. Kendall, “A Survey of the Statistical Theory of Shape,” Statist. Sci., vol. 4,
    > no. 2, pp. 87–99, May 1989,
    > doi: [10.1214/ss/1177012582](https://doi.org/10.1214/ss/1177012582).
[^Kendall1984]:
    > D. G. Kendall, “Shape Manifolds, Procrustean Metrics, and Complex Projective Spaces,”
    > Bull. London Math. Soc., vol. 16, no. 2, pp. 81–121, Mar. 1984
    > doi: [10.1112/blms/16.2.81](https://doi.org/10.1112/blms/16.2.81).
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
function check_vector(M::KendallsPreShapeSpace, p, X; atol=sqrt(eps(eltype(X))), kwargs...)
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

"""
    get_embedding(M::KendallsPreShapeSpace)

Return the space [`KendallsPreShapeSpace`](@ref) `M` is embedded in, i.e. [`ArraySphere`](@ref)
of matrices of the same shape.
"""
function get_embedding(::KendallsPreShapeSpace{N,K}) where {N,K}
    return ArraySphere(N, K)
end

@doc raw"""
    manifold_dimension(M::KendallsPreShapeSpace)

Return the dimension of the [`KendallsPreShapeSpace`](@ref) manifold `M`. The dimension is
given by ``n(k - 1) - 1``.
"""
manifold_dimension(::KendallsPreShapeSpace{n,k}) where {n,k} = n * (k - 1) - 1

"""
    project(M::KendallsPreShapeSpace, p)

Project point `p` from the embedding to [`KendallsPreShapeSpace`](@ref) by selecting
the right element from the orthogonal section representing the quotient manifold `M`.
See Section 3.7 of [^Srivastava2016] for details.

The method computes the mean of the landmarks and moves them to make their mean zero;
afterwards the Frobenius norm of the landmarks (as a matrix) is normalised to fix the scaling.
"""
project(::KendallsPreShapeSpace, p)

function project!(::KendallsPreShapeSpace, q, p)
    q .= p .- mean(p, dims=2)
    q ./= norm(q)
    return q
end

"""
    project(M::KendallsPreShapeSpace, p, X)

Project tangent vector `X` at point `p` from the embedding to [`KendallsPreShapeSpace`](@ref)
by selecting the right element from the tangent space to orthogonal section representing the
quotient manifold `M`. See Section 3.7 of [^Srivastava2016] for details.

# References

[^Srivastava2016]:
    > A. Srivastava and E. P. Klassen, Functional and Shape Data Analysis. Springer New York, 2016.
    > ISBN: 978-1-4939-4018-9.
    > doi: [10.1007/978-1-4939-4020-2](https://doi.org/10.1007/978-1-4939-4020-2).
"""
project(::KendallsPreShapeSpace, p, X)

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
