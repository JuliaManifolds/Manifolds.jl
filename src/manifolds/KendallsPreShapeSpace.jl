@doc raw"""
    KendallsPreShapeSpace{T} <: AbstractSphere{ℝ}

Kendall's pre-shape space of ``k`` landmarks in ``ℝ^n`` represented by n×k matrices.
In each row the sum of elements of a matrix is equal to 0. The Frobenius norm of the matrix
is equal to 1 [Kendall:1984](@cite)[Kendall:1989](@cite).

The space can be interpreted as tuples of ``k`` points in ``ℝ^n`` up to simultaneous
translation and scaling of all points, so this can be thought of as a quotient manifold.

# Constructor

    KendallsPreShapeSpace(n::Int, k::Int; parameter::Symbol=:type)

# See also
[`KendallsShapeSpace`](@ref), esp. for the references
"""
struct KendallsPreShapeSpace{T} <: AbstractSphere{ℝ}
    size::T
end

function KendallsPreShapeSpace(n::Int, k::Int; parameter::Symbol = :type)
    size = wrap_type_parameter(parameter, (n, k))
    return KendallsPreShapeSpace{typeof(size)}(size)
end

representation_size(M::KendallsPreShapeSpace) = get_parameter(M.size)

"""
    check_point(M::KendallsPreShapeSpace, p; atol=sqrt(max_eps(X, Y)), kwargs...)

Check whether `p` is a valid point on [`KendallsPreShapeSpace`](@ref), i.e. whether
each row has zero mean. Other conditions are checked via embedding in [`ArraySphere`](@ref).
"""
function check_point(
        M::KendallsPreShapeSpace,
        p;
        atol::Real = sqrt(eps(float(eltype(p)))),
        kwargs...,
    )
    for p_row in eachrow(p)
        if !isapprox(mean(p_row), 0; atol = atol, kwargs...)
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
function check_vector(
        M::KendallsPreShapeSpace,
        p,
        X;
        atol::Real = sqrt(eps(float(eltype(X)))),
        kwargs...,
    )
    for X_row in eachrow(X)
        if !isapprox(mean(X_row), 0; atol = atol, kwargs...)
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
get_embedding(::KendallsPreShapeSpace)

function get_embedding(::KendallsPreShapeSpace{TypeParameter{Tuple{n, k}}}) where {n, k}
    return ArraySphere(n, k)
end
function get_embedding(M::KendallsPreShapeSpace{Tuple{Int, Int}})
    n, k = get_parameter(M.size)
    return ArraySphere(n, k; parameter = :field)
end

function ManifoldsBase.get_embedding_type(::KendallsPreShapeSpace)
    return ManifoldsBase.EmbeddedSubmanifoldType()
end

@doc raw"""
    manifold_dimension(M::KendallsPreShapeSpace)

Return the dimension of the [`KendallsPreShapeSpace`](@ref) manifold `M`. The dimension is
given by ``n(k - 1) - 1``.
"""
function manifold_dimension(M::KendallsPreShapeSpace)
    n, k = get_parameter(M.size)
    return n * (k - 1) - 1
end

@doc raw"""
    helmert_submatrix(::Type{T}, k::Int)

Return the (`k`-1)×`k` matrix below the first row of the Helmert matrix of order `k`, whose
`j`-th row is ``(h_j, …, h_j, -jh_j, 0, …, 0)`` with ``h_j = (j(j+1))^{-1/2}``. With the first
row ``(1/\sqrt{k}, …, 1/\sqrt{k})`` the Helmert matrix is orthogonal, see Eq. (8.87) in
[Gentle:2017; Section 8.8.1](@cite).
"""
function helmert_submatrix(::Type{T}, k::Int) where {T}
    H = zeros(T, k - 1, k)
    for j in 1:(k - 1)
        h = inv(sqrt(T(j) * (j + 1)))
        H[j, 1:j] .= h
        H[j, j + 1] = -j * h
    end
    return H
end

@doc raw"""
    get_coordinates(M::KendallsPreShapeSpace, p, X, ::DefaultOrthonormalBasis)

Compute the coordinates of the tangent vector `X` at `p` on the [`KendallsPreShapeSpace`](@ref) `M`
in an orthonormal basis. With the [`helmert_submatrix`](@ref Manifolds.helmert_submatrix) ``H``,
``p ↦ pH^{\mathrm{T}}`` maps `M` isometrically onto the [`ArraySphere`](@ref)`(n, k-1)`, see
[Kendall:1989; Section 2](@cite), and the coordinates are those of ``XH^{\mathrm{T}}`` at
``pH^{\mathrm{T}}`` on that sphere.
"""
get_coordinates(::KendallsPreShapeSpace, p, X, ::DefaultOrthonormalBasis)

function get_coordinates_orthonormal!(M::KendallsPreShapeSpace, c, p, X, ::RealNumbers)
    n, k = get_parameter(M.size)
    H = helmert_submatrix(eltype(p), k)
    return get_coordinates_orthonormal!(ArraySphere(n, k - 1), c, p * H', X * H', ℝ)
end

@doc raw"""
    get_vector(M::KendallsPreShapeSpace, p, c, ::DefaultOrthonormalBasis)

Compute the tangent vector at `p` on the [`KendallsPreShapeSpace`](@ref) `M` with the coordinates `c`
in an orthonormal basis. It is ``YH``, where ``H`` is the
[`helmert_submatrix`](@ref Manifolds.helmert_submatrix) and ``Y`` is the tangent vector with the
coordinates `c` at ``pH^{\mathrm{T}}`` on the [`ArraySphere`](@ref)`(n, k-1)`, see
[Kendall:1989; Section 2](@cite).
"""
get_vector(::KendallsPreShapeSpace, p, c, ::DefaultOrthonormalBasis)

function get_vector_orthonormal!(M::KendallsPreShapeSpace, Y, p, c, ::RealNumbers)
    n, k = get_parameter(M.size)
    H = helmert_submatrix(eltype(p), k)
    Y .= get_vector(ArraySphere(n, k - 1), p * H', c, DefaultOrthonormalBasis()) * H
    return Y
end

"""
    project(M::KendallsPreShapeSpace, p)

Project point `p` from the embedding to [`KendallsPreShapeSpace`](@ref) by selecting
the right element from the orthogonal section representing the quotient manifold `M`.
See Section 3.7 of [SrivastavaKlassen:2016](@cite) for details.

The method computes the mean of the landmarks and moves them to make their mean zero;
afterwards the Frobenius norm of the landmarks (as a matrix) is normalised to fix the scaling.
"""
project(::KendallsPreShapeSpace, p)

function project!(::KendallsPreShapeSpace, q, p)
    q .= p .- mean(p, dims = 2)
    q ./= norm(q)
    return q
end

"""
    project(M::KendallsPreShapeSpace, p, X)

Project tangent vector `X` at point `p` from the embedding to [`KendallsPreShapeSpace`](@ref)
by selecting the right element from the tangent space to orthogonal section representing the
quotient manifold `M`. See Section 3.7 of [SrivastavaKlassen:2016](@cite) for details.
"""
project(::KendallsPreShapeSpace, p, X)

function project!(::KendallsPreShapeSpace, Y, p, X)
    Y .= X .- mean(X, dims = 2)
    Y .-= dot(p, Y) .* p
    return Y
end

function Random.rand!(
        rng::AbstractRNG,
        M::KendallsPreShapeSpace,
        pX;
        vector_at = nothing,
        σ = one(eltype(pX)),
    )
    if vector_at === nothing
        project!(M, pX, randn(rng, representation_size(M)))
    else
        n = σ * randn(rng, size(pX)) # Gaussian in embedding
        project!(M, pX, vector_at, n) #project to TpM (keeps Gaussianness)
    end
    return pX
end

function Base.show(io::IO, ::KendallsPreShapeSpace{TypeParameter{Tuple{n, k}}}) where {n, k}
    return print(io, "KendallsPreShapeSpace($n, $k)")
end
function Base.show(io::IO, M::KendallsPreShapeSpace{Tuple{Int, Int}})
    n, k = get_parameter(M.size)
    return print(io, "KendallsPreShapeSpace($n, $k; parameter=:field)")
end
