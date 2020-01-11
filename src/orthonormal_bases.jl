
"""
    AbstractBasis

Abstract type that represents a basis on a manifold or a subset of it.
"""
abstract type AbstractBasis end

"""
    AbstractOrthonormalBasis

Abstract type that represents an orthonormal basis on a manifold or a subset of it.
"""
abstract type AbstractOrthonormalBasis <: AbstractBasis end

"""
    AbstractPrecomputedOrthonormalBasis

Abstract type that represents an orthonormal basis of the tangent space at a point
on a manifold. Tangent vectors can be obtained using function [`vectors`](@ref).

The vectors are not always fully precomputed because a partially precomputed
basis may be enough for implementing [`get_vector`](@ref) and [`get_coordinates`](@ref).
"""
abstract type AbstractPrecomputedOrthonormalBasis <: AbstractOrthonormalBasis end

"""
    ArbitraryOrthonormalBasis

An arbitrary orthonormal basis on a manifold. This will usually
be the fastest [`OrthonormalBasis`](@ref) available for a manifold.
"""
struct ArbitraryOrthonormalBasis <: AbstractOrthonormalBasis end

"""
    ProjectedOrthonormalBasis(method::Symbol)

An orthonormal basis that comes from orthonormalization of basis vectors
of the ambient space projected onto the subspace representing the tangent space
at a given point.

Available methods:
  - `:gram_schmidt` TODO
  - `:svd` uses SVD decomposition to orthogonalize projected vectors.
    The SVD-based method should be more numerically stable at the cost of
    an additional assumption (local metric tensor at a point where the
    basis is calculated has to be diagonal).
"""
struct ProjectedOrthonormalBasis{Method} <: AbstractOrthonormalBasis end

ProjectedOrthonormalBasis(method::Symbol) = ProjectedOrthonormalBasis{method}()

@doc doc"""
    DiagonalizingOrthonormalBasis(v)

An orthonormal basis `Ξ` as a vector of tangent vectors (of length determined by
[`manifold_dimension`](@ref)) in the tangent space that diagonalizes the curvature
tensor $R(u,v)w$ and where the direction `v` has curvature `0`.
"""
struct DiagonalizingOrthonormalBasis{TV} <: AbstractOrthonormalBasis
    v::TV
end

const ArbitraryOrDiagonalizingBasis = Union{ArbitraryOrthonormalBasis, DiagonalizingOrthonormalBasis}

"""
    PrecomputedOrthonormalBasis(vectors::AbstractVector)

A precomputed orthonormal basis at a point on a manifold.
"""
struct PrecomputedOrthonormalBasis{TV<:AbstractVector} <: AbstractPrecomputedOrthonormalBasis
    vectors::TV
end

@doc doc"""
    DiagonalizingOrthonormalBasis(vectors, kappas)

A precomputed orthonormal basis `Ξ` as a vector of tangent vectors (of length determined
by [`manifold_dimension`](@ref)) in the tangent space that diagonalizes the curvature
tensor $R(u,v)w$ with eigenvalues `kappas` and where the direction `v` has curvature `0`.
"""
struct PrecomputedDiagonalizingOrthonormalBasis{TV<:AbstractVector, TK<:AbstractVector} <: AbstractPrecomputedOrthonormalBasis
    vectors::TV
    kappas::TK
end

"""
    get_coordinates(M::Manifold, x, v, B::AbstractBasis)

Compute a one-dimentional vector of coefficients of the tangent vector `v`
at point denoted by `x` on manifold `M` in basis `B`.

Depending on the basis, `x` may not directly represent a point on the manifold.
For example if a basis transported along a curve is used, `x` may be the coordinate
along the curve.

See also: [`get_vector`](@ref), [`basis`](@ref)
"""
function get_coordinates(M::Manifold, x, v, B::AbstractBasis)
    error("get_coordinates not implemented for manifold of type $(typeof(M)) a point of type $(typeof(x)), tangent vector of type $(typeof(v)) and basis of type $(typeof(B)).")
end

function get_coordinates(M::Manifold, x, v, B::AbstractPrecomputedOrthonormalBasis)
    return map(vb -> real(inner(M, x, v, vb)), vectors(M, x, B))
end

"""
    get_vector(M::Manifold, x, v, B::AbstractBasis)

Convert a one-dimensional vector of coefficients in a basis `B` of
the tangent space at `x` on manifold `M` to a tangent vector `v` at `x`.

Depending on the basis, `x` may not directly represent a point on the manifold.
For example if a basis transported along a curve is used, `x` may be the coordinate
along the curve.

See also: [`get_coordinates`](@ref), [`basis`](@ref)
"""
function get_vector(M::Manifold, x, v, B::AbstractBasis)
    error("get_vector not implemented for manifold of type $(typeof(M)) a point of type $(typeof(x)), tangent vector of type $(typeof(v)) and basis of type $(typeof(B)).")
end

function get_vector(M::Manifold, x, v, B::AbstractPrecomputedOrthonormalBasis)
    # quite convoluted but:
    #  1) preserves the correct `eltype`
    #  2) guarantees a reasonable array type `vout`
    #     (for example scalar * `SizedArray` is an `SArray`)
    bvectors = vectors(M, x, B)
    vt = v[1] .* bvectors[1]
    vout = similar(bvectors[1], eltype(vt))
    copyto!(vout, vt)
    for i in 2:length(v)
        vout .+= v[i] .* bvectors[i]
    end
    return vout
end

"""
    basis(M::Manifold, x, B::AbstractBasis) -> AbstractBasis

Compute the basis vectors of the tangent space at a point on manifold `M`
represented by `x`.

Returned object derives from [`AbstractBasis`](@ref) and has a field `.vectors`
that stores tangent vectors.

See also: [`get_coordinates`](@ref), [`get_vector`](@ref)
"""
function basis(M::Manifold, x, B::AbstractBasis)
    error("basis not implemented for manifold of type $(typeof(M)) a point of type $(typeof(x)) and basis of type $(typeof(B)).")
end

"""
    basis(M::Manifold, x, B::ArbitraryOrthonormalBasis)

Compute the basis vectors of an [`ArbitraryOrthonormalBasis`](@ref).
"""
function basis(M::Manifold, x, B::ArbitraryOrthonormalBasis)
    dim = manifold_dimension(M)
    return PrecomputedOrthonormalBasis(
        [get_vector(M, x, [ifelse(i == j, 1, 0) for j in 1:dim], B) for i in 1:dim]
    )
end

basis(M::Manifold, x, B::AbstractPrecomputedOrthonormalBasis) = B

function basis(M::ArrayManifold, x, B::AbstractPrecomputedOrthonormalBasis)
    N = length(B)
    M_dim = manifold_dimension(M)
    N == M_dim || throw(ArgumentError("Incorrect number of basis vectors; expected: $M_dim, given: $N"))
    bvectors = vectors(M, x, B)
    for i in 1:N
        vi_norm = norm(M, x, bvectors[i])
        isapprox(vi_norm, 1) || throw(ArgumentError("vector number $i is not normalized (norm = $vi_norm)"))
        for j in i+1:N
            dot_val = real(inner(M, x, bvectors[i], bvectors[j]))
            isapprox(dot_val, 0; atol = eps(eltype(x))) || throw(ArgumentError("vectors number $i and $j are not orthonormal (inner product = $dot_val)"))
        end
    end
    return B
end

function _euclidean_basis_vector(x, i)
    y = zero(x)
    y[i] = 1
    return y
end

function basis(M::Manifold, x, B::ProjectedOrthonormalBasis{:svd})
    S = representation_size(M)
    PS = prod(S)
    dim = manifold_dimension(M)
    # projection
    # TODO: find a better way to obtain a basis of the ambient space
    vs = [convert(Vector, reshape(project_tangent(M, x, _euclidean_basis_vector(x, i)), PS)) for i in eachindex(x)]
    O = reduce(hcat, vs)
    # orthogonalization
    # TODO: try using rank-revealing QR here
    decomp = svd(O)
    rotated = Diagonal(decomp.S) * decomp.Vt
    vecs = [collect(reshape(rotated[i,:], S)) for i in 1:dim]
    # normalization
    for i in 1:dim
        i_norm = norm(M, x, vecs[i])
        vecs[i] /= i_norm
    end
    return PrecomputedOrthonormalBasis(vecs)
end

"""
    vectors(M::Manifold, x, B::AbstractBasis)

Get the basis vectors of basis `B` of the tangent space at point `x`.
"""
function vectors(M::Manifold, x, B::AbstractBasis)
    error("vectors not implemented for manifold of type $(typeof(M)) a point of type $(typeof(x)) and basis of type $(typeof(B)).")
end

vectors(::Manifold, x, B::PrecomputedOrthonormalBasis) = B.vectors
vectors(::Manifold, x, B::PrecomputedDiagonalizingOrthonormalBasis) = B.vectors

# related to DefaultManifold; to be moved to ManifoldsBase.jl in the future
function get_coordinates(M::ManifoldsBase.DefaultManifold, x, v, ::ArbitraryOrthonormalBasis)
    return reshape(v, manifold_dimension(M))
end
function get_vector(M::ManifoldsBase.DefaultManifold, x, v, ::ArbitraryOrthonormalBasis)
    return reshape(v, representation_size(M))
end

function basis(M::ManifoldsBase.DefaultManifold, x, ::ArbitraryOrthonormalBasis)
    return PrecomputedOrthonormalBasis([_euclidean_basis_vector(x, i) for i in eachindex(x)])
end
