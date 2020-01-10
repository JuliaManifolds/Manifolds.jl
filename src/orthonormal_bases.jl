
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
on a manifold. Stores tangent vectors in field `.vectors`.
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

"""
    PrecomputedOrthonormalBasis(vectors::AbstractVector)

A precomputed orthonormal basis at a point on a manifold.
"""
struct PrecomputedOrthonormalBasis{TV<:AbstractVector} <: AbstractPrecomputedOrthonormalBasis
    vectors::TV
end

"""
    DiagonalizingOrthonormalBasis(vectors, kappas)
"""
struct DiagonalizingOrthonormalBasis{TV<:AbstractVector, TK<:AbstractVector} <: AbstractPrecomputedOrthonormalBasis
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

function get_coordinates(M::Manifold, x, v, B::PrecomputedOrthonormalBasis)
    return map(vb -> real(inner(M, x, v, vb)), B.vectors)
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

function get_vector(M::Manifold, x, v, B::PrecomputedOrthonormalBasis)
    # quite convoluted but:
    #  1) preserves the correct `eltype`
    #  2) guarantees a reasonable array type `vout`
    #     (for example scalar * `SizedArray` is an `SArray`)
    vt = v[1] .* B.vectors[1]
    vout = similar(B.vectors[1], eltype(vt))
    copyto!(vout, vt)
    for i in 2:length(v)
        vout .+= v[i] .* B.vectors[i]
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

basis(M::Manifold, x, B::AbstractPrecomputedOrthonormalBasis) = B

function basis(M::ArrayManifold, x, B::AbstractPrecomputedOrthonormalBasis)
    N = length(B)
    M_dim = manifold_dimension(M)
    N == M_dim || throw(ArgumentError("Incorrect number of basis vectors; expected: $M_dim, given: $N"))
    for i in 1:N
        vi_norm = norm(M, x, B.vectors[i])
        isapprox(vi_norm, 1) || throw(ArgumentError("vector number $i is not normalized (norm = $vi_norm)"))
        for j in i+1:N
            dot_val = real(inner(M, x, B.vectors[i], B.vectors[j]))
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
