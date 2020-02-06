"""
    AbstractBasis{𝔽}

Abstract type that represents a basis on a manifold or a subset of it.

The type parameter `𝔽` denotes the [`AbstractNumbers`](@ref) that will be used as scalars.
"""
abstract type AbstractBasis{𝔽} end

"""
    number_system(::AbstractBasis)

The number system used as scalars in the given basis.
"""
number_system(::AbstractBasis{𝔽}) where {𝔽} = 𝔽

"""
    AbstractOrthonormalBasis{𝔽}

Abstract type that represents an orthonormal basis on a manifold or a subset of it.

The type parameter `𝔽` denotes the [`AbstractNumbers`](@ref) that will be used as scalars.
"""
abstract type AbstractOrthonormalBasis{𝔽} <: AbstractBasis{𝔽} end

"""
    AbstractPrecomputedOrthonormalBasis{𝔽}

Abstract type that represents an orthonormal basis of the tangent space at a point
on a manifold. Tangent vectors can be obtained using function [`get_vectors`](@ref).

The vectors are not always fully precomputed because a partially precomputed
basis may be enough for implementing [`get_vector`](@ref) and [`get_coordinates`](@ref).

The type parameter `𝔽` denotes the [`AbstractNumbers`](@ref) that will be used as scalars.
"""
abstract type AbstractPrecomputedOrthonormalBasis{𝔽} <: AbstractOrthonormalBasis{𝔽} end

"""
    ArbitraryOrthonormalBasis(field::AbstractNumbers = ℝ)

An arbitrary orthonormal basis on a manifold. This will usually
be the fastest orthonormal basis available for a manifold.

The type parameter `field` denotes the [`AbstractNumbers`](@ref) that will be used as
scalars.
"""
struct ArbitraryOrthonormalBasis{𝔽} <: AbstractOrthonormalBasis{𝔽} end

ArbitraryOrthonormalBasis(field::AbstractNumbers = ℝ) = ArbitraryOrthonormalBasis{field}()

"""
    ProjectedOrthonormalBasis(method::Symbol, field::AbstractNumbers = ℝ)

An orthonormal basis that comes from orthonormalization of basis vectors
of the ambient space projected onto the subspace representing the tangent space
at a given point.

The type parameter `field` denotes the [`AbstractNumbers`](@ref) that will be used as
scalars.

Available methods:
  - `:gram_schmidt` uses a modified Gram-Schmidt orthonormalization.
  - `:svd` uses SVD decomposition to orthogonalize projected vectors.
    The SVD-based method should be more numerically stable at the cost of
    an additional assumption (local metric tensor at a point where the
    basis is calculated has to be diagonal).
"""
struct ProjectedOrthonormalBasis{Method,𝔽} <: AbstractOrthonormalBasis{𝔽} end

function ProjectedOrthonormalBasis(method::Symbol, field::AbstractNumbers = ℝ)
    return ProjectedOrthonormalBasis{method,field}()
end

@doc raw"""
    DiagonalizingOrthonormalBasis(frame_direction, field::AbstractNumbers = ℝ)

An orthonormal basis `Ξ` as a vector of tangent vectors (of length determined by
[`manifold_dimension`](@ref)) in the tangent space that diagonalizes the curvature
tensor $R(u,v)w$ and where the direction `frame_direction` $v$ has curvature `0`.

The type parameter `field` denotes the [`AbstractNumbers`](@ref) that will be used as
scalars.
"""
struct DiagonalizingOrthonormalBasis{TV,𝔽} <: AbstractOrthonormalBasis{𝔽}
    frame_direction::TV
end

function DiagonalizingOrthonormalBasis(X, field::AbstractNumbers = ℝ)
    return DiagonalizingOrthonormalBasis{typeof(X),field}(X)
end

const ArbitraryOrDiagonalizingBasis =
    Union{ArbitraryOrthonormalBasis,DiagonalizingOrthonormalBasis}

"""
    PrecomputedOrthonormalBasis(vectors::AbstractVector, field::AbstractNumbers = ℝ)

A precomputed orthonormal basis at a point on a manifold.

The type parameter `field` denotes the [`AbstractNumbers`](@ref) that will be used as
scalars.
"""
struct PrecomputedOrthonormalBasis{TV<:AbstractVector,𝔽} <:
       AbstractPrecomputedOrthonormalBasis{𝔽}
    vectors::TV
end

function PrecomputedOrthonormalBasis(vectors::AbstractVector, field::AbstractNumbers = ℝ)
    return PrecomputedOrthonormalBasis{typeof(vectors),field}(vectors)
end

@doc raw"""
    DiagonalizingOrthonormalBasis(vectors, kappas, field::AbstractNumbers = ℝ)

A precomputed orthonormal basis `Ξ` as a vector of tangent vectors (of length determined
by [`manifold_dimension`](@ref)) in the tangent space that diagonalizes the curvature
tensor $R(u,v)w$ with eigenvalues `kappas` and where the direction `v` has eigenvalue `0`.

The type parameter `field` denotes the [`AbstractNumbers`](@ref) that will be used as
scalars.
"""
struct PrecomputedDiagonalizingOrthonormalBasis{TV<:AbstractVector,TK<:AbstractVector,𝔽} <:
       AbstractPrecomputedOrthonormalBasis{𝔽}
    vectors::TV
    kappas::TK
end

function PrecomputedDiagonalizingOrthonormalBasis(
    vectors::AbstractVector,
    kappas::AbstractVector,
    field::AbstractNumbers = ℝ,
)
    return PrecomputedDiagonalizingOrthonormalBasis{typeof(vectors),typeof(kappas),field}(
        vectors,
        kappas,
    )
end

"""
    get_coordinates(M::Manifold, p, X, B::AbstractBasis)

Compute a one-dimensional vector of coefficients of the tangent vector `X`
at point denoted by `p` on manifold `M` in basis `B`.

Depending on the basis, `p` may not directly represent a point on the manifold.
For example if a basis transported along a curve is used, `p` may be the coordinate
along the curve.

See also: [`get_vector`](@ref), [`get_basis`](@ref)
"""
function get_coordinates(M::Manifold, p, X, B::AbstractBasis)
    error("get_coordinates not implemented for manifold of type $(typeof(M)) a point of type $(typeof(p)), tangent vector of type $(typeof(X)) and basis of type $(typeof(B)).")
end
function get_coordinates(M::Manifold, p, X, B::AbstractPrecomputedOrthonormalBasis{ℝ})
    return map(vb -> real(inner(M, p, X, vb)), get_vectors(M, p, B))
end
function get_coordinates(M::Manifold, p, X, B::AbstractPrecomputedOrthonormalBasis)
    return map(vb -> inner(M, p, X, vb), get_vectors(M, p, B))
end

"""
    get_vector(M::Manifold, p, X, B::AbstractBasis)

Convert a one-dimensional vector of coefficients in a basis `B` of
the tangent space at `p` on manifold `M` to a tangent vector `X` at `p`.

Depending on the basis, `p` may not directly represent a point on the manifold.
For example if a basis transported along a curve is used, `p` may be the coordinate
along the curve.

See also: [`get_coordinates`](@ref), [`get_basis`](@ref)
"""
function get_vector(M::Manifold, p, X, B::AbstractBasis)
    error("get_vector not implemented for manifold of type $(typeof(M)) a point of type $(typeof(p)), tangent vector of type $(typeof(X)) and basis of type $(typeof(B)).")
end
function get_vector(M::Manifold, p, X, B::AbstractPrecomputedOrthonormalBasis)
    # quite convoluted but:
    #  1) preserves the correct `eltype`
    #  2) guarantees a reasonable array type `Y`
    #     (for example scalar * `SizedArray` is an `SArray`)
    bvectors = get_vectors(M, p, B)
    if isa(bvectors[1], ProductRepr)
        Xt = X[1] * bvectors[1]
        Y = allocate(bvectors[1], eltype(Xt))
        copyto!(Y, Xt)
        for i = 2:length(X)
            Y += X[i] * bvectors[i]
        end
        return Y
    else
        Xt = X[1] .* bvectors[1]
        Y = allocate(bvectors[1], eltype(Xt))
        copyto!(Y, Xt)
        for i = 2:length(X)
            Y .+= X[i] .* bvectors[i]
        end
        return Y
    end
end

"""
    get_basis(M::Manifold, p, B::AbstractBasis) -> AbstractBasis

Compute the basis vectors of the tangent space at a point on manifold `M`
represented by `p`.

Returned object derives from [`AbstractBasis`](@ref) and may have a field `.vectors`
that stores tangent vectors or it may store them implicitly, in which case
the function [`get_vectors`](@ref) needs to be used to retrieve the basis vectors.

See also: [`get_coordinates`](@ref), [`get_vector`](@ref)
"""
function get_basis(M::Manifold, p, B::AbstractBasis)
    error("get_basis not implemented for manifold of type $(typeof(M)) a point of type $(typeof(p)) and basis of type $(typeof(B)).")
end
"""
    get_basis(M::Manifold, p, B::ArbitraryOrthonormalBasis)

Compute the basis vectors of an [`ArbitraryOrthonormalBasis`](@ref).
"""
function get_basis(M::Manifold, p, B::ArbitraryOrthonormalBasis)
    dim = manifold_dimension(M)
    return PrecomputedOrthonormalBasis([
        get_vector(M, p, [ifelse(i == j, 1, 0) for j = 1:dim], B) for i = 1:dim
    ])
end
get_basis(M::Manifold, p, B::AbstractPrecomputedOrthonormalBasis) = B
function get_basis(M::ArrayManifold, p, B::AbstractPrecomputedOrthonormalBasis{ℝ})
    bvectors = get_vectors(M, p, B)
    N = length(bvectors)
    M_dim = manifold_dimension(M)
    if N != M_dim
        throw(ArgumentError("Incorrect number of basis vectors; expected: $M_dim, given: $N"))
    end
    for i = 1:N
        Xi_norm = norm(M, p, bvectors[i])
        if !isapprox(Xi_norm, 1)
            throw(ArgumentError("vector number $i is not normalized (norm = $Xi_norm)"))
        end
        for j = i+1:N
            dot_val = real(inner(M, p, bvectors[i], bvectors[j]))
            if !isapprox(dot_val, 0; atol = eps(eltype(p)))
                throw(ArgumentError("vectors number $i and $j are not orthonormal (inner product = $dot_val)"))
            end
        end
    end
    return B
end

function get_coordinates(M::ArrayManifold, p, X, B::AbstractBasis; kwargs...)
    is_tangent_vector(M, p, X, true; kwargs...)
    return get_coordinates(M.manifold, p, X, B)
end

function get_vector(M::ArrayManifold, p, X, B::AbstractBasis; kwargs...)
    is_manifold_point(M, p, true; kwargs...)
    size(X) == (manifold_dimension(M),) || error("Incorrect size of vector X")
    return get_vector(M.manifold, p, X, B)
end

function _euclidean_basis_vector(p, i)
    X = zero(p)
    X[i] = 1
    return X
end

function get_basis(M::Manifold, p, B::ProjectedOrthonormalBasis{:svd,ℝ})
    S = representation_size(M)
    PS = prod(S)
    dim = manifold_dimension(M)
    # projection
    # TODO: find a better way to obtain a basis of the ambient space
    Xs = [
        convert(Vector, reshape(project_tangent(M, p, _euclidean_basis_vector(p, i)), PS))
        for i in eachindex(p)
    ]
    O = reduce(hcat, Xs)
    # orthogonalization
    # TODO: try using rank-revealing QR here
    decomp = svd(O)
    rotated = Diagonal(decomp.S) * decomp.Vt
    vecs = [collect(reshape(rotated[i, :], S)) for i = 1:dim]
    # normalization
    for i = 1:dim
        i_norm = norm(M, p, vecs[i])
        vecs[i] /= i_norm
    end
    return PrecomputedOrthonormalBasis(vecs)
end

"""
    get_vectors(M::Manifold, p, B::AbstractBasis)

Get the basis vectors of basis `B` of the tangent space at point `p`.
"""
function get_vectors(M::Manifold, p, B::AbstractBasis)
    error("get_vectors not implemented for manifold of type $(typeof(M)) a point of type $(typeof(p)) and basis of type $(typeof(B)).")
end

get_vectors(::Manifold, p, B::PrecomputedOrthonormalBasis) = B.vectors
get_vectors(::Manifold, p, B::PrecomputedDiagonalizingOrthonormalBasis) = B.vectors

# related to DefaultManifold; to be moved to ManifoldsBase.jl in the future
function get_coordinates(M::DefaultManifold, p, X, ::ArbitraryOrthonormalBasis)
    return reshape(X, manifold_dimension(M))
end

function get_vector(M::DefaultManifold, p, X, ::ArbitraryOrthonormalBasis)
    return reshape(X, representation_size(M))
end

function get_basis(M::DefaultManifold, p, ::ArbitraryOrthonormalBasis)
    return PrecomputedOrthonormalBasis([
        _euclidean_basis_vector(p, i) for i in eachindex(p)
    ])
end

function get_basis(M::Manifold, p, B::ProjectedOrthonormalBasis{:gram_schmidt,ℝ}; kwargs...)
    E = [_euclidean_basis_vector(p, i) for i in eachindex(p)]
    N = length(E)
    Ξ = empty(E)
    dim = manifold_dimension(M)
    N < dim && @warn "Input only has $(N) vectors, but manifold dimension is $(dim)."
    K = 0
    @inbounds for n = 1:N
        Ξₙ = project_tangent(M, p, E[n])
        for k = 1:K
            Ξₙ .-= real(inner(M, p, Ξ[k], Ξₙ)) .* Ξ[k]
        end
        nrmΞₙ = norm(M, p, Ξₙ)
        if nrmΞₙ == 0
            @warn "Input vector $(n) has length 0."
            @goto skip
        end
        Ξₙ ./= nrmΞₙ
        for k = 1:K
            if !isapprox(real(inner(M, p, Ξ[k], Ξₙ)), 0; kwargs...)
                @warn "Input vector $(n) is not linearly independent of output basis vector $(k)."
                @goto skip
            end
        end
        push!(Ξ, Ξₙ)
        K += 1
        K * real_dimension(number_system(B)) == dim && return PrecomputedOrthonormalBasis(Ξ)
        @label skip
    end
    @warn "get_basis with bases $(typeof(B)) only found $(K) orthonormal basis vectors, but manifold dimension is $(dim)."
    return PrecomputedOrthonormalBasis(Ξ)
end

function _show_basis_vector(io::IO, X; pre = "", head = "")
    sX = sprint(show, "text/plain", X, context = io, sizehint = 0)
    sX = replace(sX, '\n' => "\n$(pre)")
    print(io, head, pre, sX)
end

function _show_basis_vector_range(io::IO, Ξ, range; pre = "", sym = "E")
    for i in range
        _show_basis_vector(io, Ξ[i]; pre = pre, head = "\n$(sym)$(i) =\n")
    end
    return nothing
end

function _show_basis_vector_range_noheader(io::IO, Ξ; max_vectors = 4, pre = "", sym = "E")
    nv = length(Ξ)
    if nv ≤ max_vectors
        _show_basis_vector_range(io, Ξ, 1:nv; pre = "  ", sym = " E")
    else
        halfn = div(max_vectors, 2)
        _show_basis_vector_range(io, Ξ, 1:halfn; pre = "  ", sym = " E")
        print(io, "\n ⋮")
        _show_basis_vector_range(io, Ξ, (nv-halfn+1):nv; pre = "  ", sym = " E")
    end
end

function show(io::IO, ::ArbitraryOrthonormalBasis{𝔽}) where {𝔽}
    print(io, "ArbitraryOrthonormalBasis($(𝔽))")
end
function show(io::IO, ::ProjectedOrthonormalBasis{method,𝔽}) where {method,𝔽}
    print(io, "ProjectedOrthonormalBasis($(repr(method)), $(𝔽))")
end
function show(io::IO, mime::MIME"text/plain", onb::DiagonalizingOrthonormalBasis)
    println(
        io,
        "DiagonalizingOrthonormalBasis with coordinates in $(number_system(onb)) and eigenvalue 0 in direction:",
    )
    sk = sprint(show, "text/plain", onb.frame_direction, context = io, sizehint = 0)
    sk = replace(sk, '\n' => "\n ")
    print(io, sk)
end
function show(io::IO, mime::MIME"text/plain", onb::PrecomputedOrthonormalBasis)
    nv = length(onb.vectors)
    print(
        io,
        "PrecomputedOrthonormalBasis with coordinates in $(number_system(onb)) and $(nv) basis vector$(nv == 1 ? "" : "s"):",
    )
    _show_basis_vector_range_noheader(
        io,
        onb.vectors;
        max_vectors = 4,
        pre = "  ",
        sym = " E",
    )
end
function show(io::IO, mime::MIME"text/plain", onb::PrecomputedDiagonalizingOrthonormalBasis)
    nv = length(onb.vectors)
    println(
        io,
        "PrecomputedDiagonalizingOrthonormalBasis with coordinates in $(number_system(onb)) and $(nv) basis vector$(nv == 1 ? "" : "s")",
    )
    print(io, "Basis vectors:")
    _show_basis_vector_range_noheader(
        io,
        onb.vectors;
        max_vectors = 4,
        pre = "  ",
        sym = " E",
    )
    println(io, "\nEigenvalues:")
    sk = sprint(show, "text/plain", onb.kappas, context = io, sizehint = 0)
    sk = replace(sk, '\n' => "\n ")
    print(io, ' ', sk)
end
