"""
    AbstractBasis{F}

Abstract type that represents a basis on a manifold or a subset of it.

The type parameter `F` denotes the [`AbstractNumbers`](@ref) that will be used as scalars.
"""
abstract type AbstractBasis{F} end

"""
    number_system(::AbstractBasis)

The number system used as scalars in the given basis.
"""
number_system(::AbstractBasis{F}) where {F} = F

"""
    AbstractOrthonormalBasis{F}

Abstract type that represents an orthonormal basis on a manifold or a subset of it.

The type parameter `F` denotes the [`AbstractNumbers`](@ref) that will be used as scalars.
"""
abstract type AbstractOrthonormalBasis{F} <: AbstractBasis{F} end

"""
    AbstractPrecomputedOrthonormalBasis{F}

Abstract type that represents an orthonormal basis of the tangent space at a point
on a manifold. Tangent vectors can be obtained using function [`get_vectors`](@ref).

The vectors are not always fully precomputed because a partially precomputed
basis may be enough for implementing [`get_vector`](@ref) and [`get_coordinates`](@ref).

The type parameter `F` denotes the [`AbstractNumbers`](@ref) that will be used as scalars.
"""
abstract type AbstractPrecomputedOrthonormalBasis{F} <: AbstractOrthonormalBasis{F} end

"""
    ArbitraryOrthonormalBasis(F::AbstractNumbers = ℝ)

An arbitrary orthonormal basis on a manifold. This will usually
be the fastest orthonormal basis available for a manifold.

The type parameter `F` denotes the [`AbstractNumbers`](@ref) that will be used as scalars.
"""
struct ArbitraryOrthonormalBasis{F} <: AbstractOrthonormalBasis{F} end

ArbitraryOrthonormalBasis(F::AbstractNumbers = ℝ) = ArbitraryOrthonormalBasis{F}()

"""
    ProjectedOrthonormalBasis(method::Symbol, F::AbstractNumbers = ℝ)

An orthonormal basis that comes from orthonormalization of basis vectors
of the ambient space projected onto the subspace representing the tangent space
at a given point.

The type parameter `F` denotes the [`AbstractNumbers`](@ref) that will be used as scalars.

Available methods:
  - `:gram_schmidt` uses a modified Gram-Schmidt orthonormalization.
  - `:svd` uses SVD decomposition to orthogonalize projected vectors.
    The SVD-based method should be more numerically stable at the cost of
    an additional assumption (local metric tensor at a point where the
    basis is calculated has to be diagonal).
"""
struct ProjectedOrthonormalBasis{Method,F} <: AbstractOrthonormalBasis{F} end

function ProjectedOrthonormalBasis(method::Symbol, F::AbstractNumbers = ℝ)
    return ProjectedOrthonormalBasis{method,F}()
end

@doc doc"""
    DiagonalizingOrthonormalBasis(v, F::AbstractNumbers = ℝ)

An orthonormal basis `Ξ` as a vector of tangent vectors (of length determined by
[`manifold_dimension`](@ref)) in the tangent space that diagonalizes the curvature
tensor $R(u,v)w$ and where the direction `v` has curvature `0`.

The type parameter `F` denotes the [`AbstractNumbers`](@ref) that will be used as scalars.
"""
struct DiagonalizingOrthonormalBasis{TV,F} <: AbstractOrthonormalBasis{F}
    v::TV
end

function DiagonalizingOrthonormalBasis(v, F::AbstractNumbers = ℝ)
    return DiagonalizingOrthonormalBasis{typeof(v),F}(v)
end

const ArbitraryOrDiagonalizingBasis =
    Union{ArbitraryOrthonormalBasis,DiagonalizingOrthonormalBasis}

"""
    PrecomputedOrthonormalBasis(vectors::AbstractVector, F::AbstractNumbers = ℝ)

A precomputed orthonormal basis at a point on a manifold.

The type parameter `F` denotes the [`AbstractNumbers`](@ref) that will be used as scalars.
"""
struct PrecomputedOrthonormalBasis{TV<:AbstractVector,F} <:
       AbstractPrecomputedOrthonormalBasis{F}
    vectors::TV
end

function PrecomputedOrthonormalBasis(vectors::AbstractVector, F::AbstractNumbers = ℝ)
    return PrecomputedOrthonormalBasis{typeof(vectors),F}(vectors)
end

@doc doc"""
    DiagonalizingOrthonormalBasis(vectors, kappas, F::AbstractNumbers = ℝ)

A precomputed orthonormal basis `Ξ` as a vector of tangent vectors (of length determined
by [`manifold_dimension`](@ref)) in the tangent space that diagonalizes the curvature
tensor $R(u,v)w$ with eigenvalues `kappas` and where the direction `v` has curvature `0`.

The type parameter `F` denotes the [`AbstractNumbers`](@ref) that will be used as scalars.
"""
struct PrecomputedDiagonalizingOrthonormalBasis{TV<:AbstractVector,TK<:AbstractVector,F} <:
       AbstractPrecomputedOrthonormalBasis{F}
    vectors::TV
    kappas::TK
end

function PrecomputedDiagonalizingOrthonormalBasis(
    vectors::AbstractVector,
    kappas::AbstractVector,
    F::AbstractNumbers = ℝ,
)
    return PrecomputedDiagonalizingOrthonormalBasis{typeof(vectors),typeof(kappas),F}(
        vectors,
        kappas,
    )
end

"""
    get_coordinates(M::Manifold, x, v, B::AbstractBasis)

Compute a one-dimensional vector of coefficients of the tangent vector `v`
at point denoted by `x` on manifold `M` in basis `B`.

Depending on the basis, `x` may not directly represent a point on the manifold.
For example if a basis transported along a curve is used, `x` may be the coordinate
along the curve.

See also: [`get_vector`](@ref), [`get_basis`](@ref)
"""
function get_coordinates(M::Manifold, x, v, B::AbstractBasis)
    error("get_coordinates not implemented for manifold of type $(typeof(M)) a point of type $(typeof(x)), tangent vector of type $(typeof(v)) and basis of type $(typeof(B)).")
end
function get_coordinates(M::Manifold, x, v, B::AbstractPrecomputedOrthonormalBasis{ℝ})
    return map(vb -> real(inner(M, x, v, vb)), get_vectors(M, x, B))
end
function get_coordinates(M::Manifold, x, v, B::AbstractPrecomputedOrthonormalBasis)
    return map(vb -> inner(M, x, v, vb), get_vectors(M, x, B))
end

"""
    get_vector(M::Manifold, x, v, B::AbstractBasis)

Convert a one-dimensional vector of coefficients in a basis `B` of
the tangent space at `x` on manifold `M` to a tangent vector `v` at `x`.

Depending on the basis, `x` may not directly represent a point on the manifold.
For example if a basis transported along a curve is used, `x` may be the coordinate
along the curve.

See also: [`get_coordinates`](@ref), [`get_basis`](@ref)
"""
function get_vector(M::Manifold, x, v, B::AbstractBasis)
    error("get_vector not implemented for manifold of type $(typeof(M)) a point of type $(typeof(x)), tangent vector of type $(typeof(v)) and basis of type $(typeof(B)).")
end
function get_vector(M::Manifold, x, v, B::AbstractPrecomputedOrthonormalBasis)
    # quite convoluted but:
    #  1) preserves the correct `eltype`
    #  2) guarantees a reasonable array type `vout`
    #     (for example scalar * `SizedArray` is an `SArray`)
    bvectors = get_vectors(M, x, B)
    if isa(bvectors[1], ProductRepr)
        vt = v[1] * bvectors[1]
        vout = allocate(bvectors[1], eltype(vt))
        copyto!(vout, vt)
        for i = 2:length(v)
            vout += v[i] * bvectors[i]
        end
        return vout
    else
        vt = v[1] .* bvectors[1]
        vout = allocate(bvectors[1], eltype(vt))
        copyto!(vout, vt)
        for i = 2:length(v)
            vout .+= v[i] .* bvectors[i]
        end
        return vout
    end
end

"""
    get_basis(M::Manifold, x, B::AbstractBasis) -> AbstractBasis

Compute the basis vectors of the tangent space at a point on manifold `M`
represented by `x`.

Returned object derives from [`AbstractBasis`](@ref) and may have a field `.vectors`
that stores tangent vectors or it may store them implicitly, in which case
the function [`get_vectors`](@ref) needs to be used to retrieve the basis vectors.

See also: [`get_coordinates`](@ref), [`get_vector`](@ref)
"""
function get_basis(M::Manifold, x, B::AbstractBasis)
    error("get_basis not implemented for manifold of type $(typeof(M)) a point of type $(typeof(x)) and basis of type $(typeof(B)).")
end
"""
    get_basis(M::Manifold, x, B::ArbitraryOrthonormalBasis)

Compute the basis vectors of an [`ArbitraryOrthonormalBasis`](@ref).
"""
function get_basis(M::Manifold, x, B::ArbitraryOrthonormalBasis)
    dim = manifold_dimension(M)
    return PrecomputedOrthonormalBasis([
        get_vector(M, x, [ifelse(i == j, 1, 0) for j = 1:dim], B) for i = 1:dim
    ])
end
get_basis(M::Manifold, x, B::AbstractPrecomputedOrthonormalBasis) = B
function get_basis(M::ArrayManifold, x, B::AbstractPrecomputedOrthonormalBasis{ℝ})
    bvectors = get_vectors(M, x, B)
    N = length(bvectors)
    M_dim = manifold_dimension(M)
    if N != M_dim
        throw(ArgumentError("Incorrect number of basis vectors; expected: $M_dim, given: $N"))
    end
    for i = 1:N
        vi_norm = norm(M, x, bvectors[i])
        if !isapprox(vi_norm, 1)
            throw(ArgumentError("vector number $i is not normalized (norm = $vi_norm)"))
        end
        for j = i+1:N
            dot_val = real(inner(M, x, bvectors[i], bvectors[j]))
            if !isapprox(dot_val, 0; atol = eps(eltype(x)))
                throw(ArgumentError("vectors number $i and $j are not orthonormal (inner product = $dot_val)"))
            end
        end
    end
    return B
end

function get_coordinates(M::ArrayManifold, x, v, B::AbstractBasis; kwargs...)
    is_tangent_vector(M, x, v, true; kwargs...)
    return get_coordinates(M.manifold, x, v, B)
end

function get_vector(M::ArrayManifold, x, v, B::AbstractBasis; kwargs...)
    is_manifold_point(M, x, true; kwargs...)
    size(v) == (manifold_dimension(M),) || error("Incorrect size of vector v")
    return get_vector(M.manifold, x, v, B)
end

function _euclidean_basis_vector(x, i)
    y = zero(x)
    y[i] = 1
    return y
end

function get_basis(M::Manifold, x, B::ProjectedOrthonormalBasis{:svd,ℝ})
    S = representation_size(M)
    PS = prod(S)
    dim = manifold_dimension(M)
    # projection
    # TODO: find a better way to obtain a basis of the ambient space
    vs = [
        convert(Vector, reshape(project_tangent(M, x, _euclidean_basis_vector(x, i)), PS))
        for i in eachindex(x)
    ]
    O = reduce(hcat, vs)
    # orthogonalization
    # TODO: try using rank-revealing QR here
    decomp = svd(O)
    rotated = Diagonal(decomp.S) * decomp.Vt
    vecs = [collect(reshape(rotated[i, :], S)) for i = 1:dim]
    # normalization
    for i = 1:dim
        i_norm = norm(M, x, vecs[i])
        vecs[i] /= i_norm
    end
    return PrecomputedOrthonormalBasis(vecs)
end

"""
    get_vectors(M::Manifold, x, B::AbstractBasis)

Get the basis vectors of basis `B` of the tangent space at point `x`.
"""
function get_vectors(M::Manifold, x, B::AbstractBasis)
    error("get_vectors not implemented for manifold of type $(typeof(M)) a point of type $(typeof(x)) and basis of type $(typeof(B)).")
end

get_vectors(::Manifold, x, B::PrecomputedOrthonormalBasis) = B.vectors
get_vectors(::Manifold, x, B::PrecomputedDiagonalizingOrthonormalBasis) = B.vectors

# related to DefaultManifold; to be moved to ManifoldsBase.jl in the future
function get_coordinates(M::DefaultManifold, x, v, ::ArbitraryOrthonormalBasis)
    return reshape(v, manifold_dimension(M))
end

function get_vector(M::DefaultManifold, x, v, ::ArbitraryOrthonormalBasis)
    return reshape(v, representation_size(M))
end

function get_basis(M::DefaultManifold, x, ::ArbitraryOrthonormalBasis)
    return PrecomputedOrthonormalBasis([
        _euclidean_basis_vector(x, i) for i in eachindex(x)
    ])
end

function get_basis(M::Manifold, x, B::ProjectedOrthonormalBasis{:gram_schmidt,ℝ}; kwargs...)
    E = [_euclidean_basis_vector(x, i) for i in eachindex(x)]
    N = length(E)
    Ξ = empty(E)
    dim = manifold_dimension(M)
    N < dim && @warn "Input only has $(N) vectors, but manifold dimension is $(dim)."
    K = 0
    @inbounds for n = 1:N
        Ξₙ = project_tangent(M, x, E[n])
        for k = 1:K
            Ξₙ .-= real(inner(M, x, Ξ[k], Ξₙ)) .* Ξ[k]
        end
        nrmΞₙ = norm(M, x, Ξₙ)
        if nrmΞₙ == 0
            @warn "Input vector $(n) has length 0."
            @goto skip
        end
        Ξₙ ./= nrmΞₙ
        for k = 1:K
            if !isapprox(real(inner(M, x, Ξ[k], Ξₙ)), 0; kwargs...)
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

function _show_basis_vector(io::IO, v; pre = "", head = "")
    sx = sprint(show, "text/plain", v, context = io, sizehint = 0)
    sx = replace(sx, '\n' => "\n$(pre)")
    print(io, head, pre, sx)
end

function _show_basis_vector_range(io::IO, vs, range; pre = "", sym = "E")
    for i in range
        _show_basis_vector(io, vs[i]; pre = pre, head = "\n$(sym)$(i) =\n")
    end
    return nothing
end

function show(io::IO, ::ArbitraryOrthonormalBasis{F}) where {F}
    print(io, "ArbitraryOrthonormalBasis($(F))")
end
function show(io::IO, ::ProjectedOrthonormalBasis{method,F}) where {method,F}
    print(io, "ProjectedOrthonormalBasis($(repr(method)), $(F))")
end
function show(io::IO, mime::MIME"text/plain", onb::DiagonalizingOrthonormalBasis)
    println(
        io,
        "DiagonalizingOrthonormalBasis with coordinates in $(number_system(onb)) and 0 curvature in direction:",
    )
    sk = sprint(show, "text/plain", onb.v, context = io, sizehint = 0)
    sk = replace(sk, '\n' => "\n ")
    print(io, sk)
end
function show(io::IO, mime::MIME"text/plain", onb::PrecomputedOrthonormalBasis)
    nv = length(onb.vectors)
    print(
        io,
        "PrecomputedOrthonormalBasis with coordinates in $(number_system(onb)) and $(nv) basis vector$(nv == 1 ? "" : "s"):",
    )
    if nv ≤ 4
        _show_basis_vector_range(io, onb.vectors, 1:nv; pre = "  ", sym = " E")
    else
        _show_basis_vector_range(io, onb.vectors, 1:2; pre = "  ", sym = " E")
        print(io, "\n ⋮")
        _show_basis_vector_range(io, onb.vectors, (nv-1):nv; pre = "  ", sym = " E")
    end
    return nothing
end
function show(io::IO, mime::MIME"text/plain", onb::PrecomputedDiagonalizingOrthonormalBasis)
    nv = length(onb.vectors)
    println(
        io,
        "PrecomputedDiagonalizingOrthonormalBasis with coordinates in $(number_system(onb)) and $(nv) basis vector$(nv == 1 ? "" : "s")",
    )
    print(io, "Basis vectors:")
    if nv ≤ 4
        _show_basis_vector_range(io, onb.vectors, 1:nv; pre = "  ", sym = " E")
    else
        _show_basis_vector_range(io, onb.vectors, 1:2; pre = "  ", sym = " E")
        print(io, "\n ⋮")
        _show_basis_vector_range(io, onb.vectors, (nv-1):nv; pre = "  ", sym = " E")
    end
    println(io, "\nEigenvalues:")
    sk = sprint(show, "text/plain", onb.kappas, context = io, sizehint = 0)
    sk = replace(sk, '\n' => "\n ")
    print(io, ' ', sk)
end
