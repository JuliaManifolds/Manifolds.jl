"""
    AbstractBasis{𝔽}

Abstract type that represents a basis on a manifold or a subset of it.

The type parameter `𝔽` denotes the [`AbstractNumbers`](@ref) that will be used as scalars.
"""
abstract type AbstractBasis{𝔽} end

"""
    DefaultBasis{𝔽}

An arbitrary basis on a manifold. This will usually
be the fastest basis available for a manifold.

The type parameter `𝔽` denotes the [`AbstractNumbers`](@ref) that will be used as scalars.
"""
struct DefaultBasis{𝔽} <: AbstractBasis{𝔽} end
DefaultBasis(𝔽::AbstractNumbers = ℝ) = DefaultBasis{𝔽}()

"""
    DefaultOrthogonalBasis{𝔽}

An arbitrary orthogonal basis on a manifold. This will usually
be the fastest orthogonal basis available for a manifold.

The type parameter `𝔽` denotes the [`AbstractNumbers`](@ref) that will be used as scalars.
"""
struct DefaultOrthogonalBasis{𝔽} <: AbstractBasis{𝔽} end
DefaultOrthogonalBasis(𝔽::AbstractNumbers = ℝ) = DefaultOrthogonalBasis{𝔽}()

"""
    AbstractOrthonormalBasis{𝔽}

Abstract type that represents an orthonormal basis on a manifold or a subset of it.

The type parameter `𝔽` denotes the [`AbstractNumbers`](@ref) that will be used as scalars.
"""
abstract type AbstractOrthonormalBasis{𝔽} <: AbstractBasis{𝔽} end

"""
    DefaultOrthonormalBasis(𝔽::AbstractNumbers = ℝ)

An arbitrary orthonormal basis on a manifold. This will usually
be the fastest orthonormal basis available for a manifold.

The type parameter `𝔽` denotes the [`AbstractNumbers`](@ref) that will be used as
scalars.
"""
struct DefaultOrthonormalBasis{𝔽} <: AbstractOrthonormalBasis{𝔽} end

DefaultOrthonormalBasis(𝔽::AbstractNumbers = ℝ) = DefaultOrthonormalBasis{𝔽}()

"""
    ProjectedOrthonormalBasis(method::Symbol, 𝔽::AbstractNumbers = ℝ)

An orthonormal basis that comes from orthonormalization of basis vectors
of the ambient space projected onto the subspace representing the tangent space
at a given point.

The type parameter `𝔽` denotes the [`AbstractNumbers`](@ref) that will be used as
scalars.

Available methods:
  - `:gram_schmidt` uses a modified Gram-Schmidt orthonormalization.
  - `:svd` uses SVD decomposition to orthogonalize projected vectors.
    The SVD-based method should be more numerically stable at the cost of
    an additional assumption (local metric tensor at a point where the
    basis is calculated has to be diagonal).
"""
struct ProjectedOrthonormalBasis{Method,𝔽} <: AbstractOrthonormalBasis{𝔽} end

function ProjectedOrthonormalBasis(method::Symbol, 𝔽::AbstractNumbers = ℝ)
    return ProjectedOrthonormalBasis{method,𝔽}()
end

@doc raw"""
    DiagonalizingOrthonormalBasis(frame_direction, 𝔽::AbstractNumbers = ℝ)

An orthonormal basis `Ξ` as a vector of tangent vectors (of length determined by
[`manifold_dimension`](@ref)) in the tangent space that diagonalizes the curvature
tensor $R(u,v)w$ and where the direction `frame_direction` $v$ has curvature `0`.

The type parameter `𝔽` denotes the [`AbstractNumbers`](@ref) that will be used as
scalars.
"""
struct DiagonalizingOrthonormalBasis{TV,𝔽} <: AbstractOrthonormalBasis{𝔽}
    frame_direction::TV
end
function DiagonalizingOrthonormalBasis(X, 𝔽::AbstractNumbers = ℝ)
    return DiagonalizingOrthonormalBasis{typeof(X),𝔽}(X)
end
struct DiagonalizingBasisData{D,V,ET}
    frame_direction::D
    eigenvalues::ET
    vectors::V
end

const DefaultOrDiagonalizingBasis =
    Union{DefaultOrthonormalBasis,DiagonalizingOrthonormalBasis}


struct CachedBasis{B,V,𝔽} <: AbstractBasis{𝔽} where {BT<:AbstractBasis,V}
    data::V
end
function CachedBasis(basis::B, data::V, 𝔽::AbstractNumbers = ℝ) where {V,B<:AbstractBasis}
    return CachedBasis{B,V,𝔽}(data)
end
function CachedBasis(basis::CachedBasis) # avoid double encapsulation
    return basis
end
function CachedBasis(
    basis::DiagonalizingOrthonormalBasis,
    eigenvalues::ET,
    vectors::T,
    𝔽::AbstractNumbers = ℝ,
) where {ET<:AbstractVector,T<:AbstractVector}
    data = DiagonalizingBasisData(basis.frame_direction, eigenvalues, vectors)
    return CachedBasis(basis, data, 𝔽)
end

# forward declarations
function get_coordinates end
function get_vector end

const all_uncached_bases = Union{AbstractBasis, DefaultBasis, DefaultOrthogonalBasis, DefaultOrthonormalBasis}

function allocate_result(M::Manifold, f::typeof(get_coordinates), p, X)
    T = allocate_result_type(M, f, (p, X))
    return allocate(p, T, Size(manifold_dimension(M)))
end

@inline function allocate_result_type(
    M::Manifold,
    f::Union{typeof(get_coordinates), typeof(get_vector)},
    args::Tuple,
)
    apf = allocation_promotion_function(M, f, args)
    return apf(invoke(allocate_result_type, Tuple{Manifold,Any,typeof(args)}, M, f, args))
end

"""
    allocation_promotion_function(M::Manifold, f, args::Tuple)

Determine the function that must be used to ensure that the allocated representation is of
the right type. This is needed for [`get_vector`](@ref) when a point on a complex manifold
is represented by a real-valued vectors with a real-coefficient basis, so that
a complex-valued vector representation is allocated.
"""
allocation_promotion_function(M::Manifold, f, args::Tuple) = identity

function combine_allocation_promotion_functions(f::T, ::T) where {T}
    return f
end
function combine_allocation_promotion_functions(::typeof(complex), ::typeof(identity))
    return complex
end
function combine_allocation_promotion_functions(::typeof(identity), ::typeof(complex))
    return complex
end

function _euclidean_basis_vector(p, i)
    X = zero(p)
    X[i] = 1
    return X
end

"""
    get_basis(M::Manifold, p, B::AbstractBasis) -> CachedBasis

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

function get_basis(M::Manifold, p, B::DefaultOrthonormalBasis)
    dim = manifold_dimension(M)
    return CachedBasis(
        B,
        [get_vector(M, p, [ifelse(i == j, 1, 0) for j = 1:dim], B) for i = 1:dim],
    )
end
function get_basis(M::Manifold, p, B::CachedBasis)
    return B
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
    return CachedBasis(B, vecs)
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
        K * real_dimension(number_system(B)) == dim && return CachedBasis(B, Ξ, ℝ)
        @label skip
    end
    @warn "get_basis with bases $(typeof(B)) only found $(K) orthonormal basis vectors, but manifold dimension is $(dim)."
    return CachedBasis(B, Ξ)
end

"""
    get_coordinates(M::Manifold, p, X, B::AbstractBasis)
    get_coordinates(M::Manifold, p, X, B::CachedBasis)

Compute a one-dimensional vector of coefficients of the tangent vector `X`
at point denoted by `p` on manifold `M` in basis `B`.

Depending on the basis, `p` may not directly represent a point on the manifold.
For example if a basis transported along a curve is used, `p` may be the coordinate
along the curve. If a [`CachedBasis`](@ref) is provided, their stored vectors are used,
otherwise the user has to provide a method to compute the coordinates.

For the [`CachedBasis`](@ref) keep in mind that the reconstruction with [`get_vector`](@ref)
requires either a dual basis or the cached basis to be selfdual, for example orthonormal

See also: [`get_vector`](@ref), [`get_basis`](@ref)
"""
function get_coordinates(M::Manifold, p, X, B::AbstractBasis)
    Y = allocate_result(M, get_coordinates, p, X)
    return get_coordinates!(M, Y, p, X, B)
end

function get_coordinates!(M::Manifold, Y, p, X, B::AbstractBasis)
    error("get_coordinates! not implemented for manifold of type $(typeof(M)) coordinates of type $(typeof(Y)), a point of type $(typeof(p)), tangent vector of type $(typeof(X)) and basis of type $(typeof(B)).")
end
function get_coordinates!(M::Manifold, Y, p, X, B::DefaultBasis)
    return get_coordinates!(M, Y, p, X, DefaultOrthogonalBasis(number_system(B)))
end
function get_coordinates!(M::Manifold, Y, p, X, B::DefaultOrthogonalBasis)
    return get_coordinates!(M, Y, p, X, DefaultOrthonormalBasis(number_system(B)))
end
function get_coordinates!(
    M::Manifold,
    Y,
    p,
    X,
    B::CachedBasis{BT},
) where {BT<:AbstractBasis{ℝ}}
    map!(vb -> real(inner(M, p, X, vb)), Y, get_vectors(M, p, B))
    return Y
end
function get_coordinates!(M::Manifold, Y, p, X, B::CachedBasis)
    map!(vb -> inner(M, p, X, vb), Y, get_vectors(M, p, B))
    return Y
end


"""
    get_vector(M::Manifold, p, X, B::AbstractBasis)

Convert a one-dimensional vector of coefficients in a basis `B` of
the tangent space at `p` on manifold `M` to a tangent vector `X` at `p`.

Depending on the basis, `p` may not directly represent a point on the manifold.
For example if a basis transported along a curve is used, `p` may be the coordinate
along the curve.

For the [`CachedBasis`](@ref) keep in mind that the reconstruction from [`get_coordinates`](@ref)
requires either a dual basis or the cached basis to be selfdual, for example orthonormal

See also: [`get_coordinates`](@ref), [`get_basis`](@ref)
"""
function get_vector(M::Manifold, p, X, B::AbstractBasis)
    Y = allocate_result(M, get_vector, p, X)
    return get_vector!(M, Y, p, X, B)
end

function get_vector!(M::Manifold, Y, p, X, B::AbstractBasis)
    error("get_vector! not implemented for manifold of type $(typeof(M)) vector of type $(typeof(Y)), a point of type $(typeof(p)), coordinates of type $(typeof(X)) and basis of type $(typeof(B)).")
end
function get_vector!(M::Manifold, Y, p, X, B::DefaultBasis)
    return get_vector!(M, Y, p, X, DefaultOrthogonalBasis(number_system(B)))
end
function get_vector!(M::Manifold, Y, p, X, B::DefaultOrthogonalBasis)
    return get_vector!(M, Y, p, X, DefaultOrthonormalBasis(number_system(B)))
end
function get_vector!(M::Manifold, Y, p, X, B::CachedBasis)
    # quite convoluted but:
    #  1) preserves the correct `eltype`
    #  2) guarantees a reasonable array type `Y`
    #     (for example scalar * `SizedArray` is an `SArray`)
    bvectors = get_vectors(M, p, B)
    if isa(bvectors[1], ProductRepr)
        Xt = X[1] * bvectors[1]
        copyto!(Y, Xt)
        for i = 2:length(X)
            Y += X[i] * bvectors[i]
        end
        return Y
    else
        Xt = X[1] .* bvectors[1]
        copyto!(Y, Xt)
        for i = 2:length(X)
            Y .+= X[i] .* bvectors[i]
        end
        return Y
    end
end

"""
    get_vectors(M::Manifold, p, B::AbstractBasis)

Get the basis vectors of basis `B` of the tangent space at point `p`.
"""
function get_vectors(M::Manifold, p, B::AbstractBasis)
    error("get_vectors not implemented for manifold of type $(typeof(M)) a point of type $(typeof(p)) and basis of type $(typeof(B)).")
end
function get_vectors(
    M::Manifold,
    p,
    B::CachedBasis{<:AbstractBasis,<:AbstractArray},
)
    return B.data
end
function get_vectors(
    M::Manifold,
    p,
    B::CachedBasis{<:AbstractBasis,<:DiagonalizingBasisData},
)
    return B.data.vectors
end

#internal for directly cached basis i.e. those that are just arrays – used in show
_get_vectors(B::CachedBasis{<:AbstractBasis,<:AbstractArray}) = B.data
_get_vectors(B::CachedBasis{<:AbstractBasis,<:DiagonalizingBasisData}) = B.data.vectors

hat(M::Manifold, p, Xⁱ) = get_vector(M, p, Xⁱ, DefaultOrthogonalBasis())
hat!(M::Manifold, X, p, Xⁱ) = get_vector!(M, X, p, Xⁱ, DefaultOrthogonalBasis())

"""
    number_system(::AbstractBasis)

The number system used as scalars in the given basis.
"""
number_system(::AbstractBasis{𝔽}) where {𝔽} = 𝔽

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

function show(io::IO, ::DefaultOrthonormalBasis{𝔽}) where {𝔽}
    print(io, "DefaultOrthonormalBasis($(𝔽))")
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
function show(
    io::IO,
    mime::MIME"text/plain",
    B::CachedBasis{T,D,𝔽},
) where {T<:AbstractBasis,D,𝔽}
    vectors = _get_vectors(B)
    nv = length(vectors)
    print(
        io,
        "$(T()) with coordinates in $(number_system(B)) and $(nv) basis vector$(nv == 1 ? "" : "s"):",
    )
    _show_basis_vector_range_noheader(io, vectors; max_vectors = 4, pre = "  ", sym = " E")
end
function show(
    io::IO,
    mime::MIME"text/plain",
    B::CachedBasis{T,D,𝔽},
) where {T<:DiagonalizingOrthonormalBasis,D<:DiagonalizingBasisData,𝔽}
    vectors = _get_vectors(B)
    nv = length(vectors)
    sk = sprint(show, "text/plain", T(B.data.frame_direction), context = io, sizehint = 0)
    sk = replace(sk, '\n' => "\n ")
    print(io, sk)
    println(io, "\nand $(nv) basis vector$(nv == 1 ? "" : "s").")
    print(io, "Basis vectors:")
    _show_basis_vector_range_noheader(io, vectors; max_vectors = 4, pre = "  ", sym = " E")
    println(io, "\nEigenvalues:")
    sk = sprint(show, "text/plain", B.data.eigenvalues, context = io, sizehint = 0)
    sk = replace(sk, '\n' => "\n ")
    print(io, ' ', sk)
end

vee(M::Manifold, p, X) = get_coordinates(M, p, X, DefaultOrthogonalBasis())

vee!(M::Manifold, Xⁱ, p, X) = get_coordinates!(M, Xⁱ, p, X, DefaultOrthogonalBasis())


#
# Transparency
#
@decorator_transparent_signature get_coordinates(M::AbstractDecoratorManifold, p, X, B::AbstractBasis)
@decorator_transparent_signature get_coordinates(M::AbstractDecoratorManifold, p, X, B::CachedBasis)
@decorator_transparent_signature get_coordinates(M::AbstractDecoratorManifold, p, X, B::DefaultBasis)
@decorator_transparent_signature get_coordinates(M::AbstractDecoratorManifold, p, X, B::DefaultOrthogonalBasis)
@decorator_transparent_signature get_coordinates(M::AbstractDecoratorManifold, p, X, B::DefaultOrthonormalBasis)
function decorator_transparent_dispatch(::typeof(get_coordinates), ::Manifold, args...)
    return Val(:parent)
end
@decorator_transparent_signature get_coordinates!(M::AbstractDecoratorManifold, Y, p, X, B::CachedBasis)
@decorator_transparent_signature get_coordinates!(M::AbstractDecoratorManifold, Y, p, X, B::CachedBasis{BT,V,𝔽}) where {BT<:AbstractBasis{ℝ}, 𝔽, V}
@decorator_transparent_signature get_coordinates!(M::AbstractDecoratorManifold, Y, p, X, B::DefaultBasis)
@decorator_transparent_signature get_coordinates!(M::AbstractDecoratorManifold, Y, p, X, B::DefaultOrthogonalBasis)
@decorator_transparent_signature get_coordinates!(M::AbstractDecoratorManifold, Y, p, X, B::DefaultOrthonormalBasis)
function decorator_transparent_dispatch(::typeof(get_coordinates!), ::Manifold, args...)
    return Val(:parent)
end

@decorator_transparent_signature get_vector(M::AbstractDecoratorManifold, p, X, B::AbstractBasis)
@decorator_transparent_signature get_vector(M::AbstractDecoratorManifold, p, X, B::CachedBasis)
@decorator_transparent_signature get_vector(M::AbstractDecoratorManifold, p, X, B::DefaultBasis)
@decorator_transparent_signature get_vector(M::AbstractDecoratorManifold, p, X, B::DefaultOrthogonalBasis)
@decorator_transparent_signature get_vector(M::AbstractDecoratorManifold, p, X, B::DefaultOrthonormalBasis)
function decorator_transparent_dispatch(::typeof(get_vector), ::Manifold, args...)
    return Val(:parent)
end

@decorator_transparent_signature get_vector!(M::AbstractDecoratorManifold, Y, p, X, B::AbstractBasis)
@decorator_transparent_signature get_vector!(M::AbstractDecoratorManifold, Y, p, X, B::CachedBasis)
@decorator_transparent_signature get_vector!(M::AbstractDecoratorManifold, Y, p, X, B::DefaultBasis)
@decorator_transparent_signature get_vector!(M::AbstractDecoratorManifold, Y, p, X, B::DefaultOrthogonalBasis)
@decorator_transparent_signature get_vector!(M::AbstractDecoratorManifold, Y, p, X, B::DefaultOrthonormalBasis)
function decorator_transparent_dispatch(::typeof(get_vector!), ::Manifold, args...)
    return Val(:parent)
end

#
# Array Manifold
#
function get_basis(
    M::ArrayManifold,
    p,
    B::CachedBasis{<:AbstractOrthonormalBasis{ℝ},T,ℝ},
) where {T<:AbstractVector}
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
# the following is not nice, can we do better when using decorators and a specific last part?
get_coordinates(M::ArrayManifold, p, X, B::AbstractBasis; kwargs...) = _get_coordinates(M,p,X,B, kwargs...)
get_coordinates(M::ArrayManifold, p, X, B::CachedBasis; kwargs...) = _get_coordinates(M,p,X,B, kwargs...)
get_coordinates(M::ArrayManifold, p, X, B::DefaultBasis; kwargs...) = _get_coordinates(M,p,X,B, kwargs...)
get_coordinates(M::ArrayManifold, p, X, B::DefaultOrthogonalBasis; kwargs...) = _get_coordinates(M,p,X,B, kwargs...)
get_coordinates(M::ArrayManifold, p, X, B::DefaultOrthonormalBasis; kwargs...) = _get_coordinates(M,p,X,B, kwargs...)

function _get_coordinates(M::ArrayManifold, p, X, B::AbstractBasis;  kwargs...)
    is_tangent_vector(M, p, X, true; kwargs...)
    return get_coordinates(M.manifold, p, X, B)
end
function get_coordinates!(M::ArrayManifold, Y, p, X, B::all_uncached_bases; kwargs...)
    is_tangent_vector(M, p, X, true; kwargs...)
    get_coordinates!(M, Y, p, X, B)
    return Y
end
get_vector(M::ArrayManifold, p, X, B::AbstractBasis; kwargs...) = _get_vector(M,p,X,B, kwargs...)
get_vector(M::ArrayManifold, p, X, B::CachedBasis; kwargs...) = _get_vector(M,p,X,B, kwargs...)
get_vector(M::ArrayManifold, p, X, B::DefaultBasis; kwargs...) = _get_vector(M,p,X,B, kwargs...)
get_vector(M::ArrayManifold, p, X, B::DefaultOrthogonalBasis; kwargs...) = _get_vector(M,p,X,B, kwargs...)
get_vector(M::ArrayManifold, p, X, B::DefaultOrthonormalBasis; kwargs...) = _get_vector(M,p,X,B, kwargs...)

function _get_vector(M::ArrayManifold, p, X, B::AbstractBasis;  kwargs...)
    is_manifold_point(M, p, true; kwargs...)
    size(X) == (manifold_dimension(M),) || error("Incorrect size of coefficient vector X")
    Y = get_vector(M.manifold, p, X, B)
    size(Y) == representation_size(M) || error("Incorrect size of tangent vector Y")
    return Y
end
function get_vector!(M::ArrayManifold, Y, p, X, B::all_uncached_bases; kwargs...)
    is_manifold_point(M, p, true; kwargs...)
    size(X) == (manifold_dimension(M),) || error("Incorrect size of coefficient vector X")
    get_vector!(M.manifold, Y, p, X, B)
    size(Y) == representation_size(M) || error("Incorrect size of tangent vector Y")
    return Y
end


#
# DefaultManifold
#
function get_basis(M::DefaultManifold, p, B::DefaultOrthonormalBasis)
    return CachedBasis(B, [_euclidean_basis_vector(p, i) for i in eachindex(p)])
end
function get_coordinates!(M::DefaultManifold, Y, p, X, B::DefaultOrthonormalBasis)
    Y .= reshape(X, manifold_dimension(M))
    return Y
end

function get_vector!(M::DefaultManifold, Y, p, X, B::DefaultOrthonormalBasis)
    Y .= reshape(X, representation_size(M))
    return Y
end
