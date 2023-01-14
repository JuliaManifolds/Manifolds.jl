@doc raw"""
    ProductManifold{𝔽,TM<:Tuple} <: AbstractManifold{𝔽}

Product manifold $M_1 × M_2 × …  × M_n$ with product geometry.

# Constructor

    ProductManifold(M_1, M_2, ..., M_n)

generates the product manifold $M_1 × M_2 × … × M_n$.
Alternatively, the same manifold can be contructed using the `×` operator:
`M_1 × M_2 × M_3`.
"""
struct ProductManifold{𝔽,TM<:Tuple} <: AbstractDecoratorManifold{𝔽}
    manifolds::TM
end

function ProductManifold(manifolds::AbstractManifold...)
    𝔽 = ManifoldsBase._unify_number_systems((number_system.(manifolds))...)
    return ProductManifold{𝔽,typeof(manifolds)}(manifolds)
end

"""
    getindex(M::ProductManifold, i)
    M[i]

access the `i`th manifold component from the [`ProductManifold`](@ref) `M`.
"""
@inline Base.getindex(M::ProductManifold, i::Integer) = M.manifolds[i]

ProductManifold() = throw(MethodError("No method matching ProductManifold()."))

const PRODUCT_BASIS_LIST = [
    VeeOrthogonalBasis,
    DefaultBasis,
    DefaultBasis{<:Any,TangentSpaceType},
    DefaultOrthogonalBasis,
    DefaultOrthogonalBasis{<:Any,TangentSpaceType},
    DefaultOrthonormalBasis,
    DefaultOrthonormalBasis{<:Any,TangentSpaceType},
    ProjectedOrthonormalBasis{:gram_schmidt,ℝ},
    ProjectedOrthonormalBasis{:svd,ℝ},
]

"""
    ProductBasisData

A typed tuple to store tuples of data of stored/precomputed bases for a [`ProductManifold`](@ref).
"""
struct ProductBasisData{T<:Tuple}
    parts::T
end

const PRODUCT_BASIS_LIST_CACHED = [CachedBasis]

"""
    ProductMetric <: AbstractMetric

A type to represent the product of metrics for a [`ProductManifold`](@ref).
"""
struct ProductMetric <: AbstractMetric end

"""
    ProductFVectorDistribution([type::VectorBundleFibers], [x], distrs...)

Generates a random vector at point `x` from vector space (a fiber of a tangent
bundle) of type `type` using the product distribution of given distributions.

Vector space type and `x` can be automatically inferred from distributions `distrs`.
"""
struct ProductFVectorDistribution{
    TSpace<:VectorBundleFibers{<:VectorSpaceType,<:ProductManifold},
    TD<:(NTuple{N,Distribution} where {N}),
    TX,
} <: FVectorDistribution{TSpace,TX}
    type::TSpace
    x::TX
    distributions::TD
end

"""
    ProductPointDistribution(M::ProductManifold, distributions)

Product distribution on manifold `M`, combined from `distributions`.
"""
struct ProductPointDistribution{
    TM<:ProductManifold,
    TD<:(NTuple{N,Distribution} where {N}),
} <: MPointDistribution{TM}
    manifold::TM
    distributions::TD
end

"""
    ProductRetraction(retractions::AbstractRetractionMethod...)

Product retraction of `retractions`. Works on [`ProductManifold`](@ref).
"""
struct ProductRetraction{TR<:Tuple} <: AbstractRetractionMethod
    retractions::TR
end

function ProductRetraction(retractions::AbstractRetractionMethod...)
    return ProductRetraction{typeof(retractions)}(retractions)
end

"""
    InverseProductRetraction(retractions::AbstractInverseRetractionMethod...)

Product inverse retraction of `inverse retractions`. Works on [`ProductManifold`](@ref).
"""
struct InverseProductRetraction{TR<:Tuple} <: AbstractInverseRetractionMethod
    inverse_retractions::TR
end

function InverseProductRetraction(inverse_retractions::AbstractInverseRetractionMethod...)
    return InverseProductRetraction{typeof(inverse_retractions)}(inverse_retractions)
end

@inline function allocate_result(M::ProductManifold, f)
    return ProductRepr(map(N -> allocate_result(N, f), M.manifolds))
end

function allocation_promotion_function(M::ProductManifold, f, args::Tuple)
    apfs = map(MM -> allocation_promotion_function(MM, f, args), M.manifolds)
    return reduce(combine_allocation_promotion_functions, apfs)
end

"""
    ProductVectorTransport(methods::AbstractVectorTransportMethod...)

Product vector transport type of `methods`. Works on [`ProductManifold`](@ref).
"""
struct ProductVectorTransport{TR<:Tuple} <: AbstractVectorTransportMethod
    methods::TR
end

function ProductVectorTransport(methods::AbstractVectorTransportMethod...)
    return ProductVectorTransport{typeof(methods)}(methods)
end

function active_traits(f, ::ProductManifold, args...)
    return merge_traits(IsDefaultMetric(ProductMetric()))
end

function allocate_coordinates(M::AbstractManifold, p::ArrayPartition, T, n::Int)
    return allocate_coordinates(M, p.x[1], T, n)
end

"""
    change_representer(M::ProductManifold, ::AbstractMetric, p, X)

Since the metric on a product manifold decouples, the change of a representer can be done elementwise
"""
change_representer(::ProductManifold, ::AbstractMetric, ::Any, ::Any)

function change_representer!(M::ProductManifold, Y, G::AbstractMetric, p, X)
    map(
        (m, y, P, x) -> change_representer!(m, y, G, P, x),
        M.manifolds,
        submanifold_components(M, Y),
        submanifold_components(M, p),
        submanifold_components(M, X),
    )
    return Y
end

"""
    change_metric(M::ProductManifold, ::AbstractMetric, p, X)

Since the metric on a product manifold decouples, the change of metric can be done elementwise.
"""
change_metric(::ProductManifold, ::AbstractMetric, ::Any, ::Any)

function change_metric!(M::ProductManifold, Y, G::AbstractMetric, p, X)
    map(
        (m, y, P, x) -> change_metric!(m, y, G, P, x),
        M.manifolds,
        submanifold_components(M, Y),
        submanifold_components(M, p),
        submanifold_components(M, X),
    )
    return Y
end

"""
    check_point(M::ProductManifold, p; kwargs...)

Check whether `p` is a valid point on the [`ProductManifold`](@ref) `M`.
If `p` is not a point on `M` a [`CompositeManifoldError`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions.html#ManifoldsBase.CompositeManifoldError).consisting of all error messages of the
components, for which the tests fail is returned.

The tolerance for the last test can be set using the `kwargs...`.
"""
function check_point(M::ProductManifold, p::Union{ProductRepr,ArrayPartition}; kwargs...)
    ts = ziptuples(Tuple(1:length(M.manifolds)), M.manifolds, submanifold_components(M, p))
    e = [(t[1], check_point(t[2:end]...; kwargs...)) for t in ts]
    errors = filter((x) -> !(x[2] === nothing), e)
    cerr = [ComponentManifoldError(er...) for er in errors]
    (length(errors) > 1) && return CompositeManifoldError(cerr)
    (length(errors) == 1) && return cerr[1]
    return nothing
end
function check_point(M::ProductManifold, p; kwargs...)
    return DomainError(
        typeof(p),
        "The point $p is not a point on $M, since currently only ProductRepr and ArrayPartition are supported types for points on arbitrary product manifolds",
    )
end

"""
    check_size(M::ProductManifold, p; kwargs...)

Check whether `p` is of valid size on the [`ProductManifold`](@ref) `M`.
If `p` has components of wrong size a [`CompositeManifoldError`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions.html#ManifoldsBase.CompositeManifoldError).consisting of all error messages of the
components, for which the tests fail is returned.

The tolerance for the last test can be set using the `kwargs...`.
"""
function check_size(M::ProductManifold, p::Union{ProductRepr,ArrayPartition})
    ts = ziptuples(Tuple(1:length(M.manifolds)), M.manifolds, submanifold_components(M, p))
    e = [(t[1], check_size(t[2:end]...)) for t in ts]
    errors = filter((x) -> !(x[2] === nothing), e)
    cerr = [ComponentManifoldError(er...) for er in errors]
    (length(errors) > 1) && return CompositeManifoldError(cerr)
    (length(errors) == 1) && return cerr[1]
    return nothing
end
function check_size(M::ProductManifold, p; kwargs...)
    return DomainError(
        typeof(p),
        "The point $p is not a point on $M, since currently only ProductRepr and ArrayPartition are supported types for points on arbitrary product manifolds",
    )
end
function check_size(
    M::ProductManifold,
    p::Union{ProductRepr,ArrayPartition},
    X::Union{ProductRepr,ArrayPartition},
)
    ts = ziptuples(
        Tuple(1:length(M.manifolds)),
        M.manifolds,
        submanifold_components(M, p),
        submanifold_components(M, X),
    )
    e = [(t[1], check_size(t[2:end]...)) for t in ts]
    errors = filter(x -> !(x[2] === nothing), e)
    cerr = [ComponentManifoldError(er...) for er in errors]
    (length(errors) > 1) && return CompositeManifoldError(cerr)
    (length(errors) == 1) && return cerr[1]
    return nothing
end
function check_size(M::ProductManifold, p, X; kwargs...)
    return DomainError(
        typeof(X),
        "The vector $X is not a tangent vector to any tangent space on $M, since currently only ProductRepr and ArrayPartition are supported types for tangent vectors on arbitrary product manifolds",
    )
end
"""
    check_vector(M::ProductManifold, p, X; kwargs... )

Check whether `X` is a tangent vector to `p` on the [`ProductManifold`](@ref)
`M`, i.e. all projections to base manifolds must be respective tangent vectors.
If `X` is not a tangent vector to `p` on `M` a [`CompositeManifoldError`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions.html#ManifoldsBase.CompositeManifoldError).consisting
of all error messages of the components, for which the tests fail is returned.

The tolerance for the last test can be set using the `kwargs...`.
"""
function check_vector(
    M::ProductManifold,
    p::Union{ProductRepr,ArrayPartition},
    X::Union{ProductRepr,ArrayPartition};
    kwargs...,
)
    ts = ziptuples(
        Tuple(1:length(M.manifolds)),
        M.manifolds,
        submanifold_components(M, p),
        submanifold_components(M, X),
    )
    e = [(t[1], check_vector(t[2:end]...; kwargs...)) for t in ts]
    errors = filter(x -> !(x[2] === nothing), e)
    cerr = [ComponentManifoldError(er...) for er in errors]
    (length(errors) > 1) && return CompositeManifoldError(cerr)
    (length(errors) == 1) && return cerr[1]
    return nothing
end
function check_vector(M::ProductManifold, p, X; kwargs...)
    return DomainError(
        typeof(X),
        "The vector $X is not a tangent vector to any tangent space on $M, since currently only ProductRepr and ArrayPartition are supported types for tangent vectors on arbitrary product manifolds",
    )
end

for TP in [ProductRepr, ArrayPartition]
    eval(
        quote
            function copyto!(M::ProductManifold, q::$TP, p::$TP)
                map(
                    copyto!,
                    M.manifolds,
                    submanifold_components(q),
                    submanifold_components(p),
                )
                return q
            end
            function copyto!(M::ProductManifold, Y::$TP, p::$TP, X::$TP)
                map(
                    copyto!,
                    M.manifolds,
                    submanifold_components(Y),
                    submanifold_components(p),
                    submanifold_components(X),
                )
                return Y
            end
        end,
    )
end

@doc raw"""
    cross(M, N)
    cross(M1, M2, M3,...)

Return the [`ProductManifold`](@ref) For two `AbstractManifold`s `M` and `N`,
where for the case that one of them is a [`ProductManifold`](@ref) itself,
the other is either prepended (if `N` is a product) or appenden (if `M`) is.
If both are product manifold, they are combined into one product manifold,
keeping the order.

For the case that more than one is a product manifold of these is build with the
same approach as above
"""
cross(::AbstractManifold...)
LinearAlgebra.cross(M1::AbstractManifold, M2::AbstractManifold) = ProductManifold(M1, M2)
function LinearAlgebra.cross(M1::ProductManifold, M2::AbstractManifold)
    return ProductManifold(M1.manifolds..., M2)
end
function LinearAlgebra.cross(M1::AbstractManifold, M2::ProductManifold)
    return ProductManifold(M1, M2.manifolds...)
end
function LinearAlgebra.cross(M1::ProductManifold, M2::ProductManifold)
    return ProductManifold(M1.manifolds..., M2.manifolds...)
end

function default_retraction_method(M::ProductManifold)
    return ProductRetraction(map(default_retraction_method, M.manifolds)...)
end

function default_inverse_retraction_method(M::ProductManifold)
    return InverseProductRetraction(map(default_inverse_retraction_method, M.manifolds)...)
end

function default_vector_transport_method(M::ProductManifold)
    return ProductVectorTransport(map(default_vector_transport_method, M.manifolds)...)
end

@doc raw"""
    distance(M::ProductManifold, p, q)

Compute the distance between two points `p` and `q` on the [`ProductManifold`](@ref) `M`, which is
the 2-norm of the elementwise distances on the internal manifolds that build `M`.
"""
function distance(M::ProductManifold, p, q)
    return sqrt(
        sum(
            map(
                distance,
                M.manifolds,
                submanifold_components(M, p),
                submanifold_components(M, q),
            ) .^ 2,
        ),
    )
end

@doc raw"""
    exp(M::ProductManifold, p, X)

compute the exponential map from `p` in the direction of `X` on the [`ProductManifold`](@ref) `M`,
which is the elementwise exponential map on the internal manifolds that build `M`.
"""
exp(::ProductManifold, ::Any...)
function Base.exp(M::ProductManifold, p::ProductRepr, X::ProductRepr)
    return ProductRepr(
        map(
            exp,
            M.manifolds,
            submanifold_components(M, p),
            submanifold_components(M, X),
        )...,
    )
end
function Base.exp(M::ProductManifold, p::ArrayPartition, X::ArrayPartition)
    return ArrayPartition(
        map(
            exp,
            M.manifolds,
            submanifold_components(M, p),
            submanifold_components(M, X),
        )...,
    )
end

function exp!(M::ProductManifold, q, p, X)
    map(
        exp!,
        M.manifolds,
        submanifold_components(M, q),
        submanifold_components(M, p),
        submanifold_components(M, X),
    )
    return q
end

@doc raw"""
    flat(M::ProductManifold, p, X::FVector{TangentSpaceType})

use the musical isomorphism to transform the tangent vector `X` from the tangent space at
`p` on the [`ProductManifold`](@ref) `M` to a cotangent vector.
This can be done elementwise for every entry of `X` (with respect to the corresponding
entry in `p`) separately.
"""
flat(::ProductManifold, ::Any...)

function get_basis(M::ProductManifold, p, B::AbstractBasis)
    parts = map(t -> get_basis(t..., B), ziptuples(M.manifolds, submanifold_components(p)))
    return CachedBasis(B, ProductBasisData(parts))
end
function get_basis(M::ProductManifold, p, B::CachedBasis)
    return invoke(get_basis, Tuple{AbstractManifold,Any,CachedBasis}, M, p, B)
end
function get_basis(M::ProductManifold, p, B::DiagonalizingOrthonormalBasis)
    vs = map(
        ziptuples(
            M.manifolds,
            submanifold_components(p),
            submanifold_components(B.frame_direction),
        ),
    ) do t
        return get_basis(t[1], t[2], DiagonalizingOrthonormalBasis(t[3]))
    end
    return CachedBasis(B, ProductBasisData(vs))
end

"""
    get_component(M::ProductManifold, p, i)

Get the `i`th component of a point `p` on a [`ProductManifold`](@ref) `M`.
"""
function get_component(M::ProductManifold, p, i)
    return submanifold_component(M, p, i)
end

function get_coordinates(M::ProductManifold, p, X, B::AbstractBasis)
    reps = map(
        t -> get_coordinates(t..., B),
        ziptuples(M.manifolds, submanifold_components(M, p), submanifold_components(M, X)),
    )
    return vcat(reps...)
end
function get_coordinates(
    M::ProductManifold,
    p,
    X,
    B::CachedBasis{𝔽,<:AbstractBasis{𝔽},<:ProductBasisData},
) where {𝔽}
    reps = map(
        get_coordinates,
        M.manifolds,
        submanifold_components(M, p),
        submanifold_components(M, X),
        B.data.parts,
    )
    return vcat(reps...)
end

function get_coordinates!(M::ProductManifold, Xⁱ, p, X, B::AbstractBasis)
    dim = manifold_dimension(M)
    @assert length(Xⁱ) == dim
    i = one(dim)
    ts = ziptuples(M.manifolds, submanifold_components(M, p), submanifold_components(M, X))
    for t in ts
        SM = first(t)
        dim = manifold_dimension(SM)
        tXⁱ = @inbounds view(Xⁱ, i:(i + dim - 1))
        get_coordinates!(SM, tXⁱ, Base.tail(t)..., B)
        i += dim
    end
    return Xⁱ
end
function get_coordinates!(
    M::ProductManifold,
    Xⁱ,
    p,
    X,
    B::CachedBasis{𝔽,<:AbstractBasis{𝔽},<:ProductBasisData},
) where {𝔽}
    dim = manifold_dimension(M)
    @assert length(Xⁱ) == dim
    i = one(dim)
    ts = ziptuples(
        M.manifolds,
        submanifold_components(M, p),
        submanifold_components(M, X),
        B.data.parts,
    )
    for t in ts
        SM = first(t)
        dim = manifold_dimension(SM)
        tXⁱ = @inbounds view(Xⁱ, i:(i + dim - 1))
        get_coordinates!(SM, tXⁱ, Base.tail(t)...)
        i += dim
    end
    return Xⁱ
end

function _get_dim_ranges(dims::NTuple{N,Any}) where {N}
    dims_acc = accumulate(+, vcat(1, SVector(dims)))
    return ntuple(i -> (dims_acc[i]:(dims_acc[i] + dims[i] - 1)), Val(N))
end

for TP in [ProductRepr, ArrayPartition]
    eval(
        quote
            function get_vector(
                M::ProductManifold,
                p::$TP,
                Xⁱ,
                B::AbstractBasis{𝔽,TangentSpaceType},
            ) where {𝔽}
                dims = map(manifold_dimension, M.manifolds)
                @assert length(Xⁱ) == sum(dims)
                dim_ranges = _get_dim_ranges(dims)
                tXⁱ = map(dr -> (@inbounds view(Xⁱ, dr)), dim_ranges)
                ts = ziptuples(M.manifolds, submanifold_components(M, p), tXⁱ)
                return $TP(map((@inline t -> get_vector(t..., B)), ts))
            end
            function get_vector(
                M::ProductManifold,
                p::$TP,
                Xⁱ,
                B::CachedBasis{𝔽,<:AbstractBasis{𝔽},<:ProductBasisData},
            ) where {𝔽}
                dims = map(manifold_dimension, M.manifolds)
                @assert length(Xⁱ) == sum(dims)
                dim_ranges = _get_dim_ranges(dims)
                tXⁱ = map(dr -> (@inbounds view(Xⁱ, dr)), dim_ranges)
                ts =
                    ziptuples(M.manifolds, submanifold_components(M, p), tXⁱ, B.data.parts)
                return $TP(map((@inline t -> get_vector(t...)), ts))
            end
        end,
    )
end

function get_vector!(M::ProductManifold, X, p, Xⁱ, B::AbstractBasis)
    dims = map(manifold_dimension, M.manifolds)
    @assert length(Xⁱ) == sum(dims)
    dim_ranges = _get_dim_ranges(dims)
    tXⁱ = map(dr -> (@inbounds view(Xⁱ, dr)), dim_ranges)
    ts = ziptuples(
        M.manifolds,
        submanifold_components(M, X),
        submanifold_components(M, p),
        tXⁱ,
    )
    map(ts) do t
        return get_vector!(t..., B)
    end
    return X
end
function get_vector!(
    M::ProductManifold,
    X,
    p,
    Xⁱ,
    B::CachedBasis{𝔽,<:AbstractBasis{𝔽},<:ProductBasisData},
) where {𝔽}
    dims = map(manifold_dimension, M.manifolds)
    @assert length(Xⁱ) == sum(dims)
    dim_ranges = _get_dim_ranges(dims)
    tXⁱ = map(dr -> (@inbounds view(Xⁱ, dr)), dim_ranges)
    ts = ziptuples(
        M.manifolds,
        submanifold_components(M, X),
        submanifold_components(M, p),
        tXⁱ,
        B.data.parts,
    )
    map(ts) do t
        return get_vector!(t...)
    end
    return X
end

function get_vectors(
    M::ProductManifold,
    p::ProductRepr,
    B::CachedBasis{𝔽,<:AbstractBasis{𝔽},<:ProductBasisData},
) where {𝔽}
    N = number_of_components(M)
    xparts = submanifold_components(p)
    BVs = map(t -> get_vectors(t...), ziptuples(M.manifolds, xparts, B.data.parts))
    zero_tvs = map(t -> zero_vector(t...), ziptuples(M.manifolds, xparts))
    vs = typeof(ProductRepr(zero_tvs...))[]
    for i in 1:N, k in 1:length(BVs[i])
        push!(vs, ProductRepr(zero_tvs[1:(i - 1)]..., BVs[i][k], zero_tvs[(i + 1):end]...))
    end
    return vs
end
function get_vectors(
    M::ProductManifold,
    p::ArrayPartition,
    B::CachedBasis{𝔽,<:AbstractBasis{𝔽},<:ProductBasisData},
) where {𝔽}
    N = number_of_components(M)
    xparts = submanifold_components(p)
    BVs = map(t -> get_vectors(t...), ziptuples(M.manifolds, xparts, B.data.parts))
    zero_tvs = map(t -> zero_vector(t...), ziptuples(M.manifolds, xparts))
    vs = typeof(ArrayPartition(zero_tvs...))[]
    for i in 1:N, k in 1:length(BVs[i])
        push!(
            vs,
            ArrayPartition(zero_tvs[1:(i - 1)]..., BVs[i][k], zero_tvs[(i + 1):end]...),
        )
    end
    return vs
end

"""
    getindex(p, M::ProductManifold, i::Union{Integer,Colon,AbstractVector})
    p[M::ProductManifold, i]

Access the element(s) at index `i` of a point `p` on a [`ProductManifold`](@ref) `M` by
linear indexing.
See also [Array Indexing](https://docs.julialang.org/en/v1/manual/arrays/#man-array-indexing-1) in Julia.
"""
Base.@propagate_inbounds function Base.getindex(
    p::ProductRepr,
    M::ProductManifold,
    i::Union{Integer,Colon,AbstractVector,Val},
)
    return get_component(M, p, i)
end
Base.@propagate_inbounds function Base.getindex(
    p::ArrayPartition,
    M::ProductManifold,
    i::Union{Integer,Colon,AbstractVector,Val},
)
    return get_component(M, p, i)
end

@doc raw"""
    injectivity_radius(M::ProductManifold)
    injectivity_radius(M::ProductManifold, x)

Compute the injectivity radius on the [`ProductManifold`](@ref), which is the
minimum of the factor manifolds.
"""
injectivity_radius(::ProductManifold, ::Any...)
function injectivity_radius(M::ProductManifold, p)
    return min(map(injectivity_radius, M.manifolds, submanifold_components(M, p))...)
end
function injectivity_radius(M::ProductManifold, p, m::AbstractRetractionMethod)
    return min(
        map(
            (lM, lp) -> injectivity_radius(lM, lp, m),
            M.manifolds,
            submanifold_components(M, p),
        )...,
    )
end
function injectivity_radius(M::ProductManifold, p, m::ProductRetraction)
    return min(
        map(
            (lM, lp, lm) -> injectivity_radius(lM, lp, lm),
            M.manifolds,
            submanifold_components(M, p),
            m.retractions,
        )...,
    )
end
injectivity_radius(M::ProductManifold) = min(map(injectivity_radius, M.manifolds)...)
function injectivity_radius(M::ProductManifold, m::AbstractRetractionMethod)
    return min(map(manif -> injectivity_radius(manif, m), M.manifolds)...)
end
function injectivity_radius(M::ProductManifold, m::ProductRetraction)
    return min(map((lM, lm) -> injectivity_radius(lM, lm), M.manifolds, m.retractions)...)
end

@doc raw"""
    inner(M::ProductManifold, p, X, Y)

compute the inner product of two tangent vectors `X`, `Y` from the tangent space
at `p` on the [`ProductManifold`](@ref) `M`, which is just the sum of the
internal manifolds that build `M`.
"""
function inner(M::ProductManifold, p, X, Y)
    subproducts = map(
        inner,
        M.manifolds,
        submanifold_components(M, p),
        submanifold_components(M, X),
        submanifold_components(M, Y),
    )
    return sum(subproducts)
end

@doc raw"""
    inverse_retract(M::ProductManifold, p, q, m::InverseProductRetraction)

Compute the inverse retraction from `p` with respect to `q` on the [`ProductManifold`](@ref)
`M` using an [`InverseProductRetraction`](@ref), which by default encapsulates a inverse
retraction for each manifold of the product. Then this method is performed elementwise,
so the encapsulated inverse retraction methods have to be available per factor.
"""
inverse_retract(::ProductManifold, ::Any, ::Any, ::Any, ::InverseProductRetraction)

for TP in [ProductRepr, ArrayPartition]
    eval(
        quote
            function inverse_retract(
                M::ProductManifold,
                p::$TP,
                q::$TP,
                method::InverseProductRetraction,
            )
                return $TP(
                    map(
                        inverse_retract,
                        M.manifolds,
                        submanifold_components(M, p),
                        submanifold_components(M, q),
                        method.inverse_retractions,
                    ),
                )
            end
        end,
    )
end

function inverse_retract!(M::ProductManifold, Y, p, q, method::InverseProductRetraction)
    map(
        inverse_retract!,
        M.manifolds,
        submanifold_components(M, Y),
        submanifold_components(M, p),
        submanifold_components(M, q),
        method.inverse_retractions,
    )
    return Y
end

function Base.isapprox(M::ProductManifold, p, q; kwargs...)
    return all(
        t -> isapprox(t...; kwargs...),
        ziptuples(M.manifolds, submanifold_components(M, p), submanifold_components(M, q)),
    )
end
function Base.isapprox(M::ProductManifold, p, X, Y; kwargs...)
    return all(
        t -> isapprox(t...; kwargs...),
        ziptuples(
            M.manifolds,
            submanifold_components(M, p),
            submanifold_components(M, X),
            submanifold_components(M, Y),
        ),
    )
end

"""
    is_flat(::ProductManifold)

Return true if and only if all component manifolds of [`ProductManifold`](@ref) `M` are flat.
"""
function is_flat(M::ProductManifold)
    return all(is_flat, M.manifolds)
end

@doc raw"""
    log(M::ProductManifold, p, q)

Compute the logarithmic map from `p` to `q` on the [`ProductManifold`](@ref) `M`,
which can be computed using the logarithmic maps of the manifolds elementwise.
"""
log(::ProductManifold, ::Any...)

function Base.log(M::ProductManifold, p::ProductRepr, q::ProductRepr)
    return ProductRepr(
        map(
            log,
            M.manifolds,
            submanifold_components(M, p),
            submanifold_components(M, q),
        )...,
    )
end
function Base.log(M::ProductManifold, p::ArrayPartition, q::ArrayPartition)
    return ArrayPartition(
        map(
            log,
            M.manifolds,
            submanifold_components(M, p),
            submanifold_components(M, q),
        )...,
    )
end

function log!(M::ProductManifold, X, p, q)
    map(
        log!,
        M.manifolds,
        submanifold_components(M, X),
        submanifold_components(M, p),
        submanifold_components(M, q),
    )
    return X
end

@doc raw"""
    manifold_dimension(M::ProductManifold)

Return the manifold dimension of the [`ProductManifold`](@ref), which is the sum of the
manifold dimensions the product is made of.
"""
manifold_dimension(M::ProductManifold) = mapreduce(manifold_dimension, +, M.manifolds)

function mid_point!(M::ProductManifold, q, p1, p2)
    map(
        mid_point!,
        M.manifolds,
        submanifold_components(M, q),
        submanifold_components(M, p1),
        submanifold_components(M, p2),
    )
    return q
end

@doc raw"""
    norm(M::ProductManifold, p, X)

Compute the norm of `X` from the tangent space of `p` on the [`ProductManifold`](@ref),
i.e. from the element wise norms the 2-norm is computed.
"""
function LinearAlgebra.norm(M::ProductManifold, p, X)
    norms_squared = (
        map(
            norm,
            M.manifolds,
            submanifold_components(M, p),
            submanifold_components(M, X),
        ) .^ 2
    )
    return sqrt(sum(norms_squared))
end

"""
    number_of_components(M::ProductManifold{<:NTuple{N,Any}}) where {N}

Calculate the number of manifolds multiplied in the given [`ProductManifold`](@ref) `M`.
"""
number_of_components(::ProductManifold{𝔽,<:NTuple{N,Any}}) where {𝔽,N} = N

function ProductFVectorDistribution(
    type::VectorBundleFibers{<:VectorSpaceType,<:ProductManifold},
    p::Union{AbstractArray,AbstractManifoldPoint,ProductRepr},
    distributions::FVectorDistribution...,
)
    return ProductFVectorDistribution{typeof(type),typeof(distributions),typeof(p)}(
        type,
        p,
        distributions,
    )
end
function ProductFVectorDistribution(
    type::VectorBundleFibers{<:VectorSpaceType,<:ProductManifold},
    distributions::FVectorDistribution...,
)
    p = ProductRepr(map(d -> support(d).point, distributions))
    return ProductFVectorDistribution(type, p, distributions...)
end
function ProductFVectorDistribution(distributions::FVectorDistribution...)
    M = ProductManifold(map(d -> support(d).space.manifold, distributions)...)
    fiber = support(distributions[1]).space.fiber
    if !all(d -> support(d).space.fiber == fiber, distributions)
        error(
            "Not all distributions have support in vector spaces of the same type, which is currently not supported",
        )
    end
    # Probably worth considering sum spaces in the future?
    x = ProductRepr(map(d -> support(d).point, distributions)...)
    return ProductFVectorDistribution(VectorBundleFibers(fiber, M), x, distributions...)
end

function ProductPointDistribution(M::ProductManifold, distributions::MPointDistribution...)
    return ProductPointDistribution{typeof(M),typeof(distributions)}(M, distributions)
end
function ProductPointDistribution(distributions::MPointDistribution...)
    M = ProductManifold(map(d -> support(d).manifold, distributions)...)
    return ProductPointDistribution(M, distributions...)
end

for TP in [ProductRepr, ArrayPartition]
    eval(
        quote
            function parallel_transport_direction(
                M::ProductManifold,
                p::$TP,
                X::$TP,
                d::$TP,
            )
                return $TP(
                    map(
                        parallel_transport_direction,
                        M.manifolds,
                        submanifold_components(M, p),
                        submanifold_components(M, X),
                        submanifold_components(M, d),
                    ),
                )
            end
        end,
    )
end
function parallel_transport_direction!(M::ProductManifold, Y, p, X, d)
    map(
        parallel_transport_direction!,
        M.manifolds,
        submanifold_components(M, Y),
        submanifold_components(M, p),
        submanifold_components(M, X),
        submanifold_components(M, d),
    )
    return Y
end

for TP in [ProductRepr, ArrayPartition]
    eval(
        quote
            function parallel_transport_to(M::ProductManifold, p::$TP, X::$TP, q::$TP)
                return $TP(
                    map(
                        parallel_transport_to,
                        M.manifolds,
                        submanifold_components(M, p),
                        submanifold_components(M, X),
                        submanifold_components(M, q),
                    ),
                )
            end
        end,
    )
end
function parallel_transport_to!(M::ProductManifold, Y, p, X, q)
    map(
        parallel_transport_to!,
        M.manifolds,
        submanifold_components(M, Y),
        submanifold_components(M, p),
        submanifold_components(M, X),
        submanifold_components(M, q),
    )
    return Y
end

function project(M::ProductManifold, p::ProductRepr)
    return ProductRepr(map(project, M.manifolds, submanifold_components(M, p))...)
end
function project(M::ProductManifold, p::ArrayPartition)
    return ArrayPartition(map(project, M.manifolds, submanifold_components(M, p))...)
end

function project!(M::ProductManifold, q, p)
    map(project!, M.manifolds, submanifold_components(M, q), submanifold_components(M, p))
    return q
end

function project(M::ProductManifold, p::ProductRepr, X::ProductRepr)
    return ProductRepr(
        map(
            project,
            M.manifolds,
            submanifold_components(M, p),
            submanifold_components(M, X),
        )...,
    )
end
function project(M::ProductManifold, p::ArrayPartition, X::ArrayPartition)
    return ArrayPartition(
        map(
            project,
            M.manifolds,
            submanifold_components(M, p),
            submanifold_components(M, X),
        )...,
    )
end

function project!(M::ProductManifold, Y, p, X)
    map(
        project!,
        M.manifolds,
        submanifold_components(M, Y),
        submanifold_components(M, p),
        submanifold_components(M, X),
    )
    return Y
end

function Random.rand(rng::AbstractRNG, d::ProductPointDistribution)
    return ProductRepr(map(d -> rand(rng, d), d.distributions)...)
end
function Random.rand(rng::AbstractRNG, d::ProductFVectorDistribution)
    return ProductRepr(map(d -> rand(rng, d), d.distributions)...)
end

@doc raw"""
    rand(M::ProductManifold; parts_kwargs = map(_ -> (;), M.manifolds))

Return a random point on [`ProductManifold`](@ref)  `M`. `parts_kwargs` is
a tuple of keyword arguments for `rand` on each manifold in `M.manifolds`.
"""
function Random.rand(
    M::ProductManifold;
    vector_at=nothing,
    parts_kwargs=map(_ -> (;), M.manifolds),
)
    if vector_at === nothing
        return ProductRepr(
            map((N, kwargs) -> rand(N; kwargs...), M.manifolds, parts_kwargs)...,
        )
    elseif isa(vector_at, ProductRepr)
        return ProductRepr(
            map(
                (N, p, kwargs) -> rand(N; vector_at=p, kwargs...),
                M.manifolds,
                submanifold_components(M, vector_at),
                parts_kwargs,
            )...,
        )
    else
        return ArrayPartition(
            map(
                (N, p, kwargs) -> rand(N; vector_at=p, kwargs...),
                M.manifolds,
                submanifold_components(M, vector_at),
                parts_kwargs,
            )...,
        )
    end
end
function Random.rand(
    rng::AbstractRNG,
    M::ProductManifold;
    vector_at=nothing,
    parts_kwargs=map(_ -> (;), M.manifolds),
)
    if vector_at === nothing
        return ProductRepr(
            map((N, kwargs) -> rand(rng, N; kwargs...), M.manifolds, parts_kwargs)...,
        )
    elseif isa(vector_at, ProductRepr)
        return ProductRepr(
            map(
                (N, p, kwargs) -> rand(rng, N; vector_at=p, kwargs...),
                M.manifolds,
                submanifold_components(M, vector_at),
                parts_kwargs,
            )...,
        )
    else
        return ArrayPartition(
            map(
                (N, p, kwargs) -> rand(rng, N; vector_at=p, kwargs...),
                M.manifolds,
                submanifold_components(M, vector_at),
                parts_kwargs,
            )...,
        )
    end
end

function Random.rand!(
    M::ProductManifold,
    pX;
    vector_at=nothing,
    parts_kwargs=map(_ -> (;), M.manifolds),
)
    if vector_at === nothing
        map(
            (N, q, kwargs) -> rand!(N, q; kwargs...),
            M.manifolds,
            submanifold_components(M, pX),
            parts_kwargs,
        )
    else
        map(
            (N, X, p, kwargs) -> rand!(N, X; vector_at=p, kwargs...),
            M.manifolds,
            submanifold_components(M, pX),
            submanifold_components(M, vector_at),
            parts_kwargs,
        )
    end
    return pX
end
function Random.rand!(
    rng::AbstractRNG,
    M::ProductManifold,
    pX;
    vector_at=nothing,
    parts_kwargs=map(_ -> (;), M.manifolds),
)
    if vector_at === nothing
        map(
            (N, q, kwargs) -> rand!(rng, N, q; kwargs...),
            M.manifolds,
            submanifold_components(M, pX),
            parts_kwargs,
        )
    else
        map(
            (N, X, p, kwargs) -> rand!(rng, N, X; vector_at=p, kwargs...),
            M.manifolds,
            submanifold_components(M, pX),
            submanifold_components(M, vector_at),
            parts_kwargs,
        )
    end
    return pX
end

function Distributions._rand!(
    rng::AbstractRNG,
    d::ProductPointDistribution,
    x::AbstractArray{<:Number},
)
    return copyto!(x, rand(rng, d))
end
function Distributions._rand!(rng::AbstractRNG, d::ProductPointDistribution, p::ProductRepr)
    map(
        t -> Distributions._rand!(rng, t[1], t[2]),
        d.distributions,
        submanifold_components(d.manifold, p),
    )
    return p
end
function Distributions._rand!(
    rng::AbstractRNG,
    d::ProductFVectorDistribution,
    v::AbstractArray{<:Number},
)
    return copyto!(v, rand(rng, d))
end
function Distributions._rand!(
    rng::AbstractRNG,
    d::ProductFVectorDistribution,
    X::ProductRepr,
)
    map(
        t -> Distributions._rand!(rng, t[1], t[2]),
        d.distributions,
        submanifold_components(d.space.manifold, X),
    )
    return X
end

@doc raw"""
    retract(M::ProductManifold, p, X, m::ProductRetraction)

Compute the retraction from `p` with tangent vector `X` on the [`ProductManifold`](@ref) `M`
using an [`ProductRetraction`](@ref), which by default encapsulates retractions of the
base manifolds. Then this method is performed elementwise, so the encapsulated retractions
method has to be one that is available on the manifolds.
"""
retract(::ProductManifold, ::Any...)

for TP in [ProductRepr, ArrayPartition]
    eval(
        quote
            function _retract(
                M::ProductManifold,
                p::$TP,
                X::$TP,
                method::ProductRetraction,
            )
                return $TP(
                    map(
                        retract,
                        M.manifolds,
                        submanifold_components(M, p),
                        submanifold_components(M, X),
                        method.retractions,
                    ),
                )
            end
        end,
    )
end

function _retract!(M::ProductManifold, q, p, X, method::ProductRetraction)
    map(
        retract!,
        M.manifolds,
        submanifold_components(M, q),
        submanifold_components(M, p),
        submanifold_components(M, X),
        method.retractions,
    )
    return q
end

function representation_size(M::ProductManifold)
    return (mapreduce(m -> prod(representation_size(m)), +, M.manifolds),)
end

@doc raw"""
    riemann_tensor(M::ProductManifold, p, X, Y, Z)

Compute the Riemann tensor at point from `p` with tangent vectors `X`, `Y` and `Z` on
the [`ProductManifold`](@ref) `M`.
"""
riemann_tensor(M::ProductManifold, p, X, Y, X)

for TP in [ProductRepr, ArrayPartition]
    eval(
        quote
            function riemann_tensor(M::ProductManifold, p::$TP, X::$TP, Y::$TP, Z::$TP)
                return $TP(
                    map(
                        riemann_tensor,
                        M.manifolds,
                        submanifold_components(M, p),
                        submanifold_components(M, X),
                        submanifold_components(M, Y),
                        submanifold_components(M, Z),
                    ),
                )
            end
        end,
    )
end

function riemann_tensor!(M::ProductManifold, Xresult, p, X, Y, Z)
    map(
        riemann_tensor!,
        M.manifolds,
        submanifold_components(M, Xresult),
        submanifold_components(M, p),
        submanifold_components(M, X),
        submanifold_components(M, Y),
        submanifold_components(M, Z),
    )
    return Xresult
end

"""
    set_component!(M::ProductManifold, q, p, i)

Set the `i`th component of a point `q` on a [`ProductManifold`](@ref) `M` to `p`, where `p` is a point on the [`AbstractManifold`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/types.html#ManifoldsBase.AbstractManifold)  this factor of the product manifold consists of.
"""
function set_component!(M::ProductManifold, q, p, i)
    return copyto!(submanifold_component(M, q, i), p)
end

"""
    setindex!(q, p, M::ProductManifold, i::Union{Integer,Colon,AbstractVector})
    q[M::ProductManifold,i...] = p

set the element `[i...]` of a point `q` on a [`ProductManifold`](@ref) by linear indexing to `q`.
See also [Array Indexing](https://docs.julialang.org/en/v1/manual/arrays/#man-array-indexing-1) in Julia.
"""
Base.@propagate_inbounds function Base.setindex!(
    q::Union{ProductRepr,ArrayPartition},
    p,
    M::ProductManifold,
    i::Union{Integer,Colon,AbstractVector,Val},
)
    return set_component!(M, q, p, i)
end

@doc raw"""
    sharp(M::ProductManifold, p, ξ::FVector{CotangentSpaceType})

Use the musical isomorphism to transform the cotangent vector `ξ` from the tangent space at
`p` on the [`ProductManifold`](@ref) `M` to a tangent vector.
This can be done elementwise for every entry of `ξ` (and `p`) separately
"""
sharp(::ProductManifold, ::Any...)

function _show_submanifold(io::IO, M::AbstractManifold; pre="")
    sx = sprint(show, "text/plain", M, context=io, sizehint=0)
    if occursin('\n', sx)
        sx = sprint(show, M, context=io, sizehint=0)
    end
    sx = replace(sx, '\n' => "\n$(pre)")
    print(io, pre, sx)
    return nothing
end

function _show_submanifold_range(io::IO, Ms, range; pre="")
    for i in range
        M = Ms[i]
        print(io, '\n')
        _show_submanifold(io, M; pre=pre)
    end
    return nothing
end

function _show_product_manifold_no_header(io::IO, M)
    n = length(M.manifolds)
    sz = displaysize(io)
    screen_height, screen_width = sz[1] - 4, sz[2]
    half_height = div(screen_height, 2)
    inds = 1:n
    pre = " "
    if n > screen_height
        inds = [1:half_height; (n - div(screen_height - 1, 2) + 1):n]
    end
    if n ≤ screen_height
        _show_submanifold_range(io, M.manifolds, 1:n; pre=pre)
    else
        _show_submanifold_range(io, M.manifolds, 1:half_height; pre=pre)
        print(io, "\n$(pre)⋮")
        _show_submanifold_range(
            io,
            M.manifolds,
            (n - div(screen_height - 1, 2) + 1):n;
            pre=pre,
        )
    end
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", M::ProductManifold)
    n = length(M.manifolds)
    print(io, "ProductManifold with $(n) submanifold$(n == 1 ? "" : "s"):")
    return _show_product_manifold_no_header(io, M)
end

function Base.show(io::IO, M::ProductManifold)
    return print(io, "ProductManifold(", join(M.manifolds, ", "), ")")
end

function Base.show(
    io::IO,
    mime::MIME"text/plain",
    B::CachedBasis{𝔽,T,D},
) where {𝔽,T<:AbstractBasis{𝔽},D<:ProductBasisData}
    println(io, "$(T) for a product manifold")
    for (i, cb) in enumerate(B.data.parts)
        println(io, "Basis for component $i:")
        show(io, mime, cb)
        println(io)
    end
    return nothing
end

"""
    submanifold(M::ProductManifold, i::Integer)

Extract the `i`th factor of the product manifold `M`.
"""
submanifold(M::ProductManifold, i::Integer) = M.manifolds[i]

"""
    submanifold(M::ProductManifold, i::Val)
    submanifold(M::ProductManifold, i::AbstractVector)

Extract the factor of the product manifold `M` indicated by indices in `i`.
For example, for `i` equal to `Val((1, 3))` the product manifold constructed
from the first and the third factor is returned.

The version with `AbstractVector` is not type-stable, for better preformance use `Val`.
"""
function submanifold(M::ProductManifold, i::Val)
    return ProductManifold(select_from_tuple(M.manifolds, i)...)
end
submanifold(M::ProductManifold, i::AbstractVector) = submanifold(M, Val(tuple(i...)))

Distributions.support(d::ProductPointDistribution) = MPointSupport(d.manifold)
function Distributions.support(tvd::ProductFVectorDistribution)
    return FVectorSupport(
        tvd.type,
        ProductRepr(map(d -> support(d).point, tvd.distributions)...),
    )
end

function uniform_distribution(M::ProductManifold)
    return ProductPointDistribution(M, map(uniform_distribution, M.manifolds))
end
function uniform_distribution(M::ProductManifold, p)
    return ProductPointDistribution(
        M,
        map(uniform_distribution, M.manifolds, submanifold_components(M, p)),
    )
end

function vector_bundle_transport(::VectorSpaceType, M::ProductManifold)
    return ProductVectorTransport(map(_ -> ParallelTransport(), M.manifolds))
end

for TP in [ProductRepr, ArrayPartition]
    eval(
        quote
            function vector_transport_direction(
                M::ProductManifold,
                p::$TP,
                X::$TP,
                d::$TP,
                m::ProductVectorTransport,
            )
                return $TP(
                    map(
                        vector_transport_direction,
                        M.manifolds,
                        submanifold_components(M, p),
                        submanifold_components(M, X),
                        submanifold_components(M, d),
                        m.methods,
                    ),
                )
            end
        end,
    )
end
function vector_transport_direction!(
    M::ProductManifold,
    Y,
    p,
    X,
    d,
    m::ProductVectorTransport,
)
    map(
        vector_transport_direction!,
        M.manifolds,
        submanifold_components(M, Y),
        submanifold_components(M, p),
        submanifold_components(M, X),
        submanifold_components(M, d),
        m.methods,
    )
    return Y
end

@doc raw"""
    vector_transport_to(M::ProductManifold, p, X, q, m::ProductVectorTransport)

Compute the vector transport the tangent vector `X`at `p` to `q` on the
[`ProductManifold`](@ref) `M` using an [`ProductVectorTransport`](@ref) `m`.
This method is performed elementwise, i.e. the method `m` has to be implemented on the
base manifold.
"""
vector_transport_to(::ProductManifold, ::Any, ::Any, ::Any, ::ProductVectorTransport)

for TP in [ProductRepr, ArrayPartition]
    eval(
        quote
            function vector_transport_to(
                M::ProductManifold,
                p::$TP,
                X::$TP,
                q::$TP,
                m::ProductVectorTransport,
            )
                return $TP(
                    map(
                        vector_transport_to,
                        M.manifolds,
                        submanifold_components(M, p),
                        submanifold_components(M, X),
                        submanifold_components(M, q),
                        m.methods,
                    ),
                )
            end
            function vector_transport_to(
                M::ProductManifold,
                p::$TP,
                X::$TP,
                q::$TP,
                m::ParallelTransport,
            )
                return $TP(
                    map(
                        (iM, ip, iX, id) -> vector_transport_to(iM, ip, iX, id, m),
                        M.manifolds,
                        submanifold_components(M, p),
                        submanifold_components(M, X),
                        submanifold_components(M, q),
                    ),
                )
            end
        end,
    )
end

function vector_transport_to!(M::ProductManifold, Y, p, X, q, m::ProductVectorTransport)
    map(
        vector_transport_to!,
        M.manifolds,
        submanifold_components(M, Y),
        submanifold_components(M, p),
        submanifold_components(M, X),
        submanifold_components(M, q),
        m.methods,
    )
    return Y
end
function vector_transport_to!(M::ProductManifold, Y, p, X, q, m::ParallelTransport)
    map(
        (iM, iY, ip, iX, id) -> vector_transport_to!(iM, iY, ip, iX, id, m),
        M.manifolds,
        submanifold_components(M, Y),
        submanifold_components(M, p),
        submanifold_components(M, X),
        submanifold_components(M, q),
    ),
    return Y
end

function zero_vector!(M::ProductManifold, X, p)
    map(
        zero_vector!,
        M.manifolds,
        submanifold_components(M, X),
        submanifold_components(M, p),
    )
    return X
end
