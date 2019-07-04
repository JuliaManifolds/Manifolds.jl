@doc doc"""
    ProductManifold{TM<:Tuple, TRanges<:Tuple, TSizes<:Tuple} <: Manifold

Product manifold $M_1 \times M_2 \times \dots \times M_n$ with product geometry.
`TRanges` and `TSizes` statically define the relationship between representation
of the product manifold and representations of point, tangent vectors
and cotangent vectors of respective manifolds.

# Constructor

    ProductManifold(M_1, M_2, ..., M_n)

generates the product manifold $M_1 \times M_2 \times \dots \times M_n$.
Alternatively, the same manifold can be contructed using the `×` operator:
`M_1 × M_2 × M_3`.
"""
struct ProductManifold{TM<:Tuple} <: Manifold
    manifolds::TM
end

ProductManifold(manifolds::Manifold...) = ProductManifold{typeof(manifolds)}(manifolds)

function cross(M1::Manifold, M2::Manifold)
    return ProductManifold(M1, M2)
end

function cross(M1::ProductManifold, M2::Manifold)
    return ProductManifold(M1.manifolds..., M2)
end

function cross(M1::Manifold, M2::ProductManifold)
    return ProductManifold(M1, M2.manifolds...)
end

function cross(M1::ProductManifold, M2::ProductManifold)
    return ProductManifold(M1.manifolds..., M2.manifolds...)
end

"""
    ShapeSpecification(manifolds::Manifold...)

A structure for specifying array size and offset information for linear
storage of points and tangent vectors on the product manifold of `manifolds`.

For example, consider the shape specification for the product of
a sphere and group of rotations:

```julia-repl
julia> M1 = Sphere(2)
Sphere{2}()

julia> M3 = Rotations(2)
Rotations{2}()

julia> shape = Manifolds.ShapeSpecification(M1, M3)
Manifolds.ShapeSpecification{(1:3, 4:7),Tuple{Tuple{3},Tuple{2,2}}}()
```

`TRanges` contains ranges in the linear storage that correspond to a specific
manifold. `Sphere(2)` needs three numbers and is first, so it is allocated the
first three elements of the linear storage (`1:3`). `Rotations(2)` needs four
numbers and is second, so the next four numbers are allocated to it (`4:7`).
`TSizes` describe how the linear storage must be reshaped to correctly
represent points. In this case, `Sphere(2)` expects a three-element vector, so
the corresponding size is `Tuple{3}`. On the other hand, `Rotations(2)`
expects two-by-two matrices, so its size specification is `Tuple{2,2}`.
"""
struct ShapeSpecification{TRanges, TSizes} end

function ShapeSpecification(manifolds::Manifold...)
    sizes = map(m -> representation_size(m, MPoint), manifolds)
    lengths = map(prod, sizes)
    ranges = UnitRange{Int64}[]
    k = 1
    for len ∈ lengths
        push!(ranges, k:(k+len-1))
        k += len
    end
    TRanges = tuple(ranges...)
    TSizes = Tuple{map(s -> Tuple{s...}, sizes)...}
    return ShapeSpecification{TRanges, TSizes}()
end

"""
    ProductArray(shape::ShapeSpecification, data)

An array-based representation for points and tangent vectors on the
product manifold. `data` contains underlying representation of points
arranged according to `TRanges` and `TSizes` from `shape`.
Internal views for each specific sub-point are created and stored in `parts`.
"""
struct ProductArray{TM<:ShapeSpecification,T,N,TData<:AbstractArray{T,N},TV<:Tuple} <: AbstractArray{T,N}
    data::TData
    parts::TV
end

# The two-argument version of this constructor is substantially faster than
# the generic one.
function ProductArray(M::Type{ShapeSpecification{TRanges, Tuple{Size1, Size2}}}, data::TData) where {TRanges, Size1, Size2, T, N, TData<:AbstractArray{T,N}}
    views = (SizedAbstractArray{Size1}(view(data, TRanges[1])),
             SizedAbstractArray{Size2}(view(data, TRanges[2])))
    return ProductArray{M, T, N, TData, typeof(views)}(data, views)
end

function ProductArray(M::Type{ShapeSpecification{TRanges, Tuple{Size1, Size2, Size3}}}, data::TData) where {TRanges, Size1, Size2, Size3, T, N, TData<:AbstractArray{T,N}}
    views = (SizedAbstractArray{Size1}(view(data, TRanges[1])),
             SizedAbstractArray{Size2}(view(data, TRanges[2])),
             SizedAbstractArray{Size3}(view(data, TRanges[3])))
    return ProductArray{M, T, N, TData, typeof(views)}(data, views)
end

function ProductArray(M::ShapeSpecification{TRanges, TSizes}, data::TData) where {TM, TRanges, TSizes, T, N, TData<:AbstractArray{T,N}}
    return ProductArray(typeof(M), data)
end

"""
    prod_point(M::ProductManifold, pts...)

Construct a product point from product manifold `M` based on point `pts`
represented by arrays.
"""
function prod_point(M::ShapeSpecification, pts...)
    data = mapreduce(vcat, pts) do pt
        reshape(pt, :)
    end
    # Array(data) is used to ensure that the data is mutable
    # `mapreduce` can return `SArray` for some arguments
    return ProductArray(M, Array(data))
end

"""
    proj_product(x::ProductArray, i::Integer)

Project the product array `x` to its `i`th component. A new array is returned.
"""
function proj_product(x::ProductArray, i::Integer)
    return copy(x.parts[i])
end

Base.BroadcastStyle(::Type{<:ProductArray{ShapeSpec}}) where ShapeSpec<:ShapeSpecification = Broadcast.ArrayStyle{ProductArray{ShapeSpec}}()

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{ProductArray{ShapeSpec}}}, ::Type{ElType}) where {ShapeSpec, ElType}
    A = find_pv(bc)
    return ProductArray(ShapeSpec, similar(A.data, ElType))
end

Base.dataids(x::ProductArray) = Base.dataids(x.data)

"""
    find_pv(x...)

`A = find_pv(x...)` returns the first `ProductArray` among the arguments.
"""
@inline find_pv(bc::Base.Broadcast.Broadcasted) = find_pv(bc.args)
@inline find_pv(args::Tuple) = find_pv(find_pv(args[1]), Base.tail(args))
@inline find_pv(x) = x
@inline find_pv(a::ProductArray, rest) = a
@inline find_pv(::Any, rest) = find_pv(rest)

size(x::ProductArray) = size(x.data)
Base.@propagate_inbounds getindex(x::ProductArray, i) = getindex(x.data, i)
Base.@propagate_inbounds setindex!(x::ProductArray, val, i) = setindex!(x.data, val, i)

(+)(v1::ProductArray{ShapeSpec}, v2::ProductArray{ShapeSpec}) where ShapeSpec<:ShapeSpecification = ProductArray(ShapeSpec, v1.data + v2.data)
(-)(v1::ProductArray{ShapeSpec}, v2::ProductArray{ShapeSpec}) where ShapeSpec<:ShapeSpecification = ProductArray(ShapeSpec, v1.data - v2.data)
(-)(v::ProductArray{ShapeSpec}) where ShapeSpec<:ShapeSpecification = ProductArray(ShapeSpec, -v.data)
(*)(a::Number, v::ProductArray{ShapeSpec}) where ShapeSpec<:ShapeSpecification = ProductArray(ShapeSpec, a*v.data)

eltype(::Type{ProductArray{TM, TData, TV}}) where {TM, TData, TV} = eltype(TData)
similar(x::ProductArray{ShapeSpec}) where ShapeSpec<:ShapeSpecification = ProductArray(ShapeSpec, similar(x.data))
similar(x::ProductArray{ShapeSpec}, ::Type{T}) where {ShapeSpec<:ShapeSpecification, T} = ProductArray(ShapeSpec, similar(x.data, T))

"""
    ProductMPoint(parts)

A more general but slower representation of points on a product manifold.
"""
struct ProductMPoint{TM<:Tuple} <: MPoint
    parts::TM
end

ProductMPoint(points...) = ProductMPoint{typeof(points)}(points)
eltype(x::ProductMPoint) = eltype(Tuple{map(eltype, x.parts)...})
similar(x::ProductMPoint) = ProductMPoint(map(similar, x.parts)...)
similar(x::ProductMPoint, ::Type{T}) where T = ProductMPoint(map(t -> similar(t, T), x.parts)...)

function proj_product(x::ProductMPoint, i::Integer)
    return x.parts[i]
end

"""
    ProductTVector(parts)

A more general but slower representation of tangent vectors on a product
manifold.
"""
struct ProductTVector{TM<:Tuple} <: TVector
    parts::TM
end

ProductTVector(vectors...) = ProductTVector{typeof(vectors)}(vectors)
eltype(x::ProductTVector) = eltype(Tuple{map(eltype, x.parts)...})
similar(x::ProductTVector) = ProductTVector(map(similar, x.parts)...)
similar(x::ProductTVector, ::Type{T}) where T = ProductTVector(map(t -> similar(t, T), x.parts)...)

(+)(v1::ProductTVector, v2::ProductTVector) = ProductTVector(map(+, v1.parts, v2.parts)...)
(-)(v1::ProductTVector, v2::ProductTVector) = ProductTVector(map(-, v1.parts, v2.parts)...)
(-)(v::ProductTVector) = ProductTVector(map(-, v.parts))
(*)(a::Number, v::ProductTVector) = ProductTVector(map(t -> a*t, v.parts))

function proj_product(x::ProductTVector, i::Integer)
    return x.parts[i]
end

"""
    ProductCoTVector(parts)

A more general but slower representation of cotangent vectors on a product
manifold.
"""
struct ProductCoTVector{TM<:Tuple} <: CoTVector
    parts::TM
end

ProductCoTVector(covectors...) = ProductCoTVector{typeof(covectors)}(covectors)
eltype(x::ProductCoTVector) = eltype(Tuple{map(eltype, x.parts)...})
similar(x::ProductCoTVector) = ProductCoTVector(map(similar, x.parts)...)
similar(x::ProductCoTVector, ::Type{T}) where T = ProductCoTVector(map(t -> similar(t, T), x.parts)...)

function proj_product(x::ProductCoTVector, i::Integer)
    return x.parts[i]
end

function isapprox(M::ProductManifold, x, y; kwargs...)
    return all(t -> isapprox(t...; kwargs...), ziptuples(M.manifolds, x.parts, y.parts))
end

function isapprox(M::ProductManifold, x, v, w; kwargs...)
    return all(t -> isapprox(t...; kwargs...), ziptuples(M.manifolds, x.parts, v.parts, w.parts))
end

function representation_size(M::ProductManifold, ::Type{T}) where {T}
    return (mapreduce(m -> sum(representation_size(m, T)), +, M.manifolds),)
end

manifold_dimension(M::ProductManifold) = sum(map(m -> manifold_dimension(m), M.manifolds))

struct ProductMetric <: Metric end

function det_local_metric(M::MetricManifold{ProductManifold, ProductMetric}, x::ProductArray)
    dets = map(det_local_metric, M.manifolds, x.parts)
    return prod(dets)
end

function inner(M::ProductManifold, x, v, w)
    subproducts = map(inner, M.manifolds, x.parts, v.parts, w.parts)
    return sum(subproducts)
end

function exp!(M::ProductManifold, y, x, v)
    map(exp!, M.manifolds, y.parts, x.parts, v.parts)
    return y
end

function exp(M::ProductManifold, x::ProductMPoint, v::ProductTVector)
    return ProductMPoint(map(exp, M.manifolds, x.parts, v.parts)...)
end

function log!(M::ProductManifold, v, x, y)
    map(log!, M.manifolds, v.parts, x.parts, y.parts)
    return v
end

function log(M::ProductManifold, x::ProductMPoint, y::ProductMPoint)
    return ProductTVector(map(log, M.manifolds, x.parts, y.parts)...)
end

function distance(M::ProductManifold, x, y)
    return sqrt(sum(map(distance, M.manifolds, x.parts, y.parts).^2))
end

function injectivity_radius(M::ProductManifold, x)
    return min(map(injectivity_radius, M.manifolds, x.parts)...)
end

function injectivity_radius(M::ProductManifold)
    return min(map(injectivity_radius, M.manifolds)...)
end

"""
    ProductRetraction(retractions::AbstractRetractionMethod...)

Product retraction of `retractions`. Works on [`ProductManifold`](@ref).
"""
struct ProductRetraction{TR<:Tuple} <: AbstractRetractionMethod
    retractions::TR
end

ProductRetraction(retractions::AbstractRetractionMethod...) = ProductRetraction{typeof(retractions)}(retractions)

function retract!(M::ProductManifold, y, x, v, method::ProductRetraction)
    map(retract!, M.manifolds, y.parts, x.parts, v.parts, method.retractions)
    return y
end

struct InverseProductRetraction{TR<:Tuple} <: AbstractInverseRetractionMethod
    inverse_retractions::TR
end

"""
    InverseProductRetraction(inverse_retractions::AbstractInverseRetractionMethod...)

Product inverse retraction of `inverse_retractions`.
Works on [`ProductManifold`](@ref).
"""
InverseProductRetraction(inverse_retractions::AbstractInverseRetractionMethod...) = InverseProductRetraction{typeof(inverse_retractions)}(inverse_retractions)

function inverse_retract!(M::ProductManifold, v, x, y, method::InverseProductRetraction)
    map(inverse_retract!, M.manifolds, v.parts, x.parts, y.parts, method.inverse_retractions)
    return v
end

"""
    is_manifold_point(M::ProductManifold, x; kwargs...)

Check whether `x` is a valid point on the [`ProductManifold`](@ref) `M`.

The tolerance for the last test can be set using the ´kwargs...`.
"""
function is_manifold_point(M::ProductManifold, x::MPoint; kwargs...)
    return all(t -> is_manifold_point(t...; kwargs...), ziptuples(M.manifolds, x.parts))
end

function is_manifold_point(M::ProductManifold, x::ProductArray; kwargs...)
    return all(t -> is_manifold_point(t...; kwargs...), ziptuples(M.manifolds, x.parts))
end

"""
    is_tangent_vector(M::ProductManifold, x, v; kwargs... )

Check whether `v` is a tangent vector to `x` on the [`ProductManifold`](@ref)
`M`, i.e. atfer [`is_manifold_point`](@ref)`(M, x)`, and all projections to
base manifolds must be respective tangent vectors.

The tolerance for the last test can be set using the ´kwargs...`.
"""
function is_tangent_vector(M::ProductManifold, x::MPoint, v::TVector; kwargs...)
    is_manifold_point(M, x)
    return all(t -> is_tangent_vector(t...; kwargs...), ziptuples(M.manifolds, x.parts, v.parts))
end

function is_tangent_vector(M::ProductManifold, x::ProductArray, v::ProductArray; kwargs...)
    is_manifold_point(M, x)
    return all(t -> is_tangent_vector(t...; kwargs...), ziptuples(M.manifolds, x.parts, v.parts))
end

"""
    ProductPointDistribution(M::ProductManifold, distributions)

Product distribution on manifold `M`, combined from `distributions`.
"""
struct ProductPointDistribution{TM<:ProductManifold, TD<:(NTuple{N,Distribution} where N)} <: MPointDistribution{TM}
    manifold::TM
    distributions::TD
end

function ProductPointDistribution(M::ProductManifold, distributions::MPointDistribution...)
    return ProductPointDistribution{typeof(M), typeof(distributions)}(M, distributions)
end

function ProductPointDistribution(distributions::MPointDistribution...)
    M = ProductManifold(map(d -> support(d).manifold, distributions)...)
    return ProductPointDistribution(M, distributions...)
end

function support(d::ProductPointDistribution)
    return MPointSupport(d.manifold)
end

function rand(rng::AbstractRNG, d::ProductPointDistribution)
    return ProductMPoint(map(d -> rand(rng, d), d.distributions)...)
end

function _rand!(rng::AbstractRNG, d::ProductPointDistribution, x::AbstractArray{<:Number})
    x .= rand(rng, d)
    return x
end

function _rand!(rng::AbstractRNG, d::ProductPointDistribution, x::ProductMPoint)
    map(t -> _rand!(rng, t[1], t[2]), d.distributions, x.parts)
    return x
end

"""
    ProductTVectorDistribution([m::ProductManifold], [x], distrs...)

Generates a random tangent vector at point `x` from manifold `m` using the
product distribution of given distributions.

Manifold and `x` can be automatically inferred from distributions `distrs`.
"""
struct ProductTVectorDistribution{TM<:ProductManifold, TD<:(NTuple{N,Distribution} where N), TX} <: TVectorDistribution{TM, TX}
    manifold::TM
    x::TX
    distributions::TD
end

function ProductTVectorDistribution(M::ProductManifold, x::Union{AbstractArray, MPoint}, distributions::TVectorDistribution...)
    return ProductTVectorDistribution{typeof(M), typeof(distributions), typeof(x)}(M, x, distributions)
end

function ProductTVectorDistribution(M::ProductManifold, distributions::TVectorDistribution...)
    x = ProductMPoint(map(d -> support(d).x, distributions))
    return ProductTVectorDistribution(M, x, distributions...)
end

function ProductTVectorDistribution(distributions::TVectorDistribution...)
    M = ProductManifold(map(d -> support(d).manifold, distributions)...)
    x = ProductMPoint(map(d -> support(d).x, distributions)...)
    return ProductTVectorDistribution(M, x, distributions...)
end

function support(tvd::ProductTVectorDistribution)
    return TVectorSupport(tvd.manifold, ProductMPoint(map(d -> support(d).x, tvd.distributions)...))
end

function rand(rng::AbstractRNG, d::ProductTVectorDistribution)
    return ProductTVector(map(d -> rand(rng, d), d.distributions)...)
end

function _rand!(rng::AbstractRNG, d::ProductTVectorDistribution, v::AbstractArray{<:Number})
    v .= rand(rng, d)
    return v
end

function _rand!(rng::AbstractRNG, d::ProductTVectorDistribution, v::ProductTVector)
    map(t -> _rand!(rng, t[1], t[2]), d.distributions, v.parts)
    return v
end
