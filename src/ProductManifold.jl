@doc doc"""
    ProductManifold{TM<:Tuple, TRanges<:Tuple, TSizes<:Tuple} <: Manifold

Product manifold $M_1 \times M_2 \times \dots \times M_n$ with product geometry.
`TRanges` and `TSizes` statically define the relationship between representation
of the product manifold and representations of point, tangent vectors
and cotangent vectors of respective manifolds.

# Constructor

    ProductManifold(M_1, M_2, ..., M_n)

generates the product manifold $M_1 \times M_2 \times \dots \times M_n$
"""
struct ProductManifold{TM<:Tuple} <: Manifold
    manifolds::TM
end

ProductManifold(manifolds::Manifold...) = ProductManifold{typeof(manifolds)}(manifolds)

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

struct ProductArray{TM<:ShapeSpecification,T,N,TData<:AbstractArray{T,N},TV<:Tuple} <: AbstractArray{T,N}
    data::TData
    views::TV
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
    proj_product_array(x::ProductArray, i::Integer)

Project the product array `x` to its `i`th component. A new array is returned.
"""
function proj_product_array(x::ProductArray, i::Integer)
    return copy(x.views[i])
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

function isapprox(M::ProductManifold, x, y; kwargs...)
    return all(t -> isapprox(t...; kwargs...), ziptuples(M.manifolds, x.views, y.views))
end

function isapprox(M::ProductManifold, x, v, w; kwargs...)
    return all(t -> isapprox(t...; kwargs...), ziptuples(M.manifolds, x.views, v.views, w.views))
end

function representation_size(M::ProductManifold, ::Type{T}) where {T}
    return (mapreduce(m -> sum(representation_size(m, T)), +, M.manifolds),)
end

manifold_dimension(M::ProductManifold) = sum(map(m -> manifold_dimension(m), M.manifolds))

struct ProductMetric <: Metric end

@traitimpl HasMetric{ProductManifold,ProductMetric}

function local_metric(::MetricManifold{<:ProductManifold,ProductMetric}, x)
    error("TODO")
end

function inverse_local_metric(M::MetricManifold{<:ProductManifold,ProductMetric}, x)
    error("TODO")
end

function det_local_metric(M::MetricManifold{ProductManifold, ProductMetric}, x::ProductArray)
    dets = map(det_local_metric, M.manifolds, x.views)
    return prod(dets)
end

function inner(M::ProductManifold, x::ProductArray, v::ProductArray, w::ProductArray)
    subproducts = map(inner, M.manifolds, x.views, v.views, w.views)
    return sum(subproducts)
end

function exp!(M::ProductManifold, y::ProductArray, x::ProductArray, v::ProductArray)
    map(exp!, M.manifolds, y.views, x.views, v.views)
    return y
end

function log!(M::ProductManifold, v::ProductArray, x::ProductArray, y::ProductArray)
    map(log!, M.manifolds, v.views, x.views, y.views)
    return v
end
