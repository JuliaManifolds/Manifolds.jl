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
struct ProductManifold{TM<:Tuple, TRanges, TSizes} <: Manifold
    manifolds::TM
end

function ProductManifold(manifolds...)
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
    return ProductManifold{typeof(manifolds), TRanges, TSizes}(manifolds)
end

struct ProductArray{TM<:ProductManifold,T,N,TData<:AbstractArray{T,N},TV<:Tuple} <: AbstractArray{T,N}
    M::TM
    data::TData
    views::TV
end

# The two-argument version of this constructor is substantially faster than
# the generic one.
function ProductArray(M::ProductManifold{TM, TRanges, Tuple{Size1, Size2}}, data::TData) where {TM, TRanges, Size1, Size2, T, N, TData<:AbstractArray{T,N}}
    views = (SizedAbstractArray{Size1}(view(data, TRanges[1])),
             SizedAbstractArray{Size2}(view(data, TRanges[2])))
    return ProductArray{typeof(M), T, N, TData, typeof(views)}(M, data, views)
end

function ProductArray(M::ProductManifold{TM, TRanges, TSizes}, data::TData) where {TM, TRanges, TSizes, T, N, TData<:AbstractArray{T,N}}
    views = map(ziptuples(TRanges, tuple(TSizes.parameters...))) do t
        SizedAbstractArray{t[2]}(view(data, t[1]))
    end
    return ProductArray{typeof(M), T, N, TData, typeof(views)}(M, data, views)
end

"""
    prod_point(M::ProductManifold, pts...)

Construct a product point from product manifold `M` based on point `pts`
represented by arrays.
"""
function prod_point(M::ProductManifold, pts...)
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

Base.BroadcastStyle(::Type{<:ProductArray}) = Broadcast.ArrayStyle{ProductArray}()

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{ProductArray}}, ::Type{ElType}) where ElType
    # Scan the inputs for the ProductArray:
    A = find_pv(bc)
    return ProductArray(A.M, similar(A.data, ElType))
end

Base.dataids(x::ProductArray) = Base.dataids(x.data)

"""
    find_pv(x...)

`A = find_pv(x...)` returns the first `ProductArray` among the arguments.
"""
find_pv(bc::Base.Broadcast.Broadcasted) = find_pv(bc.args)
find_pv(args::Tuple) = find_pv(find_pv(args[1]), Base.tail(args))
find_pv(x) = x
find_pv(a::ProductArray, rest) = a
find_pv(::Any, rest) = find_pv(rest)

size(x::ProductArray) = size(x.data)
Base.@propagate_inbounds getindex(x::ProductArray, i) = getindex(x.data, i)
Base.@propagate_inbounds setindex!(x::ProductArray, val, i) = setindex!(x.data, val, i)

(+)(v1::ProductArray, v2::ProductArray) = ProductArray(v1.M, v1.data + v2.data)
(-)(v1::ProductArray, v2::ProductArray) = ProductArray(v1.M, v1.data - v2.data)
(-)(v::ProductArray) = ProductArray(v.M, -v.data)
(*)(a::Number, v::ProductArray) = ProductArray(v.M, a*v.data)

eltype(::Type{ProductArray{TM, TData, TV}}) where {TM, TData, TV} = eltype(TData)
similar(x::ProductArray) = ProductArray(x.M, similar(x.data))
similar(x::ProductArray, ::Type{T}) where T = ProductArray(x.M, similar(x.data, T))

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
