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

struct ProductView{TM<:ProductManifold,T,N,TData<:AbstractArray{T,N},TV<:Tuple} <: AbstractArray{T,N}
    M::TM
    data::TData
    views::TV
end

# The two-argument version of this constructor is substantially faster than
# the generic one.
function ProductView(M::ProductManifold{TM, TRanges, Tuple{Size1, Size2}}, data::TData) where {TM, TRanges, Size1, Size2, T, N, TData<:AbstractArray{T,N}}
    #println("PV2")
    views = (SizedAbstractArray{Size1}(view(data, TRanges[1])),
             SizedAbstractArray{Size2}(view(data, TRanges[2])))
    return ProductView{typeof(M), T, N, TData, typeof(views)}(M, data, views)
end

function ProductView(M::ProductManifold{TM, TRanges, TSizes}, data::TData) where {TM, TRanges, TSizes, TData<:AbstractArray}
    #println("PVn")
    views = map(ziptuples(TRanges, TSizes)) do t
        SizedAbstractArray{t[2]}(view(data, t[1]))
    end
    return ProductView{ProductManifold{TM, TRanges, TSizes}, TData, typeof(views)}(M, data, views)
end

Base.BroadcastStyle(::Type{<:ProductView}) = Broadcast.ArrayStyle{ProductView}()

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{ProductView}}, ::Type{ElType}) where ElType
    # Scan the inputs for the ProductView:
    A = find_pv(bc)
    return ProductView(A.M, similar(A.data, ElType))
end

Base.dataids(x::ProductView) = Base.dataids(x.data)

"""
    find_pv(x...)

`A = find_pv(x...)` returns the first `ProductView` among the arguments.
"""
find_pv(bc::Base.Broadcast.Broadcasted) = find_pv(bc.args)
find_pv(args::Tuple) = find_pv(find_pv(args[1]), Base.tail(args))
find_pv(x) = x
find_pv(a::ProductView, rest) = a
find_pv(::Any, rest) = find_pv(rest)

size(x::ProductView) = size(x.data)
Base.@propagate_inbounds getindex(x::ProductView, i) = getindex(x.data, i)
Base.@propagate_inbounds setindex!(x::ProductView, val, i) = setindex!(x.data, val, i)

(+)(v1::ProductView, v2::ProductView) = ProductView(v1.M, v1.data + v2.data)
(-)(v1::ProductView, v2::ProductView) = ProductView(v1.M, v1.data - v2.data)
(-)(v::ProductView) = ProductView(v.M, -v.data)
(*)(a::Number, v::ProductView) = ProductView(v.M, a*v.data)

eltype(::Type{ProductView{TM, TData, TV}}) where {TM, TData, TV} = eltype(TData)
similar(x::ProductView) = ProductView(x.M, similar(x.data))
similar(x::ProductView, ::Type{T}) where T = ProductView(x.M, similar(x.data, T))

function isapprox(M::ProductManifold, x, y; kwargs...)
    return mapreduce(&, ziptuples(M.manifolds, x.views, y.views)) do t
        return isapprox(t[1], t[2], t[3]; kwargs...)
    end
end

function isapprox(M::ProductManifold, x, v, w; kwargs...)
    return mapreduce(&, ziptuples(M.manifolds, x.views, v.views, w.views)) do t
        return isapprox(t[1], t[2], t[3], t[4]; kwargs...)
    end
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

function det_local_metric(M::MetricManifold{ProductManifold, ProductMetric}, x::ProductView)
    dets = map(det_local_metric, M.manifolds, x.views)
    return prod(dets)
end

function inner(M::ProductManifold, x::ProductView, v::ProductView, w::ProductView)
    subproducts = map(inner, M.manifolds, x.views, v.views, w.views)
    return sum(subproducts)
end

function exp!(M::ProductManifold, y::ProductView, x::ProductView, v::ProductView)
    map(exp!, M.manifolds, y.views, x.views, v.views)
    return y
end

function log!(M::ProductManifold, v::ProductView, x::ProductView, y::ProductView)
    map(log!, M.manifolds, v.views, x.views, y.views)
    return v
end
