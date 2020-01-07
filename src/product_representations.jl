
abstract type AbstractReshaper end

"""
    StaticReshaper()

Reshaper that constructs [`SizedAbstractArray`](@ref).
"""
struct StaticReshaper <: AbstractReshaper end

"""
    make_reshape(reshaper::AbstractReshaper, ::Type{Size}, data) where Size

Reshape array `data` to size `Size` using method provided by `reshaper`.
"""
function make_reshape(reshaper::AbstractReshaper, ::Type{Size}, data) where Size
    error("make_reshape is not defined for reshaper of type $(typeof(reshaper)), size $(Size) and data of type $(typeof(data)).")
end

function make_reshape(::StaticReshaper, ::Type{Size}, data) where Size
    return SizedAbstractArray{Size}(data)
end

"""
    ArrayReshaper()

Reshaper that constructs `Base.ReshapedArray`.
"""
struct ArrayReshaper <: AbstractReshaper end

function make_reshape(::ArrayReshaper, ::Type{Size}, data) where Size
    return reshape(data, size_to_tuple(Size))
end

"""
    ShapeSpecification(reshapers, manifolds::Manifold...)

A structure for specifying array size and offset information for linear
storage of points and tangent vectors on the product manifold of `manifolds`.

The first argument, `reshapers`, indicates how a view representing a point
in the [`ProductArray`](@ref) will be reshaped. It can either be an object
of type `AbstractReshaper` that will be applied to all views or a tuple
of such objects that will be applied to subsequent manifolds.

Two main reshaping methods are provided by types [`StaticReshaper`](@ref)
that is faster for manifolds represented by small arrays (up to about 100
elements) and [`ArrayReshaper`](@ref) that is faster for larger arrays.

For example, consider the shape specification for the product of
a sphere and group of rotations:

```julia-repl
julia> M1 = Sphere(2)
Sphere{2}()

julia> M2 = Manifolds.Rotations(2)
Manifolds.Rotations{2}()

julia> reshaper = Manifolds.StaticReshaper()
Manifolds.StaticReshaper()

julia> shape = Manifolds.ShapeSpecification(reshaper, M1, M2)
Manifolds.ShapeSpecification{(1:3, 4:7),Tuple{Tuple{3},Tuple{2,2}},
  Tuple{Manifolds.StaticReshaper,Manifolds.StaticReshaper}}(
  (Manifolds.StaticReshaper(), Manifolds.StaticReshaper()))
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
struct ShapeSpecification{TRanges, TSizes, TReshapers}
    reshapers::TReshapers
end

function ShapeSpecification(reshapers, manifolds::Manifold...)
    sizes = map(m -> representation_size(m), manifolds)
    lengths = map(prod, sizes)
    ranges = UnitRange{Int64}[]
    k = 1
    for len ∈ lengths
        push!(ranges, k:(k+len-1))
        k += len
    end
    TRanges = tuple(ranges...)
    TSizes = Tuple{map(s -> Tuple{s...}, sizes)...}
    if isa(reshapers, AbstractReshaper)
        rtuple = map(m -> reshapers, manifolds)
        return ShapeSpecification{TRanges, TSizes, typeof(rtuple)}(rtuple)
    else
        return ShapeSpecification{TRanges, TSizes, typeof(reshapers)}(reshapers)
    end
end

"""
    ProductArray(shape::ShapeSpecification, data)

An array-based representation for points and tangent vectors on the
product manifold. `data` contains underlying representation of points
arranged according to `TRanges` and `TSizes` from `shape`.
Internal views for each specific sub-point are created and stored in `parts`.
"""
struct ProductArray{TM<:ShapeSpecification,T,N,TData<:AbstractArray{T,N},TV<:Tuple,TReshaper} <: AbstractArray{T,N}
    data::TData
    parts::TV
    reshapers::TReshaper
end

# The two-argument version of this constructor is substantially faster than
# the generic one.
function ProductArray(M::Type{ShapeSpecification{TRanges, Tuple{Size1, Size2}, TReshapers}}, data::TData, reshapers) where {TRanges, Size1, Size2, TReshapers, T, N, TData<:AbstractArray{T,N}}
    views = (make_reshape(reshapers[1], Size1, view(data, TRanges[1])),
             make_reshape(reshapers[2], Size2, view(data, TRanges[2])))
    return ProductArray{M, T, N, TData, typeof(views), typeof(reshapers)}(data, views, reshapers)
end

function ProductArray(M::Type{ShapeSpecification{TRanges, Tuple{Size1, Size2, Size3}, TReshapers}}, data::TData, reshapers) where {TRanges, Size1, Size2, Size3, TReshapers, T, N, TData<:AbstractArray{T,N}}
    views = (make_reshape(reshapers[1], Size1, view(data, TRanges[1])),
             make_reshape(reshapers[2], Size2, view(data, TRanges[2])),
             make_reshape(reshapers[3], Size3, view(data, TRanges[3])))
    return ProductArray{M, T, N, TData, typeof(views), typeof(reshapers)}(data, views, reshapers)
end

function ProductArray(M::Type{ShapeSpecification{TRanges, TSizes, TReshapers}}, data::TData, reshapers) where {TRanges, TSizes, TReshapers, T, N, TData<:AbstractArray{T,N}}
    views = map((size, range, reshaper) -> make_reshape(reshaper, size, view(data, range)), size_to_tuple(TSizes), TRanges, reshapers)
    return ProductArray{M, T, N, TData, typeof(views), typeof(reshapers)}(data, views, reshapers)
end

function ProductArray(M::ShapeSpecification, data)
    return ProductArray(typeof(M), data, M.reshapers)
end

@doc doc"""
    prod_point(M::ShapeSpecification, pts...)

Construct a product point from product manifold `M` based on point `pts`
represented by [`ProductArray`](@ref).

# Example
To construct a point on the product manifold $S^2 \times \mathbb{R}^2$
from points on the sphere and in the euclidean space represented by,
respectively, `[1.0, 0.0, 0.0]` and `[-3.0, 2.0]` you need to construct shape
specification first. It describes how linear storage of `ProductArray`
corresponds to array representations expected by `Sphere(2)` and `Euclidean(2)`.

    M1 = Sphere(2)
    M2 = Euclidean(2)
    reshaper = Manifolds.StaticReshaper()
    Mshape = Manifolds.ShapeSpecification(reshaper, M1, M2)

Next, the desired point on the product manifold can be obtained by calling
`Manifolds.prod_point(Mshape, [1.0, 0.0, 0.0], [-3.0, 2.0])`.
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
    submanifold_component(x::ProductArray, i::Integer)

Project the product array `x` to its `i`th component. A new array is returned.
"""
function submanifold_component(x::ProductArray, i::Integer)
    return x.parts[i]
end

Base.BroadcastStyle(::Type{<:ProductArray{ShapeSpec}}) where ShapeSpec<:ShapeSpecification = Broadcast.ArrayStyle{ProductArray{ShapeSpec}}()

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{ProductArray{ShapeSpec}}}, ::Type{ElType}) where {ShapeSpec, ElType}
    A = find_pv(bc)
    return ProductArray(ShapeSpec, similar(A.data, ElType), A.reshapers)
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

(+)(v1::ProductArray{ShapeSpec}, v2::ProductArray{ShapeSpec}) where ShapeSpec<:ShapeSpecification = ProductArray(ShapeSpec, v1.data + v2.data, v1.reshapers)
(-)(v1::ProductArray{ShapeSpec}, v2::ProductArray{ShapeSpec}) where ShapeSpec<:ShapeSpecification = ProductArray(ShapeSpec, v1.data - v2.data, v1.reshapers)
(-)(v::ProductArray{ShapeSpec}) where ShapeSpec<:ShapeSpecification = ProductArray(ShapeSpec, -v.data, v.reshapers)
(*)(a::Number, v::ProductArray{ShapeSpec}) where ShapeSpec<:ShapeSpecification = ProductArray(ShapeSpec, a*v.data, v.reshapers)

eltype(::Type{ProductArray{TM, TData, TV}}) where {TM, TData, TV} = eltype(TData)
similar(x::ProductArray{ShapeSpec}) where ShapeSpec<:ShapeSpecification = ProductArray(ShapeSpec, similar(x.data), x.reshapers)
similar(x::ProductArray{ShapeSpec}, ::Type{T}) where {ShapeSpec<:ShapeSpecification, T} = ProductArray(ShapeSpec, similar(x.data, T), x.reshapers)

"""
    ProductRepr(parts)

A more general but slower representation of points and tangent vectors on
a product manifold.

# Example:

A product point on a product manifold `Sphere(2) × Euclidean(2)` might be
created as

    ProductRepr([1.0, 0.0, 0.0], [2.0, 3.0])

where `[1.0, 0.0, 0.0]` is the part corresponding to the sphere factor
and `[2.0, 3.0]` is the part corresponding to the euclidean manifold.
"""
struct ProductRepr{TM<:Tuple}
    parts::TM
end

ProductRepr(points...) = ProductRepr{typeof(points)}(points)
eltype(x::ProductRepr) = eltype(Tuple{map(eltype, x.parts)...})
similar(x::ProductRepr) = ProductRepr(map(similar, x.parts)...)
similar(x::ProductRepr, ::Type{T}) where T = ProductRepr(map(t -> similar(t, T), x.parts)...)

function copyto!(x::ProductRepr, y::ProductRepr)
    map(copyto!, x.parts, y.parts)
    return x
end

function submanifold_component(x::ProductRepr, i::Integer)
    return x.parts[i]
end

(+)(v1::ProductRepr, v2::ProductRepr) = ProductRepr(map(+, v1.parts, v2.parts)...)
(-)(v1::ProductRepr, v2::ProductRepr) = ProductRepr(map(-, v1.parts, v2.parts)...)
(-)(v::ProductRepr) = ProductRepr(map(-, v.parts))
(*)(a::Number, v::ProductRepr) = ProductRepr(map(t -> a*t, v.parts))
