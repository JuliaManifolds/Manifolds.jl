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
function make_reshape(reshaper::AbstractReshaper, ::Type{Size}, data) where {Size}
    error("make_reshape is not defined for reshaper of type $(typeof(reshaper)), size $(Size) and data of type $(typeof(data)).")
end
function make_reshape(::StaticReshaper, ::Type{Size}, data) where {Size}
    return SizedAbstractArray{Size}(data)
end

"""
    ArrayReshaper()

Reshaper that constructs `Base.ReshapedArray`.
"""
struct ArrayReshaper <: AbstractReshaper end

function make_reshape(::ArrayReshaper, ::Type{Size}, data) where {Size}
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
struct ShapeSpecification{TRanges,TSizes,TReshapers}
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
        return ShapeSpecification{TRanges,TSizes,typeof(rtuple)}(rtuple)
    else
        return ShapeSpecification{TRanges,TSizes,typeof(reshapers)}(reshapers)
    end
end

"""
    ProductArray(shape::ShapeSpecification, data)

An array-based representation for points and tangent vectors on the
product manifold. `data` contains underlying representation of points
arranged according to `TRanges` and `TSizes` from `shape`.
Internal views for each specific sub-point are created and stored in `parts`.
"""
struct ProductArray{
    TM<:ShapeSpecification,
    T,
    N,
    TData<:AbstractArray{T,N},
    TV<:Tuple,
    TReshaper,
} <: AbstractArray{T,N}
    data::TData
    parts::TV
    reshapers::TReshaper
end

# The two-argument version of this constructor is substantially faster than
# the generic one.
function ProductArray(
    M::Type{ShapeSpecification{TRanges,Tuple{Size1,Size2},TReshapers}},
    data::TData,
    reshapers,
) where {TRanges,Size1,Size2,TReshapers,T,N,TData<:AbstractArray{T,N}}
    views = (
        make_reshape(reshapers[1], Size1, view(data, TRanges[1])),
        make_reshape(reshapers[2], Size2, view(data, TRanges[2])),
    )
    return ProductArray{M,T,N,TData,typeof(views),typeof(reshapers)}(data, views, reshapers)
end
function ProductArray(
    M::Type{ShapeSpecification{TRanges,Tuple{Size1,Size2,Size3},TReshapers}},
    data::TData,
    reshapers,
) where {TRanges,Size1,Size2,Size3,TReshapers,T,N,TData<:AbstractArray{T,N}}
    views = (
        make_reshape(reshapers[1], Size1, view(data, TRanges[1])),
        make_reshape(reshapers[2], Size2, view(data, TRanges[2])),
        make_reshape(reshapers[3], Size3, view(data, TRanges[3])),
    )
    return ProductArray{M,T,N,TData,typeof(views),typeof(reshapers)}(data, views, reshapers)
end
function ProductArray(
    M::Type{ShapeSpecification{TRanges,TSizes,TReshapers}},
    data::TData,
    reshapers,
) where {TRanges,TSizes,TReshapers,T,N,TData<:AbstractArray{T,N}}
    views = map(
        (size, range, reshaper) -> make_reshape(reshaper, size, view(data, range)),
        size_to_tuple(TSizes),
        TRanges,
        reshapers,
    )
    return ProductArray{M,T,N,TData,typeof(views),typeof(reshapers)}(data, views, reshapers)
end
ProductArray(M::ShapeSpecification, data) = ProductArray(typeof(M), data, M.reshapers)

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

@doc doc"""
    submanifold_component(M::Manifold, x, i::Integer)
    submanifold_component(M::Manifold, x, ::Val(i)) where {i}
    submanifold_component(x, i::Integer)
    submanifold_component(x, ::Val(i)) where {i}

Project the product array `x` on `M` to its `i`th component. A new array is returned.
"""
submanifold_component(::Any...)
submanifold_component(M::Manifold, x, i::Integer) = submanifold_component(M, x, Val(i))
submanifold_component(M::Manifold, x, i::Val) = submanifold_component(x, i)
submanifold_component(x, ::Val{I}) where {I} = x.parts[I]
submanifold_component(x, i::Integer) = submanifold_component(x, Val(i))

@doc doc"""
    submanifold_components(M::Manifold, x)
    submanifold_components(x)

Get the projected components of `x` on the submanifolds of `M`.
"""
submanifold_components(::Any...)
submanifold_components(M::Manifold, x) = submanifold_components(x)
submanifold_components(x) = x.parts

function Base.BroadcastStyle(
    ::Type{<:ProductArray{ShapeSpec}},
) where {ShapeSpec<:ShapeSpecification}
    return Broadcast.ArrayStyle{ProductArray{ShapeSpec}}()
end

function Base.similar(
    bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{ProductArray{ShapeSpec}}},
    ::Type{ElType},
) where {ShapeSpec,ElType}
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

function (+)(
    v1::ProductArray{ShapeSpec},
    v2::ProductArray{ShapeSpec},
) where {ShapeSpec<:ShapeSpecification}
    return ProductArray(ShapeSpec, v1.data + v2.data, v1.reshapers)
end

function (-)(
    v1::ProductArray{ShapeSpec},
    v2::ProductArray{ShapeSpec},
) where {ShapeSpec<:ShapeSpecification}
    return ProductArray(ShapeSpec, v1.data - v2.data, v1.reshapers)
end
function (-)(v::ProductArray{ShapeSpec}) where {ShapeSpec<:ShapeSpecification}
    return ProductArray(ShapeSpec, -v.data, v.reshapers)
end

function (*)(a::Number, v::ProductArray{ShapeSpec}) where {ShapeSpec<:ShapeSpecification}
    return ProductArray(ShapeSpec, a * v.data, v.reshapers)
end

number_eltype(::Type{ProductArray{TM, TData, TV}}) where {TM, TData, TV} = eltype(TData)

function allocate(x::ProductArray{ShapeSpec}) where ShapeSpec<:ShapeSpecification
    return ProductArray(ShapeSpec, allocate(x.data), x.reshapers)
end
function allocate(
    x::ProductArray{ShapeSpec},
    ::Type{T}
) where {ShapeSpec<:ShapeSpecification, T}
    return ProductArray(ShapeSpec, allocate(x.data, T), x.reshapers)
end

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

function number_eltype(x::ProductRepr)
    return typeof(reduce(+, one(number_eltype(eti)) for eti ∈ x.parts))
end

allocate(x::ProductRepr) = ProductRepr(map(allocate, submanifold_components(x))...)
function allocate(x::ProductRepr, ::Type{T}) where {T}
    return ProductRepr(map(t -> allocate(t, T), submanifold_components(x))...)
end

function copyto!(x::ProductRepr, y::ProductRepr)
    map(copyto!, submanifold_components(x), submanifold_components(y))
    return x
end

function (+)(v1::ProductRepr, v2::ProductRepr)
    return ProductRepr(map(+, submanifold_components(v1), submanifold_components(v2))...)
end

function (-)(v1::ProductRepr, v2::ProductRepr)
    return ProductRepr(map(-, submanifold_components(v1), submanifold_components(v2))...)
end
(-)(v::ProductRepr) = ProductRepr(map(-, submanifold_components(v)))

(*)(a::Number, v::ProductRepr) = ProductRepr(map(t -> a * t, submanifold_components(v)))

function Base.convert(::Type{TPR}, x::ProductRepr) where {TPR<:ProductRepr}
    return ProductRepr(map(
        t -> convert(t...),
        ziptuples(tuple(TPR.parameters[1].parameters...), submanifold_components(x)),
    ))
end
