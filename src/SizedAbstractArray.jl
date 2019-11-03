

# conversion of SizedArray from StaticArrays.jl
"""
    SizedAbstractArray{Tuple{dims...}}(array)

Wraps an `AbstractArray` with a static size, so to take advantage of the (faster)
methods defined by the static array package. The size is checked once upon
construction to determine if the number of elements (`length`) match, but the
array may be reshaped.
(Also, `Size(dims...)(array)` acheives the same thing)
"""
struct SizedAbstractArray{S<:Tuple, T, N, M, TData<:AbstractArray{T,M}} <: StaticArray{S, T, N}
    data::TData

    function SizedAbstractArray{S, T, N, M, TData}(a::TData) where {S, T, N, M, TData<:AbstractArray}
        if length(a) != StaticArrays.tuple_prod(S)
            error("Dimensions $(size(a)) don't match static size $S")
        end
        new{S,T,N,M,TData}(a)
    end

    function SizedAbstractArray{S, T, N, 1}(::UndefInitializer) where {S, T, N}
        new{S, T, N, 1, Array{T, 1}}(Array{T, 1}(undef, StaticArrays.tuple_prod(S)))
    end
    function SizedAbstractArray{S, T, N, N}(::UndefInitializer) where {S, T, N}
        new{S, T, N, N, Array{T, N}}(Array{T, N}(undef, size_to_tuple(S)...))
    end
end

@inline SizedAbstractArray{S,T,N}(a::TData) where {S,T,N,M,TData<:AbstractArray{T,M}} = SizedAbstractArray{S,T,N,M,TData}(a)
@inline SizedAbstractArray{S,T}(a::TData) where {S,T,M,TData<:AbstractArray{T,M}} = SizedAbstractArray{S,T,StaticArrays.tuple_length(S),M,TData}(a)
@inline SizedAbstractArray{S}(a::TData) where {S,T,M,TData<:AbstractArray{T,M}} = SizedAbstractArray{S,T,StaticArrays.tuple_length(S),M,TData}(a)

@inline SizedAbstractArray{S,T,N}(::UndefInitializer) where {S,T,N} = SizedAbstractArray{S,T,N,N}(undef)
@inline SizedAbstractArray{S,T}(::UndefInitializer) where {S,T} = SizedAbstractArray{S,T,StaticArrays.tuple_length(S),StaticArrays.tuple_length(S)}(undef)
@generated function (::Type{SizedAbstractArray{S,T,N,M,TData}})(x::NTuple{L,Any}) where {S,T,N,M,TData,L}
    if L != StaticArrays.tuple_prod(S)
        error("Dimension mismatch")
    end
    exprs = [:(a[$i] = x[$i]) for i = 1:L]
    return quote
        $(Expr(:meta, :inline))
        a = SizedAbstractArray{S,T,N,M}(undef)
        @inbounds $(Expr(:block, exprs...))
        return a
    end
end

@inline SizedAbstractArray{S,T,N}(x::Tuple) where {S,T,N} = SizedAbstractArray{S,T,N,N,Array{T,N}}(collect(x))
@inline SizedAbstractArray{S,T}(x::Tuple) where {S,T} = SizedAbstractArray{S,T,StaticArrays.tuple_length(S)}(x)
@inline SizedAbstractArray{S}(x::NTuple{L,T}) where {S,T,L} = SizedAbstractArray{S,T}(x)

# Overide some problematic default behaviour
@inline convert(::Type{SA}, sa::SizedAbstractArray) where {SA<:SizedAbstractArray} = SA(sa.data)
@inline convert(::Type{SA}, sa::SA) where {SA<:SizedAbstractArray} = sa

# Back to Array (unfortunately need both convert and construct to overide other methods)
import Base.Array
@inline Array(sa::SizedAbstractArray{S}) where {S} = Array(reshape(sa.data, size_to_tuple(S)))
@inline Array{T}(sa::SizedAbstractArray{S,T}) where {T,S} = Array(reshape(sa.data, size_to_tuple(S)))
@inline Array{T,N}(sa::SizedAbstractArray{S,T,N}) where {T,S,N} = Array(reshape(sa.data, size_to_tuple(S)))

@inline convert(::Type{Array}, sa::SizedAbstractArray{S}) where {S} = Array(reshape(sa.data, size_to_tuple(S)))
@inline convert(::Type{Array{T}}, sa::SizedAbstractArray{S,T}) where {T,S} = Array(reshape(sa.data, size_to_tuple(S)))
@inline convert(::Type{Array{T,N}}, sa::SizedAbstractArray{S,T,N}) where {T,S,N} = Array(reshape(sa.data, size_to_tuple(S)))

Base.@propagate_inbounds getindex(a::SizedAbstractArray, i::Int) = getindex(a.data, i)
Base.@propagate_inbounds setindex!(a::SizedAbstractArray, v, i::Int) = setindex!(a.data, v, i)

SizedAbstractVector{S,T,M} = SizedAbstractArray{Tuple{S},T,1,M}
@inline SizedAbstractVector{S}(a::TData) where {S,T,M,TData<:AbstractArray{T,M}} = SizedAbstractArray{Tuple{S},T,1,M,TData}(a)
@inline SizedAbstractVector{S}(x::NTuple{L,T}) where {S,T,L} = SizedAbstractArray{Tuple{S},T,1,1,Vector{T}}(x)

SizedAbstractMatrix{S1,S2,T,M} = SizedAbstractArray{Tuple{S1,S2},T,2,M}
@inline SizedAbstractMatrix{S1,S2}(a::TData) where {S1,S2,T,M,TData<:AbstractArray{T,M}} = SizedAbstractArray{Tuple{S1,S2},T,2,M,TData}(a)
@inline SizedAbstractMatrix{S1,S2}(x::NTuple{L,T}) where {S1,S2,T,L} = SizedAbstractArray{Tuple{S1,S2},T,2,2,Matrix{T}}(x)

Base.dataids(sa::SizedAbstractArray) = Base.dataids(sa.data)

function promote_rule(::Type{<:SizedAbstractArray{S,T,N,M,TDataA}}, ::Type{<:SizedAbstractArray{S,U,N,M,TDataB}}) where {S,T,U,N,M,TDataA,TDataB}
    TU = promote_type(T,U)
    SizedAbstractArray{S,TU,N,M,promote_type(TDataA, TDataB)::Type{<:AbstractArray{TU}}}
end

@inline copy(a::SizedAbstractArray) = typeof(a)(copy(a.data))

similar(::Type{<:SizedAbstractArray{S,T,N,M}},::Type{T2}) where {S,T,N,M,T2} = SizedAbstractArray{S,T2,N,M}(undef)
similar(::Type{SA},::Type{T},s::Size{S}) where {SA<:SizedAbstractArray,T,S} = sizedabstractarray_similar_type(T,s,StaticArrays.length_val(s))(undef)
sizedabstractarray_similar_type(::Type{T},s::Size{S},::Type{Val{D}}) where {T,S,D} = SizedAbstractArray{Tuple{S...},T,D,length(s)}

"""
    size_to_tuple(::Type{S}) where S<:Tuple

Converts a size given by `Tuple{N, M, ...}` into a tuple `(N, M, ...)`.
"""
Base.@pure function size_to_tuple(::Type{S}) where S<:Tuple
    return tuple(S.parameters...)
end
