
"""
    ziptuples(a, b)

Zips tuples `a` and `b` in a fast, type-stable way. If they have different
lengths, the result is trimmed to the length of the shorter tuple.
"""
@generated function ziptuples(a::NTuple{N,Any}, b::NTuple{M,Any}) where {N,M}
    ex = Expr(:tuple)
    for i = 1:min(N, M)
        push!(ex.args, :((a[$i], b[$i])))
    end
    ex
end

"""
    ziptuples(a, b, c)

Zips tuples `a`, `b` and `c` in a fast, type-stable way. If they have different
lengths, the result is trimmed to the length of the shorter tuple.
"""
@generated function ziptuples(a::NTuple{N,Any}, b::NTuple{M,Any}, c::NTuple{L,Any}) where {N,M,L}
    ex = Expr(:tuple)
    for i = 1:min(N, M, L)
        push!(ex.args, :((a[$i], b[$i], c[$i])))
    end
    ex
end


# conversion of SizedAbstractArray from StaticArrays.jl
"""
    SizedAbstractArray{Tuple{dims...}}(array)

Wraps an `Array` with a static size, so to take advantage of the (faster)
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

    function SizedAbstractArray{S, T, N, M, TData}(::UndefInitializer) where {S, T, N, M, TData<:AbstractArray}
        new{S, T, N, M, TData}(TData(undef, S.parameters...))
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
        a = SizedAbstractArray{S,T,N,M,TData}(undef)
        @inbounds $(Expr(:block, exprs...))
        return a
    end
end

@inline SizedAbstractArray{S,T,N}(x::Tuple) where {S,T,N} = SizedAbstractArray{S,T,N,N}(x)
@inline SizedAbstractArray{S,T}(x::Tuple) where {S,T} = SizedAbstractArray{S,T,StaticArrays.tuple_length(S),StaticArrays.tuple_length(S)}(x)
@inline SizedAbstractArray{S}(x::NTuple{L,T}) where {S,T,L} = SizedAbstractArray{S,T,StaticArrays.tuple_length(S),StaticArrays.tuple_length(S)}(x)

# Overide some problematic default behaviour
@inline convert(::Type{SA}, sa::SizedAbstractArray) where {SA<:SizedAbstractArray} = SA(sa.data)
@inline convert(::Type{SA}, sa::SA) where {SA<:SizedAbstractArray} = sa

# Back to Array (unfortunately need both convert and construct to overide other methods)
import Base.Array
@inline Array(sa::SizedAbstractArray) = Array(sa.data)
@inline Array{T}(sa::SizedAbstractArray{S,T}) where {T,S} = Array(sa.data)
@inline Array{T,N}(sa::SizedAbstractArray{S,T,N}) where {T,S,N} = Array(sa.data)

@inline convert(::Type{AbstractArray}, sa::SizedAbstractArray) = sa.data
@inline convert(::Type{AbstractArray{T}}, sa::SizedAbstractArray{S,T}) where {T,S} = sa.data
@inline convert(::Type{AbstractArray{T,N}}, sa::SizedAbstractArray{S,T,N}) where {T,S,N} = sa.data

import Base: getindex, setindex!
Base.@propagate_inbounds getindex(a::SizedAbstractArray, i::Int) = getindex(a.data, i)
Base.@propagate_inbounds setindex!(a::SizedAbstractArray, v, i::Int) = setindex!(a.data, v, i)

SizedVector{S,T,M} = SizedAbstractArray{Tuple{S},T,1,M}
@inline SizedVector{S}(a::AbstractArray{T,M}) where {S,T,M} = SizedAbstractArray{Tuple{S},T,1,M}(a)
@inline SizedVector{S}(x::NTuple{L,T}) where {S,T,L} = SizedAbstractArray{Tuple{S},T,1,1}(x)

SizedMatrix{S1,S2,T,M} = SizedAbstractArray{Tuple{S1,S2},T,2,M}
@inline SizedMatrix{S1,S2}(a::AbstractArray{T,M}) where {S1,S2,T,M} = SizedAbstractArray{Tuple{S1,S2},T,2,M}(a)
@inline SizedMatrix{S1,S2}(x::NTuple{L,T}) where {S1,S2,T,L} = SizedAbstractArray{Tuple{S1,S2},T,2,2}(x)

"""
    Size(dims)(array)

Creates a `SizedAbstractArray` wrapping `array` with the specified statically-known
`dims`, so to take advantage of the (faster) methods defined by the static array
package.
"""
(::Size{S})(a::AbstractArray) where {S} = SizedAbstractArray{Tuple{S...}}(a)

function promote_rule(::Type{<:SizedAbstractArray{S,T,N,M,TDataA}}, ::Type{<:SizedAbstractArray{S,U,N,M,TDataB}}) where {S,T,U,N,M,TDataA,TDataB}
    SizedAbstractArray{S,promote_type(T,U),N,M,T,promote_type(TDataA, TDataB)}
end
