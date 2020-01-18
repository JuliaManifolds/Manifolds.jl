

# conversion of SizedArray from StaticArrays.jl
"""
    SizedAbstractArray{Tuple{dims...}}(array)

Wraps an `AbstractArray` with a static size, so to take advantage of the (faster)
methods defined by the static array package. The size is checked once upon
construction to determine if the number of elements (`length`) match, but the
array may be reshaped.
"""
struct SizedAbstractArray{S<:Tuple,T,N,M,TData<:AbstractArray{T,M}} <: StaticArray{S,T,N}
    data::TData

    function SizedAbstractArray{S,T,N,M,TData}(
        a::TData,
    ) where {S,T,N,M,TData<:AbstractArray}
        if length(a) != StaticArrays.tuple_prod(S)
            error("Dimensions $(size(a)) don't match static size $S")
        end
        return new{S,T,N,M,TData}(a)
    end

    function SizedAbstractArray{S,T,N,1}(::UndefInitializer) where {S,T,N}
        return new{S,T,N,1,Array{T,1}}(Array{T,1}(undef, StaticArrays.tuple_prod(S)))
    end
    function SizedAbstractArray{S,T,N,N}(::UndefInitializer) where {S,T,N}
        return new{S,T,N,N,Array{T,N}}(Array{T,N}(undef, size_to_tuple(S)...))
    end
end

@inline function SizedAbstractArray{S,T,N}(
    a::TData,
) where {S,T,N,M,TData<:AbstractArray{T,M}}
    return SizedAbstractArray{S,T,N,M,TData}(a)
end
@inline function SizedAbstractArray{S,T}(a::TData) where {S,T,M,TData<:AbstractArray{T,M}}
    return SizedAbstractArray{S,T,StaticArrays.tuple_length(S),M,TData}(a)
end
@inline function SizedAbstractArray{S}(a::TData) where {S,T,M,TData<:AbstractArray{T,M}}
    return SizedAbstractArray{S,T,StaticArrays.tuple_length(S),M,TData}(a)
end

@inline function SizedAbstractArray{S,T,N}(::UndefInitializer) where {S,T,N}
    return SizedAbstractArray{S,T,N,N}(undef)
end
@inline function SizedAbstractArray{S,T}(::UndefInitializer) where {S,T}
    return SizedAbstractArray{
        S,
        T,
        StaticArrays.tuple_length(S),
        StaticArrays.tuple_length(S),
    }(
        undef,
    )
end
@generated function (::Type{SizedAbstractArray{S,T,N,M,TData}})(
    x::NTuple{L,Any},
) where {S,T,N,M,TData,L}
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

@inline function SizedAbstractArray{S,T,N}(x::Tuple) where {S,T,N}
    return SizedAbstractArray{S,T,N,N,Array{T,N}}(collect(x))
end
@inline function SizedAbstractArray{S,T}(x::Tuple) where {S,T}
    return SizedAbstractArray{S,T,StaticArrays.tuple_length(S)}(x)
end
@inline function SizedAbstractArray{S}(x::NTuple{L,T}) where {S,T,L}
    return SizedAbstractArray{S,T}(x)
end

# Overide some problematic default behaviour
@inline function convert(::Type{SA}, sa::SizedAbstractArray) where {SA<:SizedAbstractArray}
    return SA(sa.data)
end
@inline convert(::Type{SA}, sa::SA) where {SA<:SizedAbstractArray} = sa

# Back to Array (unfortunately need both convert and construct to overide other methods)
import Base.Array
@inline function Array(sa::SizedAbstractArray{S}) where {S}
    return Array(reshape(sa.data, size_to_tuple(S)))
end
@inline function Array{T}(sa::SizedAbstractArray{S,T}) where {T,S}
    return Array(reshape(sa.data, size_to_tuple(S)))
end
@inline function Array{T,N}(sa::SizedAbstractArray{S,T,N}) where {T,S,N}
    return Array(reshape(sa.data, size_to_tuple(S)))
end

@inline function convert(::Type{Array}, sa::SizedAbstractArray{S}) where {S}
    return Array(reshape(sa.data, size_to_tuple(S)))
end
@inline function convert(::Type{Array{T}}, sa::SizedAbstractArray{S,T}) where {T,S}
    return Array(reshape(sa.data, size_to_tuple(S)))
end
@inline function convert(::Type{Array{T,N}}, sa::SizedAbstractArray{S,T,N}) where {T,S,N}
    return Array(reshape(sa.data, size_to_tuple(S)))
end

Base.@propagate_inbounds getindex(a::SizedAbstractArray, i::Int) = getindex(a.data, i)
Base.@propagate_inbounds function setindex!(a::SizedAbstractArray, v, i::Int)
    return setindex!(a.data, v, i)
end

SizedAbstractVector{S,T,M} = SizedAbstractArray{Tuple{S},T,1,M}
@inline function SizedAbstractVector{S}(a::TData) where {S,T,M,TData<:AbstractArray{T,M}}
    return SizedAbstractArray{Tuple{S},T,1,M,TData}(a)
end
@inline function SizedAbstractVector{S}(x::NTuple{L,T}) where {S,T,L}
    return SizedAbstractArray{Tuple{S},T,1,1,Vector{T}}(x)
end

SizedAbstractMatrix{S1,S2,T,M} = SizedAbstractArray{Tuple{S1,S2},T,2,M}
@inline function SizedAbstractMatrix{S1,S2}(
    a::TData,
) where {S1,S2,T,M,TData<:AbstractArray{T,M}}
    return SizedAbstractArray{Tuple{S1,S2},T,2,M,TData}(a)
end
@inline function SizedAbstractMatrix{S1,S2}(x::NTuple{L,T}) where {S1,S2,T,L}
    return SizedAbstractArray{Tuple{S1,S2},T,2,2,Matrix{T}}(x)
end

Base.dataids(sa::SizedAbstractArray) = Base.dataids(sa.data)

function promote_rule(
    ::Type{<:SizedAbstractArray{S,T,N,M,TDataA}},
    ::Type{<:SizedAbstractArray{S,U,N,M,TDataB}},
) where {S,T,U,N,M,TDataA,TDataB}
    TU = promote_type(T, U)
    SizedAbstractArray{S,TU,N,M,promote_type(TDataA, TDataB)::Type{<:AbstractArray{TU}}}
end

@inline copy(a::SizedAbstractArray) = typeof(a)(copy(a.data))

function similar(::Type{<:SizedAbstractArray{S,T,N,M}}, ::Type{T2}) where {S,T,N,M,T2}
    return SizedAbstractArray{S,T2,N,M}(undef)
end
function similar(::Type{SA}, ::Type{T}, s::Size{S}) where {SA<:SizedAbstractArray,T,S}
    return sizedabstractarray_similar_type(T, s, StaticArrays.length_val(s))(undef)
end
function sizedabstractarray_similar_type(
    ::Type{T},
    s::Size{S},
    ::Type{Val{D}},
) where {T,S,D}
    return SizedAbstractArray{Tuple{S...},T,D,length(s)}
end
