@doc doc"""
    usinc(θ::Real)

Unnormalized version of `sinc` function, i.e.
$\operatorname{usinc}(\theta) = \frac{\sin(\theta)}{\theta}$. This is
equivalent to `sinc(θ/π)`.
"""
@inline usinc(θ::Real) = θ == 0 ? one(θ) : isinf(θ) ? zero(θ) : sin(θ) / θ

@doc doc"""
    usinc_from_cos(x::Real)

Unnormalized version of `sinc` function, i.e.
$\operatorname{usinc}(\theta) = \frac{\sin(\theta)}{\theta}$, computed from
$x = cos(\theta)$.
"""
@inline function usinc_from_cos(x::Real)
    if x >= 1
        return one(x)
    elseif x <= -1
        return zero(x)
    else
        return sqrt(1 - x^2) / acos(x)
    end
end

"""
    eigen_safe(x)

Compute the eigendecomposition of `x`. If `x` is a `StaticMatrix`, it is
converted to a `Matrix` before the decomposition.
"""
@inline eigen_safe(x; kwargs...) = eigen(x; kwargs...)

@inline function eigen_safe(x::StaticMatrix; kwargs...)
    s = size(x)
    E = eigen!(Matrix(parent(x)); kwargs...)
    return Eigen(SizedVector{s[1]}(E.values), SizedMatrix{s...}(E.vectors))
end

"""
    log_safe(x)

Compute the matrix logarithm of `x`. If `x` is a `StaticMatrix`, it is
converted to a `Matrix` before computing the log.
"""
@inline log_safe(x) = log(x)

@inline function log_safe(x::StaticMatrix)
    s = Size(x)
    return SizedMatrix{s[1],s[2]}(log(Matrix(parent(x))))
end

"""
    select_from_tuple(t::NTuple{N, Any}, positions::Val{P})

Selects elements of tuple `t` at positions specified by the second argument.
For example `select_from_tuple(("a", "b", "c"), Val((3, 1, 1)))` returns
`("c", "a", "a")`.
"""
@generated function select_from_tuple(t::NTuple{N, Any}, positions::Val{P}) where {N, P}
    for k in P
        (k < 0 || k > N) && error("positions must be between 1 and $N")
    end
    return Expr(:tuple, [Expr(:ref, :t, k) for k in P]...)
end

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

"""
    ziptuples(a, b, c, d)

Zips tuples `a`, `b`, `c` and `d` in a fast, type-stable way. If they have
different lengths, the result is trimmed to the length of the shorter tuple.
"""
@generated function ziptuples(a::NTuple{N,Any}, b::NTuple{M,Any}, c::NTuple{L,Any}, d::NTuple{K,Any}) where {N,M,L,K}
    ex = Expr(:tuple)
    for i = 1:min(N, M, L, K)
        push!(ex.args, :((a[$i], b[$i], c[$i], d[$i])))
    end
    ex
end

"""
    ziptuples(a, b, c, d, e)

Zips tuples `a`, `b`, `c`, `d` and `e` in a fast, type-stable way. If they have
different lengths, the result is trimmed to the length of the shorter tuple.
"""
@generated function ziptuples(a::NTuple{N,Any}, b::NTuple{M,Any}, c::NTuple{L,Any}, d::NTuple{K,Any}, e::NTuple{J,Any}) where {N,M,L,K,J}
    ex = Expr(:tuple)
    for i = 1:min(N, M, L, K, J)
        push!(ex.args, :((a[$i], b[$i], c[$i], d[$i], e[$i])))
    end
    ex
end

@generated function hasdynamic(::Type{Size}) where Size<:Tuple
    for s ∈ Size.parameters
        if isa(s, StaticArrays.Dynamic)
            return true
        end
    end
    return false
end

@generated function hasdynamic_val(::Type{Size}) where Size<:Tuple
    if hasdynamic(Size)
        return Val(:dynamic_true)
    else
        return Val(:dynamic_false)
    end
end

@generated function all_dynamic_fixed_val(::Type{Size}, inds::Union{Int, StaticArray{<:Tuple, Int}, Colon}...) where Size<:Tuple
    all_fixed = true
    for (i, param) in enumerate(Size.parameters)
        if isa(param, StaticArrays.Dynamic) && inds[i] == Colon
            all_fixed = false
            break
        end
    end

    if all_fixed
        return Val(:dynamic_fixed_true)
    else
        return Val(:dynamic_fixed_false)
    end
end

@generated function tuple_nodynamic_prod(::Type{Size}) where Size<:Tuple
    i = 1
    for s ∈ Size.parameters
        if !isa(s, StaticArrays.Dynamic)
            i *= s
        end
    end
    return i
end

# conversion of SizedArray from StaticArrays.jl
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
        dynamic_nodivisible = hasdynamic(S) && mod(length(a), tuple_nodynamic_prod(S)) != 0
        nodynamic_notequal = !hasdynamic(S) && length(a) != StaticArrays.tuple_prod(S)
        if nodynamic_notequal || dynamic_nodivisible
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

@inline function convert(::Type{SizedAbstractArray{S,T,N,M}}, a::TData) where {S,T,N,M,U,TData<:AbstractArray{U,M}}
    # TODO: dimension checking
    as = similar(a, T)
    copyto!(as, a)
    return SizedAbstractArray{S,T,N,M,typeof(as)}(as)
end

@inline function convert(::Type{SizedAbstractArray{S,T,N,M}}, a::TData) where {S,T,N,M,TData<:AbstractArray{T,M}}
    # TODO: dimension checking?
    return SizedAbstractArray{S,T,N,M,typeof(a)}(a)
end

@inline function convert(::Type{SizedAbstractArray{S,T,N}}, a::TData) where {S,T,N,M,U,TData<:AbstractArray{U,M}}
    # TODO: dimension checking
    as = similar(a, T)
    copyto!(as, a)
    return SizedAbstractArray{S,T,N,M,typeof(as)}(as)
end

@inline function convert(::Type{SizedAbstractArray{S,T,N}}, a::TData) where {S,T,N,M,TData<:AbstractArray{T,M}}
    # TODO: dimension checking?
    return SizedAbstractArray{S,T,N,M,typeof(a)}(a)
end

Base.@propagate_inbounds getindex(a::SizedAbstractArray, i::Int) = getindex(a.data, i)
Base.@propagate_inbounds setindex!(a::SizedAbstractArray, v, i::Int) = setindex!(a.data, v, i)

SizedAbstractVector{S,T,M} = SizedAbstractArray{Tuple{S},T,1,M}
@inline SizedAbstractVector{S}(a::TData) where {S,T,M,TData<:AbstractArray{T,M}} = SizedAbstractArray{Tuple{S},T,1,M,TData}(a)
@inline SizedAbstractVector{S}(x::NTuple{L,T}) where {S,T,L} = SizedAbstractArray{Tuple{S},T,1,1,Vector{T}}(x)

SizedAbstractMatrix{S1,S2,T,M} = SizedAbstractArray{Tuple{S1,S2},T,2,M}
@inline SizedAbstractMatrix{S1,S2}(a::TData) where {S1,S2,T,M,TData<:AbstractArray{T,M}} = SizedAbstractArray{Tuple{S1,S2},T,2,M,TData}(a)
@inline SizedAbstractMatrix{S1,S2}(x::NTuple{L,T}) where {S1,S2,T,L} = SizedAbstractArray{Tuple{S1,S2},T,2,2,Matrix{T}}(x)

Base.dataids(sa::SizedAbstractArray) = Base.dataids(sa.data)

@inline size(sa::SizedAbstractArray) = size(sa.data)
@inline length(sa::SizedAbstractArray) = length(sa.data)

@generated function _sized_abstract_array_axes(::Type{S}, ax::Tuple) where S<:Tuple
    exprs = Any[]
    map(enumerate(S.parameters)) do (i, si)
        if isa(si, StaticArrays.Dynamic)
            push!(exprs, :(ax[$i]))
        else
            push!(exprs, SOneTo(si))
        end
    end
    return Expr(:tuple, exprs...)
end

function axes(sa::SizedAbstractArray{S}) where S
    ax = axes(sa.data)
    return _sized_abstract_array_axes(S, ax)
end

@inline function getindex(sa::SizedAbstractArray{S}, ::Colon) where S
    return _getindex(hasdynamic_val(S), sa::StaticArray, :)
end

@inline function _getindex(::Val{:dynamic_true}, sa::SizedAbstractArray{S}, ::Colon) where S
    return SizedAbstractArray{S}(getindex(sa.data, :))
end

@inline function _getindex(::Val{:dynamic_false}, sa::SizedAbstractArray, ::Colon)
    _getindex(sa::StaticArray, Length(sa), :)
end

# disambiguation
Base.@propagate_inbounds function getindex(sa::SizedAbstractArray{S}, inds::Int...) where S
    return _getindex(hasdynamic_val(S), sa, inds...)
end

function _getindex(::Val{:dynamic_true}, sa::SizedAbstractArray, inds::Int...)
    return getindex(sa.data, inds...)
end

function _getindex(::Val{:dynamic_false}, sa::SizedAbstractArray, inds::Int...)
    @boundscheck checkbounds(sa, inds...)
    _getindex_scalar(Size(sa), sa, inds...)
end

Base.@propagate_inbounds function getindex(sa::SizedAbstractArray{S}, inds::Union{Int, StaticArray{<:Tuple, Int}, Colon}...) where S
    _getindex(hasdynamic_val(S), all_dynamic_fixed_val(S, inds...), sa, inds...)
end

function _getindex(::Val{:dynamic_true}, ::Val{:dynamic_fixed_true}, sa::SizedAbstractArray{S}, inds::Union{Int, StaticArray{<:Tuple, Int}, Colon}...) where S
    error("TODO")
end

@generated function new_out_size(::Type{Size}, inds...) where Size
    os = []
    map(Size.parameters, inds) do s, i
        if i == Int
        elseif i <: StaticVector
            push!(os, i.parameters[1].parameters[1])
        elseif i == Colon
            push!(os, s)
        else
            error("Unknown index type: $i")
        end
    end
    return Tuple{os...}
end


function _getindex(::Val{:dynamic_true}, ::Val{:dynamic_fixed_false}, sa::SizedAbstractArray{S}, inds::Union{Int, StaticArray{<:Tuple, Int}, Colon}...) where S
    newsize = new_out_size(S, inds...)
    return SizedAbstractArray{newsize}(getindex(sa.data, inds...))
end

function _getindex(::Val{:dynamic_false}, ::Val, sa::SizedAbstractArray, inds::Union{Int, StaticArray{<:Tuple, Int}, Colon}...)
    _getindex(sa, index_sizes(Size(sa), inds...), inds)
end

"""
    Size(dims)(array)

Creates a `SizedAbstractArray` wrapping `array` with the specified statically-known
`dims`, so to take advantage of the (faster) methods defined by the static array
package.
"""
(::Size{S})(a::AbstractArray) where {S} = SizedAbstractArray{Tuple{S...}}(a)

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
