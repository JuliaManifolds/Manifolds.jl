@generated function hasdynamic(::Type{Size}) where Size<:Tuple
    for s ∈ Size.parameters
        if isa(s, StaticArrays.Dynamic)
            return true
        end
    end
    return false
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
    HybridArray{Tuple{dims...}}(array)

Wraps an `AbstractArray` with a static size, so to take advantage of the (faster)
methods defined by the static array package. The size is checked once upon
construction to determine if the number of elements (`length`) match, but the
array may be reshaped.
(Also, `Size(dims...)(array)` acheives the same thing)
"""
struct HybridArray{S<:Tuple, T, N, M, TData<:AbstractArray{T,M}} <: AbstractArray{T, N}
    data::TData

    function HybridArray{S, T, N, M, TData}(a::TData) where {S, T, N, M, TData<:AbstractArray{T,M}}
        tnp = tuple_nodynamic_prod(S)
        lena = length(a)
        dynamic_nodivisible = hasdynamic(S) && tnp != 0 && mod(lena, tnp) != 0
        nodynamic_notequal = !hasdynamic(S) && lena != StaticArrays.tuple_prod(S)
        if nodynamic_notequal || dynamic_nodivisible || (tnp == 0 && lena != 0)
            error("Dimensions $(size(a)) don't match static size $S")
        end
        new{S,T,N,M,TData}(a)
    end

    function HybridArray{S, T, N, 1}(::UndefInitializer) where {S, T, N}
        new{S, T, N, 1, Array{T, 1}}(Array{T, 1}(undef, StaticArrays.tuple_prod(S)))
    end
    function HybridArray{S, T, N, N}(::UndefInitializer) where {S, T, N}
        new{S, T, N, N, Array{T, N}}(Array{T, N}(undef, size_to_tuple(S)...))
    end
end

@inline HybridArray{S,T,N}(a::TData) where {S,T,N,M,TData<:AbstractArray{T,M}} = HybridArray{S,T,N,M,TData}(a)
@inline HybridArray{S,T}(a::TData) where {S,T,M,TData<:AbstractArray{T,M}} = HybridArray{S,T,StaticArrays.tuple_length(S),M,TData}(a)
@inline HybridArray{S}(a::TData) where {S,T,M,TData<:AbstractArray{T,M}} = HybridArray{S,T,StaticArrays.tuple_length(S),M,TData}(a)

@inline HybridArray{S,T,N}(::UndefInitializer) where {S,T,N} = HybridArray{S,T,N,N}(undef)
@inline HybridArray{S,T}(::UndefInitializer) where {S,T} = HybridArray{S,T,StaticArrays.tuple_length(S),StaticArrays.tuple_length(S)}(undef)
@generated function (::Type{HybridArray{S,T,N,M,TData}})(x::NTuple{L,Any}) where {S,T,N,M,TData,L}
    if L != StaticArrays.tuple_prod(S)
        error("Dimension mismatch")
    end
    exprs = [:(a[$i] = x[$i]) for i = 1:L]
    return quote
        $(Expr(:meta, :inline))
        a = HybridArray{S,T,N,M}(undef)
        @inbounds $(Expr(:block, exprs...))
        return a
    end
end

@inline HybridArray{S,T,N}(x::Tuple) where {S,T,N} = HybridArray{S,T,N,N,Array{T,N}}(x)
@inline HybridArray{S,T}(x::Tuple) where {S,T} = HybridArray{S,T,StaticArrays.tuple_length(S)}(x)
@inline HybridArray{S}(x::NTuple{L,T}) where {S,T,L} = HybridArray{S,T}(x)

Base.@propagate_inbounds (::Type{HybridArray{S,T,N,M,TData}})(a::AbstractArray) where {S,T,N,M,TData<:AbstractArray{<:Any,M}} = convert(HybridArray{S,T,N,M,TData}, a)
Base.@propagate_inbounds (::Type{HybridArray{S,T,N,M}})(a::AbstractArray) where {S,T,N,M} = convert(HybridArray{S,T,N,M}, a)
Base.@propagate_inbounds (::Type{HybridArray{S,T,N}})(a::AbstractArray) where {S,T,N} = convert(HybridArray{S,T,N}, a)
Base.@propagate_inbounds (::Type{HybridArray{S,T}})(a::AbstractArray) where {S,T} = convert(HybridArray{S,T,StaticArrays.tuple_length(S)}, a)

# Overide some problematic default behaviour
@inline convert(::Type{SA}, sa::HybridArray) where {SA<:HybridArray} = SA(sa.data)
@inline convert(::Type{SA}, sa::SA) where {SA<:HybridArray} = sa

# Back to Array (unfortunately need both convert and construct to overide other methods)
import Base.Array
@inline Array(sa::HybridArray{S}) where {S} = Array(reshape(sa.data, size_to_tuple(S)))
@inline Array{T}(sa::HybridArray{S,T}) where {T,S} = Array(reshape(sa.data, size_to_tuple(S)))
@inline Array{T,N}(sa::HybridArray{S,T,N}) where {T,S,N} = Array(reshape(sa.data, size_to_tuple(S)))

@inline convert(::Type{Array}, sa::HybridArray{S}) where {S} = Array(reshape(sa.data, size_to_tuple(S)))
@inline convert(::Type{Array{T}}, sa::HybridArray{S,T}) where {T,S} = Array(reshape(sa.data, size_to_tuple(S)))
@inline convert(::Type{Array{T,N}}, sa::HybridArray{S,T,N}) where {T,S,N} = Array(reshape(sa.data, size_to_tuple(S)))

@inline function convert(::Type{HybridArray{S,T,N,M,TData}}, a::AbstractArray) where {S,T,N,M,U,TData<:AbstractArray{U,M}}
    # TODO: dimension checking
    as = convert(TData, a)
    return HybridArray{S,T,N,M,TData}(as)
end

@inline function convert(::Type{HybridArray{S,T,N,M}}, a::TData) where {S,T,N,M,U,TData<:AbstractArray{U,M}}
    # TODO: dimension checking
    as = similar(a, T)
    copyto!(as, a)
    return HybridArray{S,T,N,M,typeof(as)}(as)
end

@inline function convert(::Type{HybridArray{S,T,N,M}}, a::TData) where {S,T,N,M,TData<:AbstractArray{T,M}}
    # TODO: dimension checking?
    return HybridArray{S,T,N,M,typeof(a)}(a)
end

@inline function convert(::Type{HybridArray{S,T,N}}, a::TData) where {S,T,N,M,U,TData<:AbstractArray{U,M}}
    # TODO: dimension checking
    as = similar(a, T)
    copyto!(as, a)
    return HybridArray{S,T,N,M,typeof(as)}(as)
end

@inline function convert(::Type{HybridArray{S,T,N}}, a::TData) where {S,T,N,M,TData<:AbstractArray{T,M}}
    # TODO: dimension checking?
    return HybridArray{S,T,N,M,typeof(a)}(a)
end

HybridVector{S,T,M} = HybridArray{Tuple{S},T,1,M}
@inline HybridVector{S}(a::TData) where {S,T,M,TData<:AbstractArray{T,M}} = HybridArray{Tuple{S},T,1,M,TData}(a)
@inline HybridVector{S}(x::NTuple{L,T}) where {S,T,L} = HybridArray{Tuple{S},T,1,1,Vector{T}}(x)

HybridMatrix{S1,S2,T,M} = HybridArray{Tuple{S1,S2},T,2,M}
@inline HybridMatrix{S1,S2}(a::TData) where {S1,S2,T,M,TData<:AbstractArray{T,M}} = HybridArray{Tuple{S1,S2},T,2,M,TData}(a)
@inline HybridMatrix{S1,S2}(x::NTuple{L,T}) where {S1,S2,T,L} = HybridArray{Tuple{S1,S2},T,2,2,Matrix{T}}(x)

Base.dataids(sa::HybridArray) = Base.dataids(sa.data)

@inline size(sa::HybridArray{S}) where S = size(sa.data)

@inline length(sa::HybridArray) = length(sa.data)

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

function axes(sa::HybridArray{S}) where S
    ax = axes(sa.data)
    return _sized_abstract_array_axes(S, ax)
end

@inline function getindex(sa::HybridArray{S}, ::Colon) where S
    return HybridArray{S}(getindex(sa.data, :))
end

Base.@propagate_inbounds function getindex(sa::HybridArray{S}, inds::Int...) where S
    return getindex(sa.data, inds...)
end

Base.@propagate_inbounds function getindex(sa::HybridArray{S}, inds::Union{Int, StaticArray{<:Tuple, Int}, Colon}...) where S
    _getindex(all_dynamic_fixed_val(S, inds...), sa, inds...)
end

Base.@propagate_inbounds function _getindex(::Val{:dynamic_fixed_true}, sa::HybridArray, inds::Union{Int, StaticArray{<:Tuple, Int}, Colon}...)
    return _getindex_all_static(sa, inds...)
end

function _get_indices(i::Tuple{}, j::Int)
    return ()
end

function _get_indices(i::Tuple, j::Int, i1::Type{Int}, inds...)
    return (:(inds[$j]), _get_indices(i, j+1, inds...)...)
end

function _get_indices(i::Tuple, j::Int, i1::Type{T}, inds...) where T<:StaticArray{<:Tuple, Int}
    return (:(inds[$j][$(i[1])]), _get_indices(i[2:end], j+1, inds...)...)
end

function _get_indices(i::Tuple, j::Int, i1::Type{Colon}, inds...)
    return (i[1], _get_indices(i[2:end], j+1, inds...)...)
end

@generated function _getindex_all_static(sa::HybridArray{S,T}, inds::Union{Int, StaticArray{<:Tuple, Int}, Colon}...) where {S,T}
    newsize = new_out_size_nongen(S, inds...)
    exprs = Vector{Expr}(undef, length(newsize))

    indices = CartesianIndices(newsize)
    exprs = similar(indices, Expr)
    for current_ind ∈ indices
        cinds = _get_indices(current_ind.I, 1, inds...)
        exprs[current_ind.I...] = :(getindex(sa.data, $(cinds...)))
    end
    Tnewsize = Tuple{newsize...}
    return quote
        Base.@_propagate_inbounds_meta
        SArray{$Tnewsize,$T}(tuple($(exprs...)))
    end
end

function new_out_size_nongen(::Type{Size}, inds...) where Size
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
    return tuple(os...)
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

@inline function _getindex(::Val{:dynamic_fixed_false}, sa::HybridArray{S}, inds::Union{Int, StaticArray{<:Tuple, Int}, Colon}...) where S
    newsize = new_out_size(S, inds...)
    return HybridArray{newsize}(getindex(sa.data, inds...))
end

# setindex stuff

import StaticArrays._setindex!_scalar

Base.@propagate_inbounds function setindex!(a::HybridArray, value, inds::Int...)
    Base.@boundscheck checkbounds(a, inds...)
    _setindex!_scalar(Size(a), a, value, inds...)
end

@generated function _setindex!_scalar(::Size{S}, a::HybridArray, value, inds::Int...) where S
    if length(inds) == 0
        return quote
            @_propagate_inbounds_meta
            a[1] = value
        end
    end

    stride = :(1)
    ind_expr = :()
    for i ∈ 1:length(inds)
        if i == 1
            ind_expr = :(inds[1])
        else
            ind_expr = :($ind_expr + $stride * (inds[$i] - 1))
        end
        if isa(S[i], StaticArrays.Dynamic)
            stride = :($stride * size(a.data, $i))
        else
            stride = :($stride * $(S[i]))
        end
    end
    return quote
        Base.@_inline_meta
        Base.@_propagate_inbounds_meta
        a.data[$ind_expr] = value
    end
end

function promote_rule(::Type{<:HybridArray{S,T,N,M,TDataA}}, ::Type{<:HybridArray{S,U,N,M,TDataB}}) where {S,T,U,N,M,TDataA,TDataB}
    TU = promote_type(T,U)
    HybridArray{S,TU,N,M,promote_type(TDataA, TDataB)::Type{<:AbstractArray{TU}}}
end

@inline copy(a::HybridArray) = typeof(a)(copy(a.data))

similar(::Type{<:HybridArray{S,T,N,M}},::Type{T2}) where {S,T,N,M,T2} = HybridArray{S,T2,N,M}(undef)
similar(::Type{SA},::Type{T},s::Size{S}) where {SA<:HybridArray,T,S} = hybridabstractarray_similar_type(T,s,StaticArrays.length_val(s))(undef)
hybridabstractarray_similar_type(::Type{T},s::Size{S},::Type{Val{D}}) where {T,S,D} = HybridArray{Tuple{S...},T,D,length(s)}

Size(::Type{<:HybridArray{S}}) where {S} = Size(S)


###############################
# BROADCAST
###############################

import Base.Broadcast: BroadcastStyle
using Base.Broadcast: AbstractArrayStyle, Broadcasted, DefaultArrayStyle

# Add a new BroadcastStyle for StaticArrays, derived from AbstractArrayStyle
# A constructor that changes the style parameter N (array dimension) is also required
struct HybridArrayStyle{N} <: AbstractArrayStyle{N} end
HybridArrayStyle{M}(::Val{N}) where {M,N} = HybridArrayStyle{N}()
BroadcastStyle(::Type{<:HybridArray{<:Tuple, <:Any, N}}) where {N} = HybridArrayStyle{N}()
# Precedence rules
BroadcastStyle(::HybridArray{M}, ::DefaultArrayStyle{N}) where {M,N} =
    DefaultArrayStyle(Val(max(M, N)))
BroadcastStyle(::HybridArray{M}, ::DefaultArrayStyle{0}) where {M} =
    HybridArrayStyle{M}()

BroadcastStyle(::HybridArray{M}, ::StaticArrays.StaticArrayStyle{N}) where {M,N} =
    StaticArrays.Hybrid(Val(max(M, N)))
BroadcastStyle(::HybridArray{M}, ::StaticArrays.StaticArrayStyle{0}) where {M} =
    HybridArrayStyle{M}()

# copy overload
@inline function Base.copy(B::Broadcasted{HybridArrayStyle{M}}) where M
    flat = Broadcast.flatten(B); as = flat.args; f = flat.f
    argsizes = StaticArrays.broadcast_sizes(as...)
    destsize = StaticArrays.combine_sizes(argsizes)
    if Length(destsize) === Length{StaticArrays.Dynamic()}()
        # destination dimension cannot be determined statically; fall back to generic broadcast
        return HybridArray{StaticArrays.size_tuple(destsize)}(copy(convert(Broadcasted{DefaultArrayStyle{M}}, B)))
    end
    _broadcast(f, destsize, argsizes, as...)
end
# copyto! overloads
@inline Base.copyto!(dest, B::Broadcasted{<:HybridArrayStyle}) = _copyto!(dest, B)
@inline Base.copyto!(dest::AbstractArray, B::Broadcasted{<:HybridArrayStyle}) = _copyto!(dest, B)
@inline function _copyto!(dest, B::Broadcasted{HybridArrayStyle{M}}) where M
    flat = Broadcast.flatten(B); as = flat.args; f = flat.f
    argsizes = StaticArrays.broadcast_sizes(as...)
    destsize = StaticArrays.combine_sizes((Size(dest), argsizes...))
    if Length(destsize) === Length{StaticArrays.Dynamic()}()
        # destination dimension cannot be determined statically; fall back to generic broadcast!
        return copyto!(dest, convert(Broadcasted{DefaultArrayStyle{M}}, B))
    end
    StaticArrays._broadcast!(f, destsize, dest, argsizes, as...)
end


###############################
# LINEAR ALGEBRA
###############################

const HybridMatrixLike{T} = Union{
    HybridMatrix{<:Any, <:Any, T},
    Transpose{T, <:HybridMatrix{T}},
    Adjoint{T, <:HybridMatrix{T}},
    Symmetric{T, <:HybridMatrix{T}},
    Hermitian{T, <:HybridMatrix{T}},
    Diagonal{T, <:HybridMatrix{<:Any, T}}
}

const HybridVecOrMatLike{T} = Union{HybridVector{<:Any, T}, HybridMatrixLike{T}}

# Binary ops
# Between arrays
@inline +(a::HybridArray, b::HybridArray) = a .+ b
@inline +(a::AbstractArray, b::HybridArray) = a .+ b
@inline +(a::HybridArray, b::AbstractArray) = a .+ b

@inline -(a::HybridArray, b::HybridArray) = a .- b
@inline -(a::AbstractArray, b::HybridArray) = a .- b
@inline -(a::HybridArray, b::AbstractArray) = a .- b

# Scalar-array
@inline *(a::Number, b::HybridArray) = a .* b
@inline *(a::HybridArray, b::Number) = a .* b

@inline /(a::HybridArray, b::Number) = a ./ b
@inline \(a::Number, b::HybridArray) = a .\ b

@inline vcat(a::HybridVecOrMatLike) = a
@inline vcat(a::HybridVecOrMatLike, b::HybridVecOrMatLike) = _vcat(Size(a), Size(b), a, b)
@inline vcat(a::HybridVecOrMatLike, b::HybridVecOrMatLike, c::HybridVecOrMatLike...) = vcat(vcat(a,b), vcat(c...))

@generated function _vcat(::Size{Sa}, ::Size{Sb}, a::HybridVecOrMatLike, b::HybridVecOrMatLike) where {Sa, Sb}
    if Size(Sa)[2] != Size(Sb)[2]
        throw(DimensionMismatch("Tried to vcat arrays of size $Sa and $Sb"))
    end

    if a <: HybridVector && b <: HybridVector
        Snew = (Sa[1] + Sb[1],)
    else
        Snew = (Sa[1] + Sb[1], Size(Sa)[2])
    end

    return quote
        Base.@_inline_meta
        @inbounds return similar_type(a, promote_type(eltype(a), eltype(b)), Size($Snew))(hcat(a.data, b.data))
    end
end

@inline hcat(a::HybridVecOrMatLike, b::HybridVecOrMatLike) = _hcat(Size(a), Size(b), a, b)
@inline hcat(a::HybridVecOrMatLike, b::HybridVecOrMatLike, c::HybridVecOrMatLike...) = hcat(hcat(a,b), hcat(c...))

@generated function _hcat(::Size{Sa}, ::Size{Sb}, a::HybridVecOrMatLike, b::HybridVecOrMatLike) where {Sa, Sb}
    if Sa[1] != Sb[1]
        throw(DimensionMismatch("Tried to hcat arrays of size $Sa and $Sb"))
    end

    Snew = (Sa[1], Size(Sa)[2] + Size(Sb)[2])

    return quote
        Base.@_inline_meta
        return similar_type(a, promote_type(eltype(a), eltype(b)), Size($Snew))(hcat(a.data, b.data))
    end
end

@generated function _broadcast(f, ::Size{newsize}, s::Tuple{Vararg{Size}}, a...) where newsize
    first_staticarray = 0
    for i = 1:length(a)
        if a[i] <: StaticArray
            first_staticarray = a[i]
            break
        end
    end
    if first_staticarray == 0
        for i = 1:length(a)
            if a[i] <: HybridArray
                first_staticarray = a[i]
                break
            end
        end
    end

    exprs = Array{Expr}(undef, newsize)
    more = prod(newsize) > 0
    current_ind = ones(Int, length(newsize))
    sizes = [sz.parameters[1] for sz ∈ s.parameters]

    make_expr(i) = begin
        if !(a[i] <: AbstractArray)
            return :(StaticArrays.scalar_getindex(a[$i]))
        elseif hasdynamic(Tuple{sizes[i]...})
            return :(a[$i][$(current_ind...)])
        else
            :(a[$i][$(StaticArrays.broadcasted_index(sizes[i], current_ind))])
        end
    end

    while more
        exprs_vals = [make_expr(i) for i = 1:length(sizes)]
        exprs[current_ind...] = :(f($(exprs_vals...)))

        # increment current_ind (maybe use CartesianIndices?)
        current_ind[1] += 1
        for i ∈ 1:length(newsize)
            if current_ind[i] > newsize[i]
                if i == length(newsize)
                    more = false
                    break
                else
                    current_ind[i] = 1
                    current_ind[i+1] += 1
                end
            else
                break
            end
        end
    end

    return quote
        Base.@_inline_meta
        @inbounds elements = tuple($(exprs...))
        @inbounds return similar_type($first_staticarray, eltype(elements), Size(newsize))(elements)
    end
end
