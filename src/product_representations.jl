
@doc raw"""
    submanifold_component(M::AbstractManifold, p, i::Integer)
    submanifold_component(M::AbstractManifold, p, ::Val(i)) where {i}
    submanifold_component(p, i::Integer)
    submanifold_component(p, ::Val(i)) where {i}

Project the product array `p` on `M` to its `i`th component. A new array is returned.
"""
submanifold_component(::Any...)
@inline function submanifold_component(M::AbstractManifold, p, i::Integer)
    return submanifold_component(M, p, Val(i))
end
@inline submanifold_component(M::AbstractManifold, p, i::Val) = submanifold_component(p, i)
@inline submanifold_component(p, ::Val{I}) where {I} = p.parts[I]
@inline submanifold_component(p::ArrayPartition, ::Val{I}) where {I} = p.x[I]
@inline submanifold_component(p, i::Integer) = submanifold_component(p, Val(i))

@doc raw"""
    submanifold_components(M::AbstractManifold, p)
    submanifold_components(p)

Get the projected components of `p` on the submanifolds of `M`. The components are returned in a Tuple.
"""
submanifold_components(::Any...)
@inline submanifold_components(M::AbstractManifold, p) = submanifold_components(p)
@inline submanifold_components(p) = p.parts
@inline submanifold_components(p::ArrayPartition) = p.x

function _show_component(io::IO, v; pre="", head="")
    sx = sprint(show, "text/plain", v, context=io, sizehint=0)
    sx = replace(sx, '\n' => "\n$(pre)")
    return print(io, head, pre, sx)
end

function _show_component_range(io::IO, vs, range; pre="", sym="Component ")
    for i in range
        _show_component(io, vs[i]; pre=pre, head="\n$(sym)$(i) =\n")
    end
    return nothing
end

function _show_product_repr(io::IO, x; name="Product representation", nmax=4)
    n = length(x.parts)
    print(io, "$(name) with $(n) submanifold component$(n == 1 ? "" : "s"):")
    half_nmax = div(nmax, 2)
    pre = "  "
    sym = " Component "
    if n ≤ nmax
        _show_component_range(io, x.parts, 1:n; pre=pre, sym=sym)
    else
        _show_component_range(io, x.parts, 1:half_nmax; pre=pre, sym=sym)
        print(io, "\n ⋮")
        _show_component_range(io, x.parts, (n - half_nmax + 1):n; pre=pre, sym=sym)
    end
    return nothing
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

Base.:(==)(x::ProductRepr, y::ProductRepr) = x.parts == y.parts

@inline function number_eltype(x::ProductRepr)
    @inline eti_to_one(eti) = one(number_eltype(eti))
    return typeof(sum(map(eti_to_one, x.parts)))
end

allocate(x::ProductRepr) = ProductRepr(map(allocate, submanifold_components(x))...)
function allocate(x::ProductRepr, ::Type{T}) where {T}
    return ProductRepr(map(t -> allocate(t, T), submanifold_components(x))...)
end
allocate(p::ProductRepr, ::Type{T}, s::Size{S}) where {S,T} = Vector{T}(undef, S)
allocate(p::ProductRepr, ::Type{T}, s::Integer) where {S,T} = Vector{T}(undef, s)

Base.copy(x::ProductRepr) = ProductRepr(map(copy, x.parts))

function Base.copyto!(x::ProductRepr, y::ProductRepr)
    map(copyto!, submanifold_components(x), submanifold_components(y))
    return x
end

function Base.:+(v1::ProductRepr, v2::ProductRepr)
    return ProductRepr(map(+, submanifold_components(v1), submanifold_components(v2))...)
end

function Base.:-(v1::ProductRepr, v2::ProductRepr)
    return ProductRepr(map(-, submanifold_components(v1), submanifold_components(v2))...)
end
Base.:-(v::ProductRepr) = ProductRepr(map(-, submanifold_components(v)))

Base.:*(a::Number, v::ProductRepr) = ProductRepr(map(t -> a * t, submanifold_components(v)))
Base.:*(v::ProductRepr, a::Number) = ProductRepr(map(t -> t * a, submanifold_components(v)))

Base.:/(v::ProductRepr, a::Number) = ProductRepr(map(t -> t / a, submanifold_components(v)))

function Base.convert(::Type{ProductRepr{TPR}}, x::ProductRepr) where {TPR<:Tuple}
    if @isdefined TPR
        return ProductRepr(convert(TPR, submanifold_components(x)))
    else
        return x
    end
end

function Base.show(io::IO, ::MIME"text/plain", x::ProductRepr)
    return _show_product_repr(io, x; name="ProductRepr")
end

ManifoldsBase._get_vector_cache_broadcast(::ProductRepr) = Val(false)

# Tuple-like broadcasting of ProductRepr

function Broadcast.BroadcastStyle(::Type{<:ProductRepr})
    return Broadcast.Style{ProductRepr}()
end
function Broadcast.BroadcastStyle(
    ::Broadcast.AbstractArrayStyle{0},
    b::Broadcast.Style{ProductRepr},
)
    return b
end

Broadcast.instantiate(bc::Broadcast.Broadcasted{Broadcast.Style{ProductRepr},Nothing}) = bc
function Broadcast.instantiate(bc::Broadcast.Broadcasted{Broadcast.Style{ProductRepr}})
    Broadcast.check_broadcast_axes(bc.axes, bc.args...)
    return bc
end

Broadcast.broadcastable(v::ProductRepr) = v

@inline function Base.copy(bc::Broadcast.Broadcasted{Broadcast.Style{ProductRepr}})
    dim = axes(bc)
    length(dim) == 1 || throw(DimensionMismatch("ProductRepr only supports one dimension"))
    N = length(dim[1])
    return ProductRepr(ntuple(k -> @inbounds(Broadcast._broadcast_getindex(bc, k)), Val(N)))
end

Base.@propagate_inbounds Broadcast._broadcast_getindex(v::ProductRepr, I) = v.parts[I[1]]

Base.axes(v::ProductRepr) = axes(v.parts)

@inline function Base.copyto!(
    dest::ProductRepr,
    bc::Broadcast.Broadcasted{Broadcast.Style{ProductRepr}},
)
    axes(dest) == axes(bc) || Broadcast.throwdm(axes(dest), axes(bc))
    # Performance optimization: broadcast!(identity, dest, A) is equivalent to copyto!(dest, A) if indices match
    if bc.f === identity && bc.args isa Tuple{ProductRepr} # only a single input argument to broadcast!
        A = bc.args[1]
        if axes(dest) == axes(A)
            return copyto!(dest, A)
        end
    end
    bc′ = Broadcast.preprocess(dest, bc)
    # Performance may vary depending on whether `@inbounds` is placed outside the
    # for loop or not. (cf. https://github.com/JuliaLang/julia/issues/38086)
    @inbounds @simd for I in eachindex(bc′)
        copyto!(dest.parts[I], bc′[I])
    end
    return dest
end

## ArrayPartition

ManifoldsBase._get_vector_cache_broadcast(::ArrayPartition) = Val(false)
