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
