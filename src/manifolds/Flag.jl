struct ZeroTuple{TupT}
    x::TupT
end

ZeroTuple(x::Tuple) = ZeroTuple{typeof(x)}(x)

function Base.getindex(t::ZeroTuple, i::Int)
    if i == 0
        return 0
    else
        return t.x[i]
    end
end

@doc raw"""
    Flag{N,d} <: AbstractDecoratorManifold{ℝ}

Flag manifold of ``d`` subspaces of ``ℝ^N``[^YeWongLim2022]. By default the manifold uses
the orthogonal coordinates representation.

Tangent space is represented in the block-skew-symmetric form.

# Constructor

    Flag(N, n1, n2, ..., nd)

Generate the manifold ``\operatorname{Flag}(n_1, n_2, ..., n_d; N)`` of subspaces
```math
𝕍_1 ⊆ 𝕍_2 ⊆ ⋯ ⊆ V_d, \quad \operatorname{dim}(𝕍_i) = n_i
```
where ``𝕍_i`` for ``i ∈ 1, 2, …, d`` are subspaces of ``ℝ^N`` of dimension
``\operatorname{dim} 𝕍_i = n_i``.


[^YeWongLim2022]:
    > K. Ye, K. S.-W. Wong, and L.-H. Lim, “Optimization on flag manifolds,” Math. Program.,
    > vol. 194, no. 1, pp. 621–660, Jul. 2022,
    > doi: [10.1007/s10107-021-01640-3](https://doi.org/10.1007/s10107-021-01640-3).
"""
struct Flag{N,dp1} <: AbstractDecoratorManifold{ℝ}
    subspace_dimensions::ZeroTuple{NTuple{dp1,Int}}
end

function Flag(N, ns::Vararg{Int,I}) where {I}
    if ns[1] <= 0
        error("First dimension in sequence (given: $(ns[1])) must be strictly positive.")
    end
    for i in 1:(length(ns) - 1)
        if ns[i] >= ns[i + 1]
            error("Sequence of dimensions must be strictly increasing, received $ns")
        end
    end
    if ns[end] >= N
        error(
            "Last dimension in sequence (given: $(ns[end])) must be strictly lower than N (given: $N).",
        )
    end
    return Flag{N,I + 1}(ZeroTuple(tuple(ns..., N)))
end

function active_traits(f, ::Flag, args...)
    return merge_traits(IsEmbeddedManifold())
end

"""
    get_embedding(M::Flag)

Get the embedding of the [`Flag`](@ref) manifold `M`, i.e. the [`Stiefel`](@ref) manifold.
"""
get_embedding(M::Flag{N,dp1}) where {N,dp1} = Stiefel(N, M.subspace_dimensions[dp1 - 1])

@doc raw"""
    injectivity_radius(M::Flag)
    injectivity_radius(M::Flag, p)

Return the injectivity radius on the [`Flag`](@ref) `M`, which is $\frac{π}{2}$.
"""
injectivity_radius(::Flag) = π / 2
injectivity_radius(::Flag, p) = π / 2
injectivity_radius(::Flag, ::AbstractRetractionMethod) = π / 2
injectivity_radius(::Flag, p, ::AbstractRetractionMethod) = π / 2

function Base.isapprox(M::Flag, p, X, Y; atol=sqrt(max_eps(X, Y)), kwargs...)
    return isapprox(norm(M, p, X - Y), 0; atol=atol, kwargs...)
end
function Base.isapprox(M::Flag, p, q; atol=sqrt(max_eps(p, q)), kwargs...)
    return isapprox(distance(M, p, q), 0; atol=atol, kwargs...)
end

function manifold_dimension(M::Flag{N,dp1}) where {N,dp1}
    dim = 0
    for i in 1:(dp1 - 1)
        dim +=
            (M.subspace_dimensions[i] - M.subspace_dimensions[i - 1]) *
            (N - M.subspace_dimensions[i])
    end
    return dim
end
