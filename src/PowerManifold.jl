@doc doc"""
    PowerManifold{TM<:Manifold, TSize<:Tuple} <: Manifold

Power manifold $M^{N_1 \times N_2 \times \dots \times N_n}$ with power geometry.
It is represented by an array-like structure with $n$ dimensions and sizes
$N_1, N_2, \dots, N_n$, along each dimension.
`TSize` statically defines the number of elements along each axis.
For example, a manifold-valued time series would be represented by a power
manifold with $n$ equal to 1 and $N_1$ equal to the number of samples.
A manifold-valued image (for example in diffusion tensor imaging) would
be represented by a two-axis power manifold ($n=2$) with $N_1$ and $N_2$
equal to width and height of the image.

While the size of the manifold is static, points on the power manifold
would not be represented by statically-sized arrays. Operations on small
power manifolds might be faster if they are represented as [`ProductManifold`](@ref).

# Constructor

    PowerManifold(M, Tuple{N_1, N_2, ..., N_n})

generates the power manifold $M^{N_1 \times N_2 \times \dots \times N_n}$.
"""
struct PowerManifold{TM<:Manifold, TSize} <: Manifold
    manifold::TM
end

function PowerManifold(manifold::Manifold, size::Tuple)
    return PowerManifold{typeof(manifold), size}(manifold)
end

function isapprox(M::PowerManifold, x, y; kwargs...)
    error("TODO")
end

function isapprox(M::PowerManifold, x, v, w; kwargs...)
    error("TODO")
end

function representation_size(M::PowerManifold{<:Manifold, TSize}) where TSize
    return (representation_size(M.manifold)^(product(size_to_tuple(TSize))),)
end

function manifold_dimension(M::ProductManifold{<:Manifold, TSize}) where TSize
    return manifold_dimension(M.manifold)^(product(size_to_tuple(TSize)))
end
