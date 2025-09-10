@doc raw"""
    Hyperrectangle{T} <: AbstractManifold{ℝ}

Hyperrectangle, also known as orthotope or box. This is a manifold with corners [Joyce:2010](@cite)
with the standard Euclidean metric.

# Constructor

    Hyperrectangle(lb::AbstractArray, ub::AbstractArray)

Generate the hyperrectangle of arrays such that each element of the array is between lower
and upper bound with the same index.
"""
struct Hyperrectangle{T <: AbstractArray} <: AbstractDecoratorManifold{ℝ}
    lb::T
    ub::T
    function Hyperrectangle(lb::T, ub::T) where {T <: AbstractArray}
        for i in eachindex(lb, ub)
            if lb[i] > ub[i]
                throw(
                    ArgumentError(
                        "Lower bound at index $i ($(lb[i])) is greater than the upper bound ($(ub[i]))",
                    ),
                )
            end
        end
        return new{T}(lb, ub)
    end
end

function check_point(M::Hyperrectangle, p)
    if !(eltype(p) <: Real)
        return DomainError(
            eltype(p),
            "The matrix $(p) is not a real-valued matrix, so it does not lie on $(M).",
        )
    end
    for i in eachindex(M.lb, M.ub, p)
        if p[i] < M.lb[i]
            return DomainError(
                p[i],
                "At index $i the point has coordinate $(p[i]), below the lower bound of $(M.lb[i])",
            )
        end
        if p[i] > M.ub[i]
            return DomainError(
                p[i],
                "At index $i the point has coordinate $(p[i]), above the upper bound of $(M.ub[i])",
            )
        end
    end
    return nothing
end

function check_vector(M::Hyperrectangle, p, X; kwargs...)
    if !(eltype(X) <: Real)
        return DomainError(
            eltype(X),
            "The matrix $(X) is not a real-valued matrix, so it can not be a tangent vector to $(p) on $(M).",
        )
    end
    return nothing
end

default_approximation_method(::Hyperrectangle, ::typeof(mean)) = EfficientEstimator()
metric(::Hyperrectangle) = EuclideanMetric()
"""
    default_retraction_method(M::Hyperrectangle)

Return [`ProjectionRetraction`](@extref `ManifoldsBase.ProjectionRetraction`) as the default
retraction for the [`Hyperrectangle`](@ref) manifold.
"""
default_retraction_method(::Hyperrectangle) = ProjectionRetraction()

"""
    distance(M::Hyperrectangle, p, q)

Compute the euclidean distance between two points on the [`Hyperrectangle`](@ref)
manifold `M`, i.e. for vectors it's just the norm of the difference, for matrices
and higher order arrays, the matrix and tensor Frobenius norm, respectively.
"""
Base.@propagate_inbounds function distance(M::Hyperrectangle, p, q)
    # Inspired by euclidean distance calculation in Distances.jl
    # Much faster for large p, q than a naive implementation
    @boundscheck if axes(p) != axes(q)
        throw(DimensionMismatch("At last one of $p and $q does not belong to $M"))
    end
    s = zero(eltype(p))
    @inbounds begin
        @simd for I in eachindex(p, q)
            p_i = p[I]
            q_i = q[I]
            s += abs2(p_i - q_i)
        end # COV_EXCL_LINE
    end
    return sqrt(s)
end
distance(::Hyperrectangle, p::Number, q::Number) = abs(p - q)

"""
    embed(M::Hyperrectangle, p)

Embed the point `p` in `M`. Equivalent to an identity map.
"""
embed(::Hyperrectangle, p) = p

"""
    embed(M::Hyperrectangle, p, X)

Embed the tangent vector `X` at point `p` in `M`. Equivalent to an identity map.
"""
embed(::Hyperrectangle, p, X) = X

@doc raw"""
    exp(M::Hyperrectangle, p, X)

Compute the exponential map on the [`Hyperrectangle`](@ref) manifold `M` from `p` in direction
`X`, which in this case is just
````math
\exp_p X = p + X.
````
"""
Base.exp(::Hyperrectangle, p, X) = p + X
exp_fused(::Hyperrectangle, p, X, t::Number) = p .+ t .* X

exp!(::Hyperrectangle, q, p, X) = (q .= p .+ X)
exp_fused!(::Hyperrectangle, q, p, X, t::Number) = (q .= p .+ t .* X)

function get_coordinates_orthonormal(::Hyperrectangle, p, X, ::RealNumbers)
    return vec(X)
end

function get_coordinates_orthonormal!(::Hyperrectangle, c, p, X, ::RealNumbers)
    copyto!(c, vec(X))
    return c
end

function get_vector_orthonormal(::Hyperrectangle{<:AbstractVector}, ::Any, c, ::RealNumbers)
    # this method is defined just to skip a reshape
    return c
end

function get_vector_orthonormal!(
        ::Hyperrectangle{<:AbstractVector},
        Y,
        ::Any,
        c,
        ::RealNumbers,
    )
    # this method is defined just to skip a reshape
    copyto!(Y, c)
    return Y
end
function get_vector_orthonormal!(M::Hyperrectangle, Y, ::Any, c, ::RealNumbers)
    S = representation_size(M)
    copyto!(Y, reshape(c, S))
    return Y
end

@doc raw"""
    injectivity_radius(M::Hyperrectangle, p)

Return the injectivity radius on the [`Hyperrectangle`](@ref) `M` at point `p`, which is
the distance to the nearest boundary the point is not on.
"""
function injectivity_radius(M::Hyperrectangle, p)
    ir = Inf
    for i in eachindex(M.lb, p)
        dist_ub = M.ub[i] - p[i]
        if dist_ub > 0
            ir = min(ir, dist_ub)
        end
        dist_lb = p[i] - M.lb[i]
        if dist_lb > 0
            ir = min(ir, dist_lb)
        end
    end
    return ir
end
injectivity_radius(M::Hyperrectangle) = 0.0
injectivity_radius(M::Hyperrectangle, ::AbstractRetractionMethod) = 0.0
injectivity_radius(M::Hyperrectangle, p, ::ProjectionRetraction) = injectivity_radius(M, p)
injectivity_radius(M::Hyperrectangle, p, ::ExponentialRetraction) = injectivity_radius(M, p)

@doc raw"""
    inner(M::Hyperrectangle, p, X, Y)

Compute the inner product on the [`Hyperrectangle`](@ref) `M`, which is just
the inner product on the real-valued vector space of arrays (or tensors)
of size ``n_1 × n_2  ×  …  × n_i``, i.e.

````math
g_p(X,Y) = \sum_{k ∈ I} X_{k} Y_{k},
````

where ``I`` is the set of vectors ``k ∈ ℕ^i``, such that for all

``i ≤ j ≤ i`` it holds ``1 ≤ k_j ≤ n_j``.

For the special case of ``i ≤ 2``, i.e. matrices and vectors, this simplifies to

````math
g_p(X,Y) = \operatorname{tr}(X^{\mathrm{T}}Y),
````

where ``⋅^{\mathrm{T}}`` denotes transposition.
"""
@inline inner(::Hyperrectangle, p, X, Y) = dot(X, Y)
"""
    is_flat(::Hyperrectangle)

Return true. [`Hyperrectangle`](@ref) is a flat manifold.
"""
is_flat(M::Hyperrectangle) = true

@doc raw"""
    log(M::Hyperrectangle, p, q)

Compute the logarithmic map on the [`Hyperrectangle`](@ref) `M` from `p` to `q`,
which in this case is just
````math
\log_p q = q-p.
````
"""
Base.log(::Hyperrectangle, ::Any...)
Base.log(::Hyperrectangle, p, q) = q .- p

log!(::Hyperrectangle, X, p, q) = (X .= q .- p)

_product_of_dimensions(M::Hyperrectangle) = prod(size(M.lb))

"""
    manifold_dimension(M::Hyperrectangle)

Return the manifold dimension of the [`Hyperrectangle`](@ref) `M`, i.e. the product of all array
dimensions.
"""
function manifold_dimension(M::Hyperrectangle)
    return _product_of_dimensions(M)
end

"""
    manifold_volume(::Hyperrectangle)

Return volume of the [`Hyperrectangle`](@ref) manifold, i.e. infinity.
"""
function manifold_volume(M::Hyperrectangle)
    vol = one(eltype(M.lb))
    for i in eachindex(M.lb, M.ub)
        vol *= M.ub[i] - M.lb[i]
    end
    return vol
end

#
# When Statistics / Statsbase.mean! is consistent with mean, we can pass this on to them as well
function Statistics.mean!(
        ::Hyperrectangle,
        y,
        x::AbstractVector,
        ::EfficientEstimator;
        kwargs...,
    )
    n = length(x)
    copyto!(y, first(x))
    @inbounds for j in 2:n
        y .+= x[j]
    end
    y ./= n
    return y
end
function Statistics.mean!(
        ::Hyperrectangle,
        y,
        x::AbstractVector,
        w::AbstractWeights,
        ::EfficientEstimator;
        kwargs...,
    )
    n = length(x)
    if length(w) != n
        throw(
            DimensionMismatch(
                "The number of weights ($(length(w))) does not match the number of points for the mean ($(n)).",
            ),
        )
    end
    copyto!(y, first(x))
    y .*= first(w)
    @inbounds for j in 2:n
        iszero(w[j]) && continue
        y .+= w[j] .* x[j]
    end
    y ./= sum(w)
    return y
end

mid_point(::Hyperrectangle, p1, p2) = (p1 .+ p2) ./ 2

function mid_point!(::Hyperrectangle, q, p1, p2)
    q .= (p1 .+ p2) ./ 2
    return q
end

@doc raw"""
    norm(M::Hyperrectangle, p, X)

Compute the norm of a tangent vector `X` at `p` on the [`Hyperrectangle`](@ref)
`M`, i.e. since every tangent space can be identified with `M` itself
in this case, just the (Frobenius) norm of `X`.
"""
LinearAlgebra.norm(::Hyperrectangle, ::Any, X) = norm(X)

"""
    parallel_transport_direction(M::Hyperrectangle, p, X, d)

the parallel transport on [`Hyperrectangle`](@ref) is the identity, i.e. returns `X`.
"""
parallel_transport_direction(::Hyperrectangle, ::Any, X, ::Any) = X
parallel_transport_direction!(::Hyperrectangle, Y, ::Any, X, ::Any) = copyto!(Y, X)

"""
    parallel_transport_to(M::Hyperrectangle, p, X, q)

the parallel transport on [`Hyperrectangle`](@ref) is the identity, i.e. returns `X`.
"""
parallel_transport_to(::Hyperrectangle, ::Any, X, ::Any) = X
parallel_transport_to!(::Hyperrectangle, Y, ::Any, X, ::Any) = copyto!(Y, X)

@doc raw"""
    project(M::Hyperrectangle, p)

Project an arbitrary point `p` onto the [`Hyperrectangle`](@ref) manifold `M`, which
is of course just the identity map.
"""
project(::Hyperrectangle, ::Any)

function project!(M::Hyperrectangle, q, p)
    copyto!(q, p)
    for i in eachindex(M.lb, q)
        q[i] = clamp(q[i], M.lb[i], M.ub[i])
    end
    return q
end

"""
    project(M::Hyperrectangle, p, X)

Project an arbitrary vector `X` into the tangent space of a point `p` on the
[`Hyperrectangle`](@ref) `M`, which is just the identity, since any tangent
space of `M` can be identified with all of `M`.
"""
project(::Hyperrectangle, ::Any, ::Any)

function project!(M::Hyperrectangle, Y, p, X)
    copyto!(Y, X)
    for i in eachindex(M.lb, Y)
        if Y[i] >= 0
            Y[i] = min(Y[i], M.ub[i] - p[i])
        else
            Y[i] = max(Y[i], M.lb[i] - p[i])
        end
    end
    return Y
end

function Random.rand!(
        rng::AbstractRNG,
        M::Hyperrectangle,
        pX;
        σ = one(eltype(pX)),
        vector_at = nothing,
    )
    if vector_at === nothing
        pX .= M.lb .+ rand(rng, eltype(M.lb), size(M.lb)) .* (M.ub .- M.lb)
    else
        pX .= randn(rng, eltype(pX), size(pX)) .* σ
        project!(M, pX, vector_at, pX)
    end
    return pX
end

"""
    representation_size(M::Hyperrectangle)

Return the array dimensions required to represent an element on the
[`Hyperrectangle`](@ref) `M`, i.e. the vector of all array dimensions.
"""
representation_size(M::Hyperrectangle) = size(M.lb)

function ManifoldsBase.retract_project!(M::Hyperrectangle, r, q, Y)
    r .= q .+ Y
    project(M, r, r)
    return r
end
function ManifoldsBase.retract_project_fused!(M::Hyperrectangle, r, q, Y, t::Number)
    r .= q .+ t .* Y
    project!(M, r, r)
    return r
end

@doc raw"""
    riemann_tensor(M::Hyperrectangle, p, X, Y, Z)

Compute the Riemann tensor ``R(X,Y)Z`` at point `p` on [`Hyperrectangle`](@ref) manifold `M`.
Its value is always the zero tangent vector.
"""
riemann_tensor(M::Hyperrectangle, p, X, Y, Z)

function riemann_tensor!(::Hyperrectangle, Xresult, p, X, Y, Z)
    return fill!(Xresult, 0)
end

@doc raw"""
    sectional_curvature(::Hyperrectangle, p, X, Y)

Sectional curvature of [`Hyperrectangle`](@ref) manifold `M` is 0.
"""
function sectional_curvature(::Hyperrectangle, p, X, Y)
    return 0.0
end

@doc raw"""
    sectional_curvature_max(::Hyperrectangle)

Sectional curvature of [`Hyperrectangle`](@ref) manifold `M` is 0.
"""
function sectional_curvature_max(::Hyperrectangle)
    return 0.0
end

@doc raw"""
    sectional_curvature_min(M::Hyperrectangle)

Sectional curvature of [`Hyperrectangle`](@ref) manifold `M` is 0.
"""
function sectional_curvature_min(::Hyperrectangle)
    return 0.0
end

function Base.show(io::IO, M::Hyperrectangle)
    return print(io, "Hyperrectangle($(M.lb), $(M.ub))")
end

function vector_transport_direction(
        M::Hyperrectangle,
        p,
        X,
        ::Any,
        ::AbstractVectorTransportMethod = default_vector_transport_method(M, typeof(p)),
        ::AbstractRetractionMethod = default_retraction_method(M, typeof(p)),
    )
    return X
end
function vector_transport_direction!(
        M::Hyperrectangle,
        Y,
        p,
        X,
        ::Any,
        ::AbstractVectorTransportMethod = default_vector_transport_method(M, typeof(p)),
        ::AbstractRetractionMethod = default_retraction_method(M, typeof(p)),
    )
    return copyto!(Y, X)
end
"""
    vector_transport_to(M::Hyperrectangle, p, X, q, ::AbstractVectorTransportMethod)

Transport the vector `X` from the tangent space at `p` to the tangent space at `q`
on the [`Hyperrectangle`](@ref) `M`, which simplifies to the identity.
"""
vector_transport_to(::Hyperrectangle, ::Any, ::Any, ::Any, ::AbstractVectorTransportMethod)
function vector_transport_to(
        M::Hyperrectangle,
        p,
        X,
        ::Any,
        ::AbstractVectorTransportMethod = default_vector_transport_method(M, typeof(p)),
        ::AbstractRetractionMethod = default_retraction_method(M, typeof(p)),
    )
    return X
end

function vector_transport_to!(
        M::Hyperrectangle,
        Y,
        p,
        X,
        ::Any,
        ::AbstractVectorTransportMethod = default_vector_transport_method(M, typeof(p)),
        ::AbstractRetractionMethod = default_retraction_method(M, typeof(p)),
    )
    return copyto!(Y, X)
end

Statistics.var(::Hyperrectangle, x::AbstractVector; kwargs...) = sum(var(x; kwargs...))

@doc raw"""
    volume_density(M::Hyperrectangle, p, X)

Return volume density function of [`Hyperrectangle`](@ref) manifold `M`, i.e. 1.
"""
function volume_density(::Hyperrectangle, p, X)
    return one(eltype(X))
end

@doc raw"""
    Y = Weingarten(M::Hyperrectangle, p, X, V)
    Weingarten!(M::Hyperrectangle, Y, p, X, V)

Compute the Weingarten map ``\mathcal W_p`` at `p` on the [`Hyperrectangle`](@ref) `M` with
respect to the tangent vector ``X \in T_p\mathcal M`` and the normal vector ``V \in N_p\mathcal M``.

Since this a flat space by itself, the result is always the zero tangent vector.
"""
Weingarten(::Hyperrectangle, p, X, V)

Weingarten!(::Hyperrectangle, Y, p, X, V) = fill!(Y, 0)

"""
    zero_vector(M::Hyperrectangle, p)

Return the zero vector in the tangent space of `p` on the [`Hyperrectangle`](@ref)
`M`, which here is just a zero filled array the same size as `p`.
"""
zero_vector(::Hyperrectangle, ::Any...)

zero_vector!(::Hyperrectangle, X, ::Any) = fill!(X, 0)
