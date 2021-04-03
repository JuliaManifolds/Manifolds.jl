@doc raw"""
    Orthogonal(n)

The group of (real) orthogonal matrices ``\mathrm{O}(n)``.

This constructor is equivalent to calling [`Unitary(n,ℝ)`](@ref).
"""
const Orthogonal{n} = Unitary{n,ℝ}

Orthogonal(n) = Orthogonal{n}()

@doc raw"""
    group_exp(G::Orthogonal{2}, X)

Compute the group exponential map on the [`Orthogonal(2)`] group.

Given ``X = \begin{pmatrix} 0 & -θ \\ θ & 0 \end{pmatrix}``, the group exponential is

````math
\exp_e \colon X ↦ \begin{pmatrix} \cos θ & -\sin θ \\ \sin θ & \cos θ \end{pmatrix}.
````
"""
function group_exp!(::Orthogonal{2}, q, X)
    @assert size(X) == (2, 2)
    @inbounds θ = (X[2, 1] - X[1, 2]) / 2
    sinθ, cosθ = sincos(θ)
    @inbounds begin
        q[1, 1] = cosθ
        q[2, 1] = sinθ
        q[1, 2] = -sinθ
        q[2, 2] = cosθ
    end
    return q
end

@doc raw"""
    injectivity_radius(G::Orthogonal)
    injectivity_radius(G::Orthogonal, p)

Return the injectivity radius on the [`Orthogonal`](@ref) group `G`, which is globally
``π \sqrt{2}``
"""
function injectivity_radius(::Orthogonal, p)
    T = float(real(eltype(p)))
    return T(sqrt(2)) * π
end
function injectivity_radius(::Orthogonal, p, ::ExponentialRetraction)
    T = float(real(eltype(p)))
    return T(sqrt(2)) * π
end

"""
    mean(
        G::Orthogonal,
        x::AbstractVector,
        [w::AbstractWeights,]
        method = GeodesicInterpolationWithinRadius(π/2/√2);
        kwargs...,
    )

Compute the Riemannian [`mean`](@ref mean(G::Manifold, args...)) of `x` using
[`GeodesicInterpolationWithinRadius`](@ref).
"""
mean(::Orthogonal, ::Any)

function Statistics.mean!(G::Orthogonal, q, x::AbstractVector, w::AbstractVector; kwargs...)
    return mean!(G, q, x, w, GeodesicInterpolationWithinRadius(π / 2 / √2); kwargs...)
end

Base.show(io::IO, ::Orthogonal{n}) where {n} = print(io, "Orthogonal($n)")

vector_transport_to(::Orthogonal, p, X, q, ::ParallelTransport) = X

vector_transport_to!(::Orthogonal, Y, p, X, q, ::ParallelTransport) = copyto!(Y, X)
