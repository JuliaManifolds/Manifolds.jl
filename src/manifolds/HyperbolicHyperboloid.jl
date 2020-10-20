
function check_manifold_point(M::Hyperbolic, p; kwargs...)
    mpv =
        invoke(check_manifold_point, Tuple{supertype(typeof(M)),typeof(p)}, M, p; kwargs...)
    mpv === nothing || return mpv
    if !isapprox(minkowski_metric(p, p), -1.0; kwargs...)
        return DomainError(
            minkowski_metric(p, p),
            "The point $(p) does not lie on $(M) since its Minkowski inner product is not -1.",
        )
    end
    return nothing
end
function check_manifold_point(M::Hyperbolic, p::HyperboloidPoint; kwargs...)
    return check_manifold_point(M, p.value; kwargs...)
end

function check_tangent_vector(M::Hyperbolic, p, X; check_base_point = true, kwargs...)
    if check_base_point
        mpe = check_manifold_point(M, p; kwargs...)
        mpe === nothing || return mpe
    end
    mpv = invoke(
        check_tangent_vector,
        Tuple{supertype(typeof(M)),typeof(p),typeof(X)},
        M,
        p,
        X;
        check_base_point = false, # already checked above
        kwargs...,
    )
    mpv === nothing || return mpv
    if !isapprox(minkowski_metric(p, X), 0.0; kwargs...)
        return DomainError(
            abs(minkowski_metric(p, X)),
            "The vector $(X) is not a tangent vector to $(p) on $(M), since it is not orthogonal (with respect to the Minkowski inner product) in the embedding.",
        )
    end
    return nothing
end
function check_tangent_vector(
    M::Hyperbolic,
    p::HyperboloidPoint,
    X::HyperboloidTVector;
    kwargs...,
)
    return check_tangent_vector(M, p.value, X.value; kwargs...)
end

function convert(::Type{HyperboloidTVector}, X::T) where {T<:AbstractVector}
    return HyperboloidTVector(X)
end
function convert(
    ::Type{HyperboloidTVector},
    p::P,
    X::T,
) where {P<:AbstractVector,T<:AbstractVector}
    return HyperboloidTVector(X)
end
convert(::Type{AbstractVector}, X::HyperboloidTVector) = X.value
function convert(
    ::Type{T},
    p::HyperboloidPoint,
    X::HyperboloidTVector,
) where {T<:AbstractVector}
    return X.value
end

function convert(::Type{HyperboloidPoint}, p::T) where {T<:AbstractVector}
    return HyperboloidPoint(p)
end
convert(::Type{AbstractVector}, p::HyperboloidPoint) = p.value

@doc raw"""
    convert(::Type{HyperboloidPoint}, p::PoincareBallPoint)
    convert(::Type{AbstractVector}, p::PoincareBallPoint)

convert a point [`PoincareBallPoint`](@ref) `x` (from $ℝ^n$) from the
Poincaré ball model of the [`Hyperbolic`](@ref) manifold $\mathcal H^n$ to a [`HyperboloidPoint`](@ref) $π(p) ∈ ℝ^{n+1}$.
The isometry is defined by

````math
π(p) = \frac{1}{1-\lVert p \rVert^2}
\begin{pmatrix}2p_1\\⋮\\2p_n\\1+\lVert p \rVert^2\end{pmatrix}
````

Note that this is also used, when the type to convert to is a vector.
"""
function convert(::Type{HyperboloidPoint}, p::PoincareBallPoint)
    return HyperboloidPoint(convert(AbstractVector, p))
end
function convert(::Type{AbstractVector}, p::PoincareBallPoint)
    return 1 / (1 - norm(p.value)^2) .* vcat(2 .* p.value, 1 + norm(p.value)^2)
end

@doc raw"""
    convert(::Type{HyperboloidPoint}, p::PoincareHalfSpacePoint)
    convert(::Type{AbstractVector}, p::PoincareHalfSpacePoint)

convert a point [`PoincareHalfSpacePoint`](@ref) `p` (from $ℝ^n$) from the
Poincaré half plane model of the [`Hyperbolic`](@ref) manifold $\mathcal H^n$ to a [`HyperboloidPoint`](@ref) $π(p) ∈ ℝ^{n+1}$.

This is done in two steps, namely transforming it to a Poincare ball point and from there further on to a Hyperboloid point.
"""
function convert(t::Type{HyperboloidPoint}, p::PoincareHalfSpacePoint)
    return convert(t, convert(PoincareBallPoint, p))
end
function convert(t::Type{AbstractVector}, p::PoincareHalfSpacePoint)
    return convert(t, convert(PoincareBallPoint, p))
end

@doc raw"""
    convert(::Type{HyperboloidTVector}, p::PoincareBallPoint, X::PoincareBallTVector)
    convert(::Type{AbstractVector}, p::PoincareBallPoint, X::PoincareBallTVector)

Convert the [`PoincareBallTVector`](@ref) `X` from the tangent space at `p` to a
[`HyperboloidTVector`](@ref) by computing the push forward of the isometric map, cf.
[`convert(::Type{HyperboloidPoint}, p::PoincareBallPoint)`](@ref).

The push forward $π_*(p)$ maps from $ℝ^n$ to a subspace of $ℝ^{n+1}$, the formula reads

````math
π_*(p)[X] = \begin{pmatrix}
    \frac{2X_1}{1-\lVert p \rVert^2} + \frac{4}{(1-\lVert p \rVert^2)^2}⟨X,p⟩p_1\\
    ⋮\\
    \frac{2X_n}{1-\lVert p \rVert^2} + \frac{4}{(1-\lVert p \rVert^2)^2}⟨X,p⟩p_n\\
    \frac{4}{(1-\lVert p \rVert^2)^2}⟨X,p⟩
\end{pmatrix}.
````
"""
function convert(::Type{HyperboloidTVector}, p::PoincareBallPoint, X::PoincareBallTVector)
    return HyperboloidTVector(convert(AbstractVector, p, X))
end
function convert(
    ::Type{T},
    p::PoincareBallPoint,
    X::PoincareBallTVector,
) where {T<:AbstractVector}
    t = (1 - norm(p.value)^2)
    den = 4 * dot(p.value, X.value) / (t^2)
    c1 = (2 / t) .* X.value + den .* p.value
    return vcat(c1, den)
end

@doc raw"""
    convert(
        ::Type{Tuple{HyperboloidPoint,HyperboloidTVector}}.
        (p,X)::Tuple{PoincareBallPoint,PoincareBallTVector}
    )
    convert(
        ::Type{Tuple{P,T}},
        (p, X)::Tuple{PoincareBallPoint,PoincareBallTVector},
    ) where {P<:AbstractVector, T <: AbstractVector}

Convert a [`PoincareBallPoint`](@ref) `p` and a [`PoincareBallTVector`](@ref) `X`
to a [`HyperboloidPoint`](@ref) and a [`HyperboloidTVector`](@ref) simultaneously,
see [`convert(::Type{HyperboloidPoint}, ::PoincareBallPoint)`](@ref) and
[`convert(::Type{HyperboloidTVector}, ::PoincareBallPoint, ::PoincareBallTVector)`](@ref)
for the formulae.
"""
function convert(
    ::Type{Tuple{HyperboloidPoint,HyperboloidTVector}},
    (p, X)::Tuple{PoincareBallPoint,PoincareBallTVector},
)
    return (convert(HyperboloidPoint, p), convert(HyperboloidTVector, p, X))
end

@doc raw"""
    convert(::Type{HyperboloidTVector}, p::PoincareHalfSpacePoint, X::PoincareHalfSpaceTVector)
    convert(::Type{AbstractVector}, p::PoincareHalfSpacePoint, X::PoincareHalfSpaceTVector)

convert a point [`PoincareHalfSpaceTVector`](@ref) `X` (from $ℝ^n$) at `p` from the
Poincaré half plane model of the [`Hyperbolic`](@ref) manifold $\mathcal H^n$ to a
[`HyperboloidTVector`](@ref) $π(p) ∈ ℝ^{n+1}$.

This is done in two steps, namely transforming it to a Poincare ball point and from there further on to a Hyperboloid point.
"""
function convert(
    t::Type{HyperboloidTVector},
    p::PoincareHalfSpacePoint,
    X::PoincareHalfSpaceTVector,
)
    return convert(t, convert(Tuple{PoincareBallPoint,PoincareBallTVector}, (p, X))...)
end
function convert(
    t::Type{T},
    p::PoincareHalfSpacePoint,
    X::PoincareHalfSpaceTVector,
) where {T<:AbstractVector}
    return convert(t, convert(Tuple{PoincareBallPoint,PoincareBallTVector}, (p, X))...)
end

@doc raw"""
    convert(
        ::Type{Tuple{HyperboloidPoint,HyperboloidTVector},
        (p,X)::Tuple{PoincareHalfSpacePoint, PoincareHalfSpaceTVector}
    )
    convert(
        ::Type{Tuple{T,T},
        (p,X)::Tuple{PoincareHalfSpacePoint, PoincareHalfSpaceTVector}
    ) where {T<:AbstractVector}

convert a point [`PoincareHalfSpaceTVector`](@ref) `X` (from $ℝ^n$) at `p` from the
Poincaré half plane model of the [`Hyperbolic`](@ref) manifold $\mathcal H^n$
to a tuple of a [`HyperboloidPoint`](@ref) and a [`HyperboloidTVector`](@ref) $π(p) ∈ ℝ^{n+1}$
simultaneously.

This is done in two steps, namely transforming it to the Poincare ball model and from there
further on to a Hyperboloid.
"""
function convert(
    t::Type{Tuple{HyperboloidPoint,HyperboloidTVector}},
    (p, X)::Tuple{PoincareHalfSpacePoint,PoincareHalfSpaceTVector},
)
    return convert(t, convert(Tuple{PoincareBallPoint,PoincareBallTVector}, (p, X)))
end

@doc raw"""
    distance(M::Hyperbolic, p, q)
    distance(M::Hyperbolic, p::HyperboloidPoint, q::HyperboloidPoint)

Compute the distance on the [`Hyperbolic`](@ref) `M`, which reads

````math
d_{\mathcal H^n}(p,q) = \operatorname{acosh}( - ⟨p, q⟩_{\mathrm{M}}),
````

where $⟨\cdot,\cdot⟩_{\mathrm{M}}$ denotes the [`MinkowskiMetric`](@ref) on the embedding,
the [`Lorentz`](@ref)ian manifold.
"""
distance(::Hyperbolic, p, q) = acosh(max(-minkowski_metric(p, q), 1.0))
function distance(M::Hyperbolic, p::HyperboloidPoint, q::HyperboloidPoint)
    return distance(M, p.value, q.value)
end

function exp!(M::Hyperbolic, q, p, X)
    vn = sqrt(max(inner(M, p, X, X), 0.0))
    vn < eps(eltype(p)) && return copyto!(q, p)
    return copyto!(q, cosh(vn) * p + sinh(vn) / vn * X)
end

function get_basis(M::Hyperbolic, p, B::DefaultOrthonormalBasis)
    n = manifold_dimension(M)
    V = [_hyperbolize(M, p, [i == k ? 1 : 0 for k in 1:n]) for i in 1:n]
    _gram_schmidt!(M, V, p, V)
    return CachedBasis(B, V)
end

function get_basis(M::Hyperbolic, p, B::DiagonalizingOrthonormalBasis)
    n = manifold_dimension(M)
    X = B.frame_direction
    V = [_hyperbolize(M, p, [i == k ? 1 : 0 for k in 1:n]) for i in 1:n]
    κ = -ones(n)
    if norm(M, p, X) != 0
        placed = false
        for i in 1:n
            if abs(inner(M, p, X, V[i])) ≈ norm(M, p, X) # is X a multiple of V[i]?
                V[i] .= V[1]
                V[1] .= X
                placed = true
                break
            end
        end
        if !placed
            V[1] .= X
        end
        κ[1] = 0.0
    end
    _gram_schmidt!(M, V, p, V)
    return CachedBasis(B, DiagonalizingBasisData(B.frame_direction, κ, V))
end

@doc raw"""
    get_coordinates(M::Hyperbolic, p, X, ::DefaultOrthonormalBasis)

Compute the coordinates of the vector `X` with respect to the orthogonalized version of
the unit vectors from $ℝ^n$, where $n$ is the manifold dimension of the [`Hyperbolic`](@ref)
 `M`, utting them intop the tangent space at `p` and orthonormalizing them.
"""
get_coordinates(M::Hyperbolic, p, X, B::DefaultOrthonormalBasis)

function get_coordinates!(M::Hyperbolic, c, p, X, B::DefaultOrthonormalBasis)
    c = get_coordinates!(M, c, p, X, get_basis(M, p, B))
    return c
end
function get_coordinates!(M::Hyperbolic, c, p, X, B::DiagonalizingOrthonormalBasis)
    c = get_coordinates!(M, c, p, X, get_basis(M, p, B))
    return c
end

@doc raw"""
    get_vector(M::Hyperbolic, p, c, ::DefaultOrthonormalBasis)

Compute the vector from the coordinates with respect to the orthogonalized version of
the unit vectors from $ℝ^n$, where $n$ is the manifold dimension of the [`Hyperbolic`](@ref)
 `M`, utting them intop the tangent space at `p` and orthonormalizing them.
"""
get_vector(M::Hyperbolic, p, c, ::DefaultOrthonormalBasis)

function get_vector!(M::Hyperbolic, X, p, c, B::DefaultOrthonormalBasis)
    X = get_vector!(M, X, p, c, get_basis(M, p, B))
    return X
end
function get_vector!(M::Hyperbolic, X, p, c, B::DiagonalizingOrthonormalBasis)
    X = get_vector!(M, X, p, c, get_basis(M, p, B))
    return X
end

@doc raw"""
    _hyperbolize(M,q)

Given the [`Hyperbolic`](@ref)`(n)` manifold using the hyperboloid model, a point from the
$q\in ℝ^n$ can be set onto the manifold by computing its last component such that for the
resulting `p` we have that its [`minkowski_metric`](@ref) is $⟨p,p⟩_{\mathrm{M}} = - 1$,
i.e. $p_{n+1} = \sqrt{\lVert q \rVert^2-^}$
"""
_hyperbolize(M::Hyperbolic, q) = vcat(q, sqrt(norm(q)^2 + 1))

@doc raw"""
    _hyperbolize(M, p, Y)

Given the [`Hyperbolic`](@ref)`(n)` manifold using the hyperboloid model and a point `p`
thereon, we can put a vector $Y\in ℝ^n$  into the tangent space by computing its last
component such that for the
resulting `p` we have that its [`minkowski_metric`](@ref) is $⟨p,X⟩_{\mathrm{M}} = 0$,
i.e. $X_{n+1} = \frac{⟨\tilde p, Y⟩}{p_{n+1}}$, where $\tilde p = (p_1,\ldots,p_n)$.
"""
_hyperbolize(M::Hyperbolic, p, Y) = vcat(Y, dot(p[1:(end - 1)], Y) / p[end])

@doc raw"""
    _gram_schmidt!(M, W, p, V)

perform a Gram-SChmidt orthognalization of the vectors from V in the tangent space of p.
This method throws an error if you provide more vectors than [`manifold_dimension`](@ref)`(M)`.

The result is returned in W
"""
function _gram_schmidt!(M::Manifold, W, p, V::AbstractVector)
    manifold_dimension(M) < length(V) &&
        throw(ErrorException("The set of vectors cvontains too many ($(length(V))) vectors for the manifold $(M) of dimension $(manifold_dimension(M))"))
    n = norm(M, p, W[1])
    n == 0 && throw(ErrorException("Vector #1 of the set of vectors is zero."))
    W[1] ./= n
    for i in 2:length(V)
        for j in 1:(i - 1)
            W[i] .-= inner(M, p, W[i], W[j]) .* W[j]
        end
        n = norm(M, p, W[i])
        n == 0 &&
            throw(ErrorException("Vector #$(i) is in the span of the previous vectors."))
        W[i] ./= n
    end
    return W
end

@doc raw"""
    inner(M::Hyperbolic{n}, p, X, Y)
    inner(M::Hyperbolic{n}, p::HyperboloidPoint, X::HyperboloidTVector, Y::HyperboloidTVector)

Cmpute the inner product in the Hyperboloid model, i.e. the [`minkowski_metric`](@ref) in
the embedding. The formula reads

````math
g_p(X,Y) = ⟨X,Y⟩_{\mathrm{M}} = -X_{n}Y_{n} + \displaystyle\sum_{k=1}^{n-1} X_kY_k.
````
This employs the metric of the embedding, see [`Lorentz`](@ref) space.
"""
function inner(
    M::Hyperbolic,
    p::HyperboloidPoint,
    X::HyperboloidTVector,
    Y::HyperboloidTVector,
)
    return inner(M, p.value, X.value, Y.value)
end

function log!(M::Hyperbolic, X, p, q)
    scp = minkowski_metric(p, q)
    w = q + scp * p
    wn = sqrt(max(scp .^ 2 - 1, 0.0))
    wn < eps(eltype(p)) && return zero_tangent_vector!(M, X, p)
    X .= acosh(max(1.0, -scp)) / wn .* w
    return X
end

function minkowski_metric(a::HyperboloidPoint, b::HyperboloidPoint)
    return minkowski_metric(a.value, b.value)
end
function minkowski_metric(a::HyperboloidTVector, b::HyperboloidPoint)
    return minkowski_metric(a.value, b.value)
end
function minkowski_metric(a::HyperboloidPoint, b::HyperboloidTVector)
    return minkowski_metric(a.value, b.value)
end
function minkowski_metric(a::HyperboloidTVector, b::HyperboloidTVector)
    return minkowski_metric(a.value, b.value)
end

project!(::Hyperbolic, Y, p, X) = (Y .= X .+ minkowski_metric(p, X) .* p)
function project!(
    ::Hyperbolic,
    Y::HyperboloidTVector,
    p::HyperboloidPoint,
    X::HyperboloidTVector,
)
    return (Y.value .= X.value .+ minkowski_metric(p.value, X.value) .* p.value)
end

function vector_transport_to!(M::Hyperbolic, Y, p, X, q, ::ParallelTransport)
    w = log(M, p, q)
    wn = norm(M, p, w)
    wn < eps(eltype(p + q)) && return copyto!(Y, X)
    return copyto!(Y, X - (inner(M, p, w, X) * (w + log(M, q, p)) / wn^2))
end
