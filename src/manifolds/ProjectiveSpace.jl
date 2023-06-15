"""
    AbstractProjectiveSpace{𝔽} <: AbstractDecoratorManifold{𝔽}

An abstract type to represent a projective space over `𝔽` that is represented isometrically
in the embedding.
"""
abstract type AbstractProjectiveSpace{𝔽} <: AbstractDecoratorManifold{𝔽} end

@doc raw"""
    ProjectiveSpace{n,𝔽} <: AbstractProjectiveSpace{𝔽}

The projective space $𝔽ℙ^n$ is the manifold of all lines in $𝔽^{n+1}$.
The default representation is in the embedding, i.e. as unit norm vectors in
$𝔽^{n+1}$:
````math
𝔽ℙ^n := \bigl\{ [p] ⊂ 𝔽^{n+1} \ \big|\ \lVert p \rVert = 1, λ ∈ 𝔽, |λ| = 1, p ∼ p λ \bigr\},
````
where $[p]$ is an equivalence class of points $p$, and $∼$ indicates equivalence.
For example, the real projective space $ℝℙ^n$ is represented as the unit sphere $𝕊^n$, where
antipodal points are considered equivalent.

The tangent space at point $p$ is given by

````math
T_p 𝔽ℙ^{n} := \bigl\{ X ∈ 𝔽^{n+1}\ \big|\ ⟨p,X⟩ = 0 \bigr \},
````
where $⟨⋅,⋅⟩$ denotes the inner product in the embedding $𝔽^{n+1}$.

When $𝔽 = ℍ$, this implementation of $ℍℙ^n$ is the right-quaternionic projective
space.

# Constructor

    ProjectiveSpace(n[, field=ℝ])

Generate the projective space $𝔽ℙ^{n} ⊂ 𝔽^{n+1}$, defaulting to the real projective space
$ℝℙ^n$, where `field` can also be used to generate the complex- and right-quaternionic
projective spaces.
"""
struct ProjectiveSpace{N,𝔽} <: AbstractProjectiveSpace{𝔽} end
ProjectiveSpace(n::Int, field::AbstractNumbers=ℝ) = ProjectiveSpace{n,field}()

function active_traits(f, ::AbstractProjectiveSpace, args...)
    return merge_traits(IsIsometricEmbeddedManifold())
end

@doc raw"""
    ArrayProjectiveSpace{T<:Tuple,𝔽} <: AbstractProjectiveSpace{𝔽}

The projective space $𝔽ℙ^{n₁,n₂,…,nᵢ}$ is the manifold of all lines in $𝔽^{n₁,n₂,…,nᵢ}$.
The default representation is in the embedding, i.e. as unit (Frobenius) norm matrices in
$𝔽^{n₁,n₂,…,nᵢ}$:

````math
𝔽ℙ^{n_1, n_2, …, n_i} := \bigl\{ [p] ⊂ 𝔽^{n_1, n_2, …, n_i} \ \big|\ \lVert p \rVert_{\mathrm{F}} = 1, λ ∈ 𝔽, |λ| = 1, p ∼ p λ \bigr\}.
````
where $[p]$ is an equivalence class of points $p$, $\sim$ indicates equivalence, and
$\lVert ⋅ \rVert_{\mathrm{F}}$ is the Frobenius norm.
Note that unlike [`ProjectiveSpace`](@ref), the argument for `ArrayProjectiveSpace`
is given by the size of the embedding.
This means that [`ProjectiveSpace(2)`](@ref) and `ArrayProjectiveSpace(3)` are the same
manifold.
Additionally, `ArrayProjectiveSpace(n,1;field=𝔽)` and [`Grassmann(n,1;field=𝔽)`](@ref) are
the same.

The tangent space at point $p$ is given by

````math
T_p 𝔽ℙ^{n_1, n_2, …, n_i} := \bigl\{ X ∈ 𝔽^{n_1, n_2, …, n_i}\ |\ ⟨p,X⟩_{\mathrm{F}} = 0 \bigr \},
````

where $⟨⋅,⋅⟩_{\mathrm{F}}$ denotes the (Frobenius) inner product in the embedding
$𝔽^{n_1, n_2, …, n_i}$.

# Constructor

    ArrayProjectiveSpace(n₁,n₂,...,nᵢ; field=ℝ)

Generate the projective space $𝔽ℙ^{n_1, n_2, …, n_i}$, defaulting to the real projective
space, where `field` can also be used to generate the complex- and right-quaternionic
projective spaces.
"""
struct ArrayProjectiveSpace{N,𝔽} <: AbstractProjectiveSpace{𝔽} where {N<:Tuple} end
function ArrayProjectiveSpace(n::Vararg{Int,I}; field::AbstractNumbers=ℝ) where {I}
    return ArrayProjectiveSpace{Tuple{n...},field}()
end

function allocation_promotion_function(::AbstractProjectiveSpace{ℂ}, f, args::Tuple)
    return complex
end

@doc raw"""
    check_point(M::AbstractProjectiveSpace, p; kwargs...)

Check whether `p` is a valid point on the [`AbstractProjectiveSpace`](@ref) `M`, i.e.
that it has the same size as elements of the embedding and has unit Frobenius norm.
The tolerance for the norm check can be set using the `kwargs...`.
"""
function check_point(M::AbstractProjectiveSpace, p; kwargs...)
    if !isapprox(norm(p), 1; kwargs...)
        return DomainError(
            norm(p),
            "The point $(p) does not lie on the $(M) since its norm is not 1.",
        )
    end
    return nothing
end

@doc raw"""
    check_vector(M::AbstractProjectiveSpace, p, X; kwargs... )

Check whether `X` is a tangent vector in the tangent space of `p` on the
[`AbstractProjectiveSpace`](@ref) `M`, i.e. that `X` has the same size as elements of the
tangent space of the embedding and that the Frobenius inner product
$⟨p, X⟩_{\mathrm{F}} = 0$.
"""
function check_vector(M::AbstractProjectiveSpace, p, X; kwargs...)
    if !isapprox(dot(p, X), 0; kwargs...)
        return DomainError(
            dot(p, X),
            "The vector $(X) is not a tangent vector to $(p) on $(M), since it is not" *
            " orthogonal in the embedding.",
        )
    end
    return nothing
end

function decorated_manifold(M::AbstractProjectiveSpace{𝔽}) where {𝔽}
    return Euclidean(representation_size(M)...; field=𝔽)
end

get_embedding(M::AbstractProjectiveSpace) = decorated_manifold(M)

embed(::AbstractProjectiveSpace, p) = p
embed(::AbstractProjectiveSpace, p, X) = X

@doc raw"""
    distance(M::AbstractProjectiveSpace, p, q)

Compute the Riemannian distance on [`AbstractProjectiveSpace`](@ref) `M`$=𝔽ℙ^n$ between
points `p` and `q`, i.e.
````math
d_{𝔽ℙ^n}(p, q) = \arccos\bigl| ⟨p, q⟩_{\mathrm{F}} \bigr|.
````

Note that this definition is similar to that of the [`AbstractSphere`](@ref).
However, the absolute value ensures that all equivalent `p` and `q` have the same pairwise
distance.
"""
function distance(::AbstractProjectiveSpace, p, q)
    z = dot(p, q)
    cosθ = abs(z)
    T = float(real(Base.promote_eltype(p, q)))
    # abs and relative error of acos is less than sqrt(eps(T))
    cosθ < 1 - sqrt(eps(T)) / 8 && return acos(cosθ)
    # improved accuracy for q close to p or -p
    λ = sign(z)
    return 2 * abs(atan(norm(p .* λ .- q), norm(p .* λ .+ q)))
end

function exp!(M::AbstractProjectiveSpace, q, p, X)
    θ = norm(M, p, X)
    q .= cos(θ) .* p .+ usinc(θ) .* X
    return q
end

function get_basis(::ProjectiveSpace{n,ℝ}, p, B::DiagonalizingOrthonormalBasis{ℝ}) where {n}
    return get_basis(Sphere{n,ℝ}(), p, B)
end

@doc raw"""
    get_coordinates(M::AbstractProjectiveSpace, p, X, B::DefaultOrthonormalBasis{ℝ})

Represent the tangent vector $X$ at point $p$ from the [`AbstractProjectiveSpace`](@ref)
$M = 𝔽ℙ^n$ in an orthonormal basis by unitarily transforming the hyperplane containing $X$,
whose normal is $p$, to the hyperplane whose normal is the $x$-axis.

Given $q = p \overline{λ} + x$, where
$λ = \frac{⟨x, p⟩_{\mathrm{F}}}{|⟨x, p⟩_{\mathrm{F}}|}$, $⟨⋅, ⋅⟩_{\mathrm{F}}$ denotes the
Frobenius inner product, and $\overline{⋅}$ denotes complex or quaternionic conjugation, the
formula for $Y$ is
````math
\begin{pmatrix}0 \\ Y\end{pmatrix} = \left(X - q\frac{2 ⟨q, X⟩_{\mathrm{F}}}{⟨q, q⟩_{\mathrm{F}}}\right)\overline{λ}.
````
"""
get_coordinates(::AbstractProjectiveSpace{ℝ}, p, X, ::DefaultOrthonormalBasis)

function get_coordinates_orthonormal!(
    M::AbstractProjectiveSpace{𝔽},
    Y,
    p,
    X,
    ::RealNumbers,
) where {𝔽}
    n = div(manifold_dimension(M), real_dimension(𝔽))
    z = p[1]
    cosθ = abs(z)
    λ = nzsign(z, cosθ)
    pend, Xend = view(p, 2:(n + 1)), view(X, 2:(n + 1))
    factor = λ' * X[1] / (1 + cosθ)
    Y .= (Xend .- pend .* factor) .* λ'
    return Y
end

@doc raw"""
    get_vector(M::AbstractProjectiveSpace, p, X, B::DefaultOrthonormalBasis{ℝ})

Convert a one-dimensional vector of coefficients $X$ in the basis `B` of the tangent space
at $p$ on the [`AbstractProjectiveSpace`](@ref) $M=𝔽ℙ^n$ to a tangent vector $Y$ at $p$ by
unitarily transforming the hyperplane containing $X$, whose normal is the $x$-axis, to the
hyperplane whose normal is $p$.

Given $q = p \overline{λ} + x$, where
$λ = \frac{⟨x, p⟩_{\mathrm{F}}}{|⟨x, p⟩_{\mathrm{F}}|}$, $⟨⋅, ⋅⟩_{\mathrm{F}}$ denotes the
Frobenius inner product, and $\overline{⋅}$ denotes complex or quaternionic conjugation, the
formula for $Y$ is
````math
Y = \left(X - q\frac{2 \left\langle q, \begin{pmatrix}0 \\ X\end{pmatrix}\right\rangle_{\mathrm{F}}}{⟨q, q⟩_{\mathrm{F}}}\right) λ.
````
"""
get_vector(::AbstractProjectiveSpace, p, X, ::DefaultOrthonormalBasis{ℝ})

function get_vector_orthonormal!(
    M::AbstractProjectiveSpace{𝔽},
    Y,
    p,
    X,
    ::RealNumbers,
) where {𝔽}
    n = div(manifold_dimension(M), real_dimension(𝔽))
    z = p[1]
    cosθ = abs(z)
    λ = nzsign(z, cosθ)
    pend = view(p, 2:(n + 1))
    pX = dot(pend, X)
    Y[1] = -λ * pX * λ
    Y[2:(n + 1)] .= (X .- pend .* (pX / (1 + cosθ))) .* λ
    return Y
end

injectivity_radius(::AbstractProjectiveSpace) = π / 2
injectivity_radius(::AbstractProjectiveSpace, p) = π / 2
injectivity_radius(::AbstractProjectiveSpace, ::AbstractRetractionMethod) = π / 2
injectivity_radius(::AbstractProjectiveSpace, p, ::AbstractRetractionMethod) = π / 2

@doc raw"""
    inverse_retract(M::AbstractProjectiveSpace, p, q, method::ProjectionInverseRetraction)
    inverse_retract(M::AbstractProjectiveSpace, p, q, method::PolarInverseRetraction)
    inverse_retract(M::AbstractProjectiveSpace, p, q, method::QRInverseRetraction)

Compute the equivalent inverse retraction [`ProjectionInverseRetraction`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions.html#ManifoldsBase.ProjectionInverseRetraction),
[`PolarInverseRetraction`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions.html#ManifoldsBase.PolarInverseRetraction), and [`QRInverseRetraction`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions.html#ManifoldsBase.QRInverseRetraction) on the
[`AbstractProjectiveSpace`](@ref) manifold `M`$=𝔽ℙ^n$, i.e.
````math
\operatorname{retr}_p^{-1} q = q \frac{1}{⟨p, q⟩_{\mathrm{F}}} - p,
````
where $⟨⋅, ⋅⟩_{\mathrm{F}}$ is the Frobenius inner product.

Note that this inverse retraction is equivalent to the three corresponding inverse
retractions on [`Grassmann(n+1,1,𝔽)`](@ref), where the three inverse retractions in this
case coincide.
For $ℝℙ^n$, it is the same as the `ProjectionInverseRetraction` on the real
[`Sphere`](@ref).
"""
inverse_retract(
    ::AbstractProjectiveSpace,
    p,
    q,
    ::Union{ProjectionInverseRetraction,PolarInverseRetraction,QRInverseRetraction},
)

function inverse_retract_qr!(::AbstractProjectiveSpace, X, p, q)
    X .= q ./ dot(p, q) .- p
    return X
end
function inverse_retract_polar!(::AbstractProjectiveSpace, X, p, q)
    X .= q ./ dot(p, q) .- p
    return X
end
function inverse_retract_project!(::AbstractProjectiveSpace, X, p, q)
    X .= q ./ dot(p, q) .- p
    return X
end

@doc raw"""
    isapprox(M::AbstractProjectiveSpace, p, q; kwargs...)

Check that points `p` and `q` on the [`AbstractProjectiveSpace`](@ref) `M`$=𝔽ℙ^n$ are
members of the same equivalence class, i.e. that $p = q λ$ for some element $λ ∈ 𝔽$ with
unit absolute value, that is, $|λ| = 1$.
This is equivalent to the Riemannian
[`distance`](@ref distance(::AbstractProjectiveSpace, p, q)) being 0.
"""
function _isapprox(::AbstractProjectiveSpace, p, q; kwargs...)
    return isapprox(abs(dot(p, q)), 1; kwargs...)
end

"""
    is_flat(M::AbstractProjectiveSpace)

Return true if [`AbstractProjectiveSpace`](@ref) is of dimension 1 and false otherwise.
"""
is_flat(M::AbstractProjectiveSpace) = manifold_dimension(M) == 1

@doc raw"""
    log(M::AbstractProjectiveSpace, p, q)

Compute the logarithmic map on [`AbstractProjectiveSpace`](@ref) `M`$ = 𝔽ℙ^n$,
i.e. the tangent vector whose corresponding [`geodesic`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions.html#ManifoldsBase.geodesic-Tuple{AbstractManifold,%20Any,%20Any}) starting from `p`
reaches `q` after time 1 on `M`. The formula reads

````math
\log_p q = (q λ - \cos θ p) \frac{θ}{\sin θ},
````
where $θ = \arccos|⟨q, p⟩_{\mathrm{F}}|$ is the
[`distance`](@ref distance(::AbstractProjectiveSpace, p, q)) between $p$ and $q$,
$⟨⋅, ⋅⟩_{\mathrm{F}}$ is the Frobenius inner product, and
$λ = \frac{⟨q, p⟩_{\mathrm{F}}}{|⟨q, p⟩_{\mathrm{F}}|} ∈ 𝔽$ is the unit scalar that
minimizes $d_{𝔽^{n+1}}(p - q λ)$.
That is, $q λ$ is the member of the equivalence class $[q]$ that is closest to $p$ in the
embedding.
As a result, $\exp_p \circ \log_p \colon q ↦ q λ$.

The logarithmic maps for the real [`AbstractSphere`](@ref) $𝕊^n$ and the real projective
space $ℝℙ^n$ are identical when $p$ and $q$ are in the same hemisphere.
"""
log(::AbstractProjectiveSpace, p, q)

function log!(M::AbstractProjectiveSpace, X, p, q)
    z = dot(q, p)
    cosθ = abs(z)
    λ = nzsign(z, cosθ)
    X .= (q .* λ .- cosθ .* p) ./ usinc_from_cos(cosθ)
    return project!(M, X, p, X)
end

@doc raw"""
    manifold_dimension(M::AbstractProjectiveSpace{𝔽}) where {𝔽}

Return the real dimension of the [`AbstractProjectiveSpace`](@ref) `M`, respectively i.e.
the real dimension of the embedding minus the real dimension of the field `𝔽`.
"""
function manifold_dimension(M::AbstractProjectiveSpace{𝔽}) where {𝔽}
    return manifold_dimension(get_embedding(M)) - real_dimension(𝔽)
end

@doc raw"""
    manifold_volume(M::AbstractProjectiveSpace{ℝ})

Volume of the ``n``-dimensional [`AbstractProjectiveSpace`](@ref) `M`. The formula reads:

````math
\frac{\pi^{(n+1)/2}}{Γ((n+1)/2)},
````

where ``Γ`` denotes the [Gamma function](https://en.wikipedia.org/wiki/Gamma_function).
For details see [^BoyaSudarshanTilma2003].

[^BoyaSudarshanTilma2003]:
    > L. J. Boya, E. C. G. Sudarshan, and T. Tilma, “Volumes of Compact Manifolds,” Reports
    > on Mathematical Physics, vol. 52, no. 3, pp. 401–422, Dec. 2003,
    > doi: [10.1016/S0034-4877(03)80038-1](https://doi.org/10.1016/S0034-4877(03)80038-1)
"""
function manifold_volume(M::AbstractProjectiveSpace{ℝ})
    n = manifold_dimension(M) + 1
    return pi^(n / 2) / gamma(n / 2)
end

"""
    mean(
        M::AbstractProjectiveSpace,
        x::AbstractVector,
        [w::AbstractWeights,]
        method = GeodesicInterpolationWithinRadius(π/4);
        kwargs...,
    )

Compute the Riemannian [`mean`](@ref mean(M::AbstractManifold, args...)) of points in vector `x`
using [`GeodesicInterpolationWithinRadius`](@ref).
"""
mean(::AbstractProjectiveSpace, ::Any...)

function default_estimation_method(::AbstractProjectiveSpace, ::typeof(mean))
    return GeodesicInterpolationWithinRadius(π / 4)
end

function mid_point!(M::ProjectiveSpace, q, p1, p2)
    z = dot(p2, p1)
    λ = nzsign(z)
    q .= p1 .+ p2 .* λ
    project!(M, q, q)
    return q
end

@doc raw"""
    project(M::AbstractProjectiveSpace, p)

Orthogonally project the point `p` from the embedding onto the
[`AbstractProjectiveSpace`](@ref) `M`:
````math
\operatorname{proj}(p) = \frac{p}{\lVert p \rVert}_{\mathrm{F}},
````
where $\lVert ⋅ \rVert_{\mathrm{F}}$ denotes the Frobenius norm.
This is identical to projection onto the [`AbstractSphere`](@ref).
"""
project(::AbstractProjectiveSpace, ::Any)

project!(::AbstractProjectiveSpace, q, p) = (q .= p ./ norm(p))

@doc raw"""
    project(M::AbstractProjectiveSpace, p, X)

Orthogonally project the point `X` onto the tangent space at `p` on the
[`AbstractProjectiveSpace`](@ref) `M`:

````math
\operatorname{proj}_p (X) = X - p⟨p, X⟩_{\mathrm{F}},
````
where $⟨⋅, ⋅⟩_{\mathrm{F}}$ denotes the Frobenius inner product.
For the real [`AbstractSphere`](@ref) and `AbstractProjectiveSpace`, this projection is the
same.
"""
project(::AbstractProjectiveSpace, ::Any, ::Any)

project!(::AbstractProjectiveSpace, Y, p, X) = (Y .= X .- p .* dot(p, X))

@doc raw"""
    representation_size(M::AbstractProjectiveSpace)

Return the size points on the [`AbstractProjectiveSpace`](@ref) `M` are represented as,
i.e., the representation size of the embedding.
"""
@generated representation_size(::ArrayProjectiveSpace{N}) where {N} = size_to_tuple(N)
@generated representation_size(::ProjectiveSpace{N}) where {N} = (N + 1,)

@doc raw"""
    retract(M::AbstractProjectiveSpace, p, X, method::ProjectionRetraction)
    retract(M::AbstractProjectiveSpace, p, X, method::PolarRetraction)
    retract(M::AbstractProjectiveSpace, p, X, method::QRRetraction)

Compute the equivalent retraction [`ProjectionRetraction`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions.html#ManifoldsBase.ProjectionRetraction), [`PolarRetraction`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions.html#ManifoldsBase.PolarRetraction),
and [`QRRetraction`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions.html#ManifoldsBase.QRRetraction) on the [`AbstractProjectiveSpace`](@ref) manifold `M`$=𝔽ℙ^n$,
i.e.
````math
\operatorname{retr}_p X = \operatorname{proj}_p(p + X).
````

Note that this retraction is equivalent to the three corresponding retractions on
[`Grassmann(n+1,1,𝔽)`](@ref), where in this case they coincide.
For $ℝℙ^n$, it is the same as the `ProjectionRetraction` on the real [`Sphere`](@ref).
"""
retract(
    ::AbstractProjectiveSpace,
    p,
    X,
    ::Union{ProjectionRetraction,PolarRetraction,QRRetraction},
)

function retract_polar!(M::AbstractProjectiveSpace, q, p, X, t::Number)
    q .= p .+ t .* X
    return project!(M, q, q)
end
function retract_project!(M::AbstractProjectiveSpace, q, p, X, t::Number)
    q .= p .+ t .* X
    return project!(M, q, q)
end
function retract_qr!(M::AbstractProjectiveSpace, q, p, X, t::Number)
    q .= p .+ t .* X
    return project!(M, q, q)
end

function Base.show(io::IO, ::ProjectiveSpace{n,𝔽}) where {n,𝔽}
    return print(io, "ProjectiveSpace($(n), $(𝔽))")
end
function Base.show(io::IO, ::ArrayProjectiveSpace{N,𝔽}) where {N,𝔽}
    return print(io, "ArrayProjectiveSpace($(join(N.parameters, ", ")); field = $(𝔽))")
end

"""
    uniform_distribution(M::ProjectiveSpace{n,ℝ}, p) where {n}

Uniform distribution on given [`ProjectiveSpace`](@ref) `M`. Generated points will be of
similar type as `p`.
"""
function uniform_distribution(M::ProjectiveSpace{n,ℝ}, p) where {n}
    d = Distributions.MvNormal(zero(p), 1.0)
    return ProjectedPointDistribution(M, d, project!, p)
end

@doc raw"""
    parallel_transport_to(M::AbstractProjectiveSpace, p, X, q)

Parallel transport a vector `X` from the tangent space at a point `p` on the
[`AbstractProjectiveSpace`](@ref) `M`$=𝔽ℙ^n$ to the tangent space at another point `q`.

This implementation proceeds by transporting $X$ to $T_{q λ} M$ using the same approach as
[`parallel_transport_direction`](@ref parallel_transport_direction(::AbstractProjectiveSpace, p, X, d)),
where $λ = \frac{⟨q, p⟩_{\mathrm{F}}}{|⟨q, p⟩_{\mathrm{F}}|} ∈ 𝔽$ is the unit scalar that
takes $q$ to the member $q λ$ of its equivalence class $[q]$ closest to $p$ in the
embedding.
It then maps the transported vector from $T_{q λ} M$ to $T_{q} M$.
The resulting transport to $T_{q} M$ is
````math
\mathcal{P}_{q ← p}(X) = \left(X - \left(p \frac{\sin θ}{θ} + d \frac{1 - \cos θ}{θ^2}\right) ⟨d, X⟩_p\right) \overline{λ},
````
where $d = \log_p q$ is the direction of the transport, $θ = \lVert d \rVert_p$ is the
[`distance`](@ref distance(::AbstractProjectiveSpace, p, q)) between $p$ and $q$, and
$\overline{⋅}$ denotes complex or quaternionic conjugation.
"""
parallel_transport_to(::AbstractProjectiveSpace, ::Any, ::Any, ::Any)

function parallel_transport_to!(::AbstractProjectiveSpace, Y, p, X, q)
    z = dot(q, p)
    λ = nzsign(z)
    m = p .+ q .* λ # un-normalized midpoint
    mnorm2 = real(dot(m, m))
    factor = λ' * dot(q, X) * (2 / mnorm2) # λ' * dot(q, X) ≡ dot(q * λ, X)
    # multiply by λ' to bring from T_{\exp_p(\log_p q)} M to T_q M
    # this ensures that subsequent functions like `exp(M, q, Y)` do the right thing
    Y .= (X .- m .* factor) .* λ'
    return Y
end
function vector_transport_to_project!(M::AbstractProjectiveSpace, Y, p, X, q)
    project!(M, Y, q, X)
    return Y
end

@doc raw"""
    parallel_transport_direction(M::AbstractProjectiveSpace, p, X, d)

Parallel transport a vector `X` from the tangent space at a point `p` on the
[`AbstractProjectiveSpace`](@ref) `M` along the [`geodesic`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions.html#ManifoldsBase.geodesic-Tuple{AbstractManifold,%20Any,%20Any}) in the direction
indicated by the tangent vector `d`, i.e.
````math
\mathcal{P}_{\exp_p (d) ← p}(X) = X - \left(p \frac{\sin θ}{θ} + d \frac{1 - \cos θ}{θ^2}\right) ⟨d, X⟩_p,
````
where $θ = \lVert d \rVert$, and $⟨⋅, ⋅⟩_p$ is the [`inner`](@ref) product at the point $p$.
For the real projective space, this is equivalent to the same vector transport on the real
[`AbstractSphere`](@ref).
"""
parallel_transport_direction(::AbstractProjectiveSpace, ::Any, ::Any, ::Any)

function parallel_transport_direction!(M::AbstractProjectiveSpace, Y, p, X, d)
    θ = norm(M, p, d)
    cosθ = cos(θ)
    dX = inner(M, p, d, X)
    α = usinc(θ) * dX
    β = ifelse(iszero(θ), zero(cosθ), (1 - cosθ) / θ^2) * dX
    Y .= X .- p .* α .- d .* β
    return Y
end
