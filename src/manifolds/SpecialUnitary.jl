@doc raw"""
    const SpecialUnitaryMatrices{T} = GeneralUnitaryMatrices{ℂ, T, DeterminantOneMatrixType}

The manifold ``SU(n)`` of ``n×n`` complex matrices such that

```math
    p^{\mathrm{H}}p = \mathrm{I}_n \text{ and } \det(p) = 1,
```

where ``p^{\mathrm{H}}`` is the conjugate transpose of ``p`` and ``\mathrm{I}_n`` is the ``n×n`` identity matrix.

The tangent spaces are given by

```math
    T_pU(n) \coloneqq \bigl\{
    X \big| pY \text{ where } Y \text{ is skew symmetric and traceless, i. e. } Y = -Y^{\mathrm{H}} \text{ and } \operatorname{tr}(Y) = 0
    \bigr\}
```

But note that tangent vectors are represented in the Lie algebra, i.e. just using ``Y`` in
the representation above.

# Constructor

    SpecialUnitaryMatrices(n; parameter::Symbol = :type)

see also [`Rotations`](@ref) for the real valued case.
"""
const SpecialUnitaryMatrices{T} = GeneralUnitaryMatrices{ℂ, T, DeterminantOneMatrixType}

function SpecialUnitaryMatrices(n::Int; parameter::Symbol = :type)
    size = wrap_type_parameter(parameter, (n,))
    return SpecialUnitaryMatrices{typeof(size)}(size)
end

@doc raw"""
    manifold_dimension(M::SpecialUnitaryMatrices)

Return the dimension of the manifold of special unitary matrices.
```math
\dim_{\mathrm{SU}(n)} = n^2-1.
```
"""
function manifold_dimension(M::SpecialUnitaryMatrices)
    n = get_parameter(M.size)[1]
    return n^2 - 1
end

@doc raw"""
    manifold_volume(::SpecialUnitaryMatrices)

Volume of the manifold of complex general unitary matrices of determinant one. The formula
reads [BoyaSudarshanTilma:2003](@cite)

```math
\sqrt{n 2^{n-1}} π^{(n-1)(n+2)/2} \prod_{k=1}^{n-1}\frac{1}{k!}.
```
"""
function manifold_volume(M::SpecialUnitaryMatrices)
    n = get_parameter(M.size)[1]
    vol = sqrt(n * 2^(n - 1)) * π^(((n - 1) * (n + 2)) // 2)
    kf = 1
    for k in 1:(n - 1)
        kf *= k
        vol /= kf
    end
    return vol
end

@doc raw"""
    injectivity_radius(G::SpecialUnitaryMatrices)

Return the injectivity radius for general complex unitary matrix manifolds, where the determinant is ``+1``,
which is[^1]

```math
    \operatorname{inj}_{\mathrm{SU}(n)} = π \sqrt{2}.
```
[^1]
    > For a derivation of the injectivity radius, see [sethaxen.com/blog/2023/02/the-injectivity-radii-of-the-unitary-groups/](https://sethaxen.com/blog/2023/02/the-injectivity-radii-of-the-unitary-groups/).
"""
function injectivity_radius(::SpecialUnitaryMatrices)
    return π * sqrt(2.0)
end

@doc raw"""
    is_vector(M::SpecialUnitaryMatrices, p, X; atol, rtol, kwargs...)

Check whether `X` is a tangent vector at `p` on the [`SpecialUnitaryMatrices`](@ref) `M`, that is
whether `X` lies in the Lie algebra ``\mathfrak{su}(n)`` of the skew-Hermitian matrices of trace
zero, see Section 3.4, Proposition 3.24 of [Hall:2015](@cite).

The skew-Hermitian check is performed with `isapprox`, which receives `atol`, `rtol` and all
further keyword arguments. The trace has to vanish up to `max(atol, rtol * sqrt(n) * norm(X))`.
The relative tolerance `rtol` refers to the size of `X`; its default is the one of `isapprox`.
"""
is_vector(::SpecialUnitaryMatrices, ::Any, ::Any)

function check_vector(
        M::SpecialUnitaryMatrices, p, X::T;
        atol::Real = sqrt(prod(representation_size(M))) * eps(real(float(number_eltype(T)))),
        rtol::Real = sqrt(eps(real(float(number_eltype(T))))),
        kwargs...,
    ) where {T}
    n = get_parameter(M.size)[1]
    s = check_point(SkewHermitianMatrices(n, ℂ), X; atol = atol, rtol = rtol, kwargs...)
    s === nothing || return s
    t = abs(tr(X))
    if !(t <= atol || t <= rtol * sqrt(n) * norm(X))
        return DomainError(
            tr(X),
            "The tangent vector $(X) does not lie in the tangent space at $(p) of $(M), since its trace is $(tr(X)) and not zero.",
        )
    end
    return nothing
end

@doc raw"""
    project(M::SpecialUnitaryMatrices, p, X)
    project!(M::SpecialUnitaryMatrices, Y, p, X)

Orthogonally project ``X ∈ ℂ^{n×n}`` onto the tangent space of `M` at `p` and change the
representer to the Lie algebra ``\mathfrak{su}(n)``, that is compute the skew-Hermitian part
``Y`` of ``p^{\mathrm{H}}X`` and subtract ``\frac{1}{n}\operatorname{tr}(Y)`` from its diagonal, as the
projection on the [`DeterminantOneMatrices`](@ref) does.
"""
project(::SpecialUnitaryMatrices, p, X)

function project!(M::SpecialUnitaryMatrices, Y, p, X)
    n = get_parameter(M.size)[1]
    project!(SkewHermitianMatrices(n, ℂ), Y, p \ X)
    # the trace part is removed as on the determinant one matrices
    return project!(DeterminantOneMatrices(n, ℂ), Y, I, Y)
end

@doc raw"""
    rand(M::SpecialUnitaryMatrices; vector_at = nothing, σ::Real = 1.0)

Generate a random point on the [`SpecialUnitaryMatrices`](@ref) `M`, if `vector_at` is `nothing`,
as on the [`UnitaryMatrices`](@ref) from the QR decomposition of an ``n×n`` matrix, with its
first row divided by the sign of the determinant, so that the determinant is one.

Generate a tangent vector at `vector_at` by projecting a normally distributed matrix
onto the tangent space.
"""
rand(M::SpecialUnitaryMatrices; vector_at = nothing, σ::Real = 1.0)

function Random.rand!(
        rng::AbstractRNG, M::SpecialUnitaryMatrices, pX;
        vector_at = nothing, σ::Real = one(real(eltype(pX))),
    )
    if vector_at === nothing
        randn!(rng, pX)
        copyto!(pX, qr(pX).Q)
        det_pX = det(pX)
        pX[1, :] ./= sign(det_pX)
    else
        Z = σ * randn(rng, eltype(pX), size(pX))
        project!(M, pX, vector_at, Z)
    end
    return pX
end
