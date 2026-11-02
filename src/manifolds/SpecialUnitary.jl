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
    n < 1 && throw(
        DomainError(n, "The matrix size n of special unitary matrices must be positive, but was $(n).")
    )
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

Volume of the manifold of complex general unitary matrices of determinant one with respect
to the metric of [`inner`](@ref), Eqs. (3.9) and (A22) in [ZyczkowskiSommers:2003](@cite):

```math
\sqrt{n} (2π)^{(n-1)(n+2)/2} \prod_{k=1}^{n-1}\frac{1}{k!}.
```
"""
function manifold_volume(M::SpecialUnitaryMatrices)
    n = get_parameter(M.size)[1]
    vol = sqrt(n) * (2π)^div((n - 1) * (n + 2), 2)
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
[^1]:
    > For a derivation of the injectivity radius, see [sethaxen.com/blog/2023/02/the-injectivity-radii-of-the-unitary-groups/](https://sethaxen.com/blog/2023/02/the-injectivity-radii-of-the-unitary-groups/).
"""
function injectivity_radius(::SpecialUnitaryMatrices)
    return π * sqrt(2.0)
end

@doc raw"""
    is_vector(M::SpecialUnitaryMatrices, p, X; atol, rtol, kwargs...)

Check whether `X` is a tangent vector at `p` on the [`SpecialUnitaryMatrices`](@ref) `M`.
That is the case if `X` lies in the Lie algebra ``\mathfrak{su}(n)`` of the skew-Hermitian
matrices of trace zero, see Section 3.4, Proposition 3.24 of [Hall:2015](@cite).

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
representer to the Lie algebra ``\mathfrak{su}(n)``. That is, compute the skew-Hermitian part
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

Generate a random point on the [`SpecialUnitaryMatrices`](@ref) `M`, if `vector_at` is `nothing`.
The point is drawn as on the [`UnitaryMatrices`](@ref) from the QR decomposition of an ``n×n``
matrix, with its first row divided by the sign of the determinant, so that the determinant is one.

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

@doc raw"""
    get_coordinates(M::SpecialUnitaryMatrices, p, X, B::DefaultOrthonormalBasis)

Compute the coordinates of the tangent vector `X` at `p` on the [`SpecialUnitaryMatrices`](@ref)
`M` in an orthonormal basis of the Lie algebra ``\mathfrak{su}(n)``. As on the
[`UnitaryMatrices`](@ref), each strict upper-triangular entry contributes its real and imaginary
part, both scaled by ``\sqrt{2}``. The last ``n-1`` coordinates are those of the imaginary part
of the diagonal, which sums to zero, in the orthonormal basis given by the rows of the
[`helmert_submatrix`](@ref Manifolds.helmert_submatrix).
"""
get_coordinates(::SpecialUnitaryMatrices, p, X, ::DefaultOrthonormalBasis)

function get_coordinates_orthonormal!(M::SpecialUnitaryMatrices, c, p, X, ::RealNumbers)
    n = get_parameter(M.size)[1]
    k = firstindex(c)
    for i in 1:n, j in (i + 1):n
        c[k] = real(X[i, j]) * sqrt(2)
        c[k + 1] = imag(X[i, j]) * sqrt(2)
        k += 2
    end
    H = helmert_submatrix(real(eltype(X)), n)
    mul!(view(c, k:lastindex(c)), H, imag.(diag(X)))
    return c
end

@doc raw"""
    get_vector(M::SpecialUnitaryMatrices, p, c, B::DefaultOrthonormalBasis)

Compute the tangent vector at `p` on the [`SpecialUnitaryMatrices`](@ref) `M` with the
coordinates `c` in the basis that
[`get_coordinates`](@ref get_coordinates(::SpecialUnitaryMatrices, ::Any, ::Any, ::DefaultOrthonormalBasis))
uses.
"""
get_vector(::SpecialUnitaryMatrices, p, c, ::DefaultOrthonormalBasis)

function get_vector_orthonormal!(M::SpecialUnitaryMatrices, Y, p, c, ::RealNumbers)
    n = get_parameter(M.size)[1]
    k = firstindex(c)
    for i in 1:n, j in (i + 1):n
        Y[i, j] = (c[k] + c[k + 1] * 1im) / sqrt(2)
        Y[j, i] = -conj(Y[i, j])
        k += 2
    end
    H = helmert_submatrix(real(eltype(Y)), n)
    d = transpose(H) * view(c, k:lastindex(c))
    for i in 1:n
        Y[i, i] = d[i] * 1im
    end
    return Y
end


@doc raw"""
    project(M::SpecialUnitaryMatrices, p)

Project the matrix ``p ∈ ℂ^{n×n}`` onto the [`SpecialUnitaryMatrices`](@ref) `M`. With the
singular value decomposition ``p = USV^{\mathrm{H}}``, the unitary polar factor
``UV^{\mathrm{H}}`` is divided by the principal ``n``-th root of its determinant, i.e.

```math
\operatorname{proj}(p) = \frac{UV^{\mathrm{H}}}{\det(UV^{\mathrm{H}})^{1/n}}.
```

This is the point on `M` nearest to `p` in the Frobenius norm if all singular values of `p`
are equal, for example if `p` is unitary, and an approximation of it otherwise.
"""
project(::SpecialUnitaryMatrices, ::Any)

function project!(M::SpecialUnitaryMatrices, q, p)
    n = get_parameter(M.size)[1]
    F = svd(p)
    mul!(q, F.U, F.Vt)
    q ./= det(q)^(1 / n)
    return q
end

@doc raw"""
    riemannian_Hessian(M::SpecialUnitaryMatrices, p, G, H, X)

Compute the Riemannian Hessian on the [`SpecialUnitaryMatrices`](@ref) `M` from the Euclidean
gradient `G` and the Euclidean Hessian `H` applied to `X`. Since ``\mathrm{SU}(n)`` is a totally
geodesic submanifold of the [`UnitaryMatrices`](@ref), this is the projection of the
[`riemannian_Hessian`](@ref riemannian_Hessian(::UnitaryMatrices, ::Any, ::Any, ::Any, ::Any))
there onto ``\mathfrak{su}(n)``, i.e. its trace is removed.
"""
riemannian_Hessian(M::SpecialUnitaryMatrices, p, G, H, X)

function riemannian_Hessian!(M::SpecialUnitaryMatrices, Y, p, G, H, X)
    symmetrize!(Y, G' * p)
    return project!(M, Y, I, p' * H - X * Y)
end

@doc raw"""
    sectional_curvature(M::SpecialUnitaryMatrices, p, X, Y)

Compute the sectional curvature of the plane spanned by `X` and `Y` at `p` on the
[`SpecialUnitaryMatrices`](@ref) `M` as on the
[`GeneralUnitaryMatrices`](@ref sectional_curvature(::GeneralUnitaryMatrices, ::Any, ::Any, ::Any)).
For ``n = 2`` it is the constant ``\frac{1}{2}``.
"""
function sectional_curvature(M::SpecialUnitaryMatrices, p, X, Y)
    get_parameter(M.size)[1] == 2 && return 0.5
    return _bi_invariant_sectional_curvature(X, Y)
end

@doc raw"""
    sectional_curvature_max(M::SpecialUnitaryMatrices)

Return the maximal sectional curvature of the [`SpecialUnitaryMatrices`](@ref) `M`, which is
``\frac{1}{2}`` for ``n ≥ 2`` and ``0`` for ``n = 1``. It is attained on the plane spanned by
two orthonormal elements of a subalgebra ``\mathfrak{su}(2)``, and it is an upper bound by
the [`sectional_curvature`](@ref sectional_curvature(::GeneralUnitaryMatrices, ::Any, ::Any, ::Any))
and ``\lVert [X, Y] \rVert ≤ \sqrt{2}\lVert X \rVert \lVert Y \rVert``, see [BoettcherWenzel:2008](@cite).
"""
function sectional_curvature_max(M::SpecialUnitaryMatrices)
    return get_parameter(M.size)[1] >= 2 ? 0.5 : 0.0
end

@doc raw"""
    sectional_curvature_min(M::SpecialUnitaryMatrices)

Return the minimal sectional curvature of the [`SpecialUnitaryMatrices`](@ref) `M`, which is
``0`` for ``n ≠ 2``, attained by commuting diagonal tangent vectors for ``n ≥ 3``, and
``\frac{1}{2}`` for ``n = 2``, since ``\mathrm{SU}(2)`` has constant sectional curvature.
"""
function sectional_curvature_min(M::SpecialUnitaryMatrices)
    return get_parameter(M.size)[1] == 2 ? 0.5 : 0.0
end

@doc raw"""
    Weingarten(M::SpecialUnitaryMatrices, p, X, V)

Compute the Weingarten map ``\mathcal W_p`` at `p` on the [`SpecialUnitaryMatrices`](@ref) `M`
with respect to the tangent vector ``X \in T_p\mathcal M`` and the normal vector
``V \in N_p\mathcal M``. Since ``\mathrm{SU}(n)`` is a totally geodesic submanifold of the
[`UnitaryMatrices`](@ref), it is the projection of the Weingarten map there, see
[AbsilMahonyTrumpf:2013](@cite), onto ``\mathfrak{su}(n)``, i.e.

```math
\mathcal W_p(X,V) = \operatorname{proj}_{\mathfrak{su}(n)}\Bigl(-\frac{1}{2}\bigl(XS + SX\bigr)\Bigr),
\qquad S = p^{\mathrm{H}}V,
```
where the projection takes the skew-Hermitian part and removes its trace.
"""
Weingarten(::SpecialUnitaryMatrices, p, X, V)

function Weingarten!(M::SpecialUnitaryMatrices, Y, p, X, V)
    S = p' * V
    Y .= -(X * S .+ S * X) ./ 2
    return project!(M, Y, I, Y)
end

function Base.show(io::IO, ::SpecialUnitaryMatrices{TypeParameter{Tuple{n}}}) where {n}
    return print(io, "SpecialUnitaryMatrices($(n))")
end
function Base.show(io::IO, M::SpecialUnitaryMatrices{Tuple{Int}})
    n = get_parameter(M.size)[1]
    return print(io, "SpecialUnitaryMatrices($n; parameter=:field)")
end
