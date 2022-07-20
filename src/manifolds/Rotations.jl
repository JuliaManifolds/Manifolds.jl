@doc raw"""
    Rotations{N} <: AbstractManifold{ℝ}

The manifold of rotation matrices of sice ``n × n``, i.e.
real-valued orthogonal matrices with determinant ``+1``.

# Constructor

    Rotations(n)

Generate the manifold of ``n × n`` rotation matrices.
"""
const Rotations{n} = GeneralUnitaryMatrices{n,ℝ,DeterminantOneMatrices}

Rotations(n::Int) = Rotations{n}()

"""
    NormalRotationDistribution(M::Rotations, d::Distribution, x::TResult)

Distribution that returns a random point on the manifold [`Rotations`](@ref)
`M`. Random point is generated using base distribution `d` and the type
of the result is adjusted to `TResult`.

See [`normal_rotation_distribution`](@ref) for details.
"""
struct NormalRotationDistribution{TResult,TM<:Rotations,TD<:Distribution} <:
       MPointDistribution{TM}
    manifold::TM
    distr::TD
end

function NormalRotationDistribution(
    M::Rotations,
    d::Distribution,
    x::TResult,
) where {TResult}
    return NormalRotationDistribution{TResult,typeof(M),typeof(d)}(M, d)
end

@doc raw"""
    angles_4d_skew_sym_matrix(A)

The Lie algebra of [`Rotations(4)`](@ref) in $ℝ^{4 × 4}$, $𝔰𝔬(4)$, consists of $4 × 4$
skew-symmetric matrices. The unique imaginary components of their eigenvalues are the
angles of the two plane rotations. This function computes these more efficiently than
`eigvals`.

By convention, the returned values are sorted in decreasing order
(corresponding to the same ordering of _angles_ as
[`cos_angles_4d_rotation_matrix`](@ref)).
"""
function angles_4d_skew_sym_matrix(A)
    @assert size(A) == (4, 4)
    @inbounds begin
        halfb = (A[1, 2]^2 + A[1, 3]^2 + A[2, 3]^2 + A[1, 4]^2 + A[2, 4]^2 + A[3, 4]^2) / 2
        c = (A[1, 2] * A[3, 4] - A[1, 3] * A[2, 4] + A[1, 4] * A[2, 3])^2
    end
    sqrtdisc = sqrt(halfb^2 - c)
    return sqrt(halfb + sqrtdisc), sqrt(halfb - sqrtdisc)
end

# from https://github.com/JuliaManifolds/Manifolds.jl/issues/453#issuecomment-1046057557
function _get_tridiagonal_elements(trian)
    N = size(trian, 1)
    res = zeros(N)
    down = true
    for i in 1:N
        if i == N && down
            elem = 0
        else
            elem = trian[i + (down ? +1 : -1), i]
        end
        if elem ≈ 0
            res[i] = 0
        else
            res[i] = elem
            down = !down
        end
    end
    return res
end

function _ev_diagonal(tridiagonal_elements, unitary, evec, evals, fill_at; i)
    a = unitary[:, i]
    b = unitary[:, i + 1]
    evec[fill_at.x] = [-b a] * [a b]' ./ sqrt(2)
    evals[fill_at.x] = 0
    return fill_at.x += 1
end

function _ev_offdiagonal(tridiagonal_elements, unitary, evec, evals, fill_at; i, j)
    a = unitary[:, i]
    b = unitary[:, i + 1]
    c = unitary[:, j]
    d = unitary[:, j + 1]
    ref = hcat(a, b, c, d)' ./ 2

    evec[fill_at.x] = [-c -d a b] * ref
    evals[fill_at.x] = (tridiagonal_elements[i] - tridiagonal_elements[j])^2 / 4
    fill_at.x += 1
    evec[fill_at.x] = [-c d a -b] * ref
    evals[fill_at.x] = (tridiagonal_elements[i] + tridiagonal_elements[j])^2 / 4
    fill_at.x += 1
    evec[fill_at.x] = [-d -c b a] * ref
    evals[fill_at.x] = (tridiagonal_elements[i] + tridiagonal_elements[j])^2 / 4
    fill_at.x += 1
    evec[fill_at.x] = [d -c b -a] * ref
    evals[fill_at.x] = (tridiagonal_elements[i] - tridiagonal_elements[j])^2 / 4
    return fill_at.x += 1
end

function _ev_zero(tridiagonal_elements, unitary, evec, evals, fill_at; i)
    N = size(unitary, 1)
    ref = unitary[:, i]
    for idx in 1:(i - 1)
        rup = ref * unitary[:, idx]'
        evec[fill_at.x] = (rup - rup') ./ sqrt(2)
        evals[fill_at.x] = tridiagonal_elements[idx]^2 / 4
        fill_at.x += 1
    end
    return (values=evals, vectors=evec)
end

function get_basis_diagonalizing(
    M::Rotations{N},
    p,
    B::DiagonalizingOrthonormalBasis{ℝ},
) where {N}
    decomp = schur(B.frame_direction)
    decomp = ordschur(decomp, map(v -> norm(v) > eps(eltype(p)), decomp.values))

    trian_elem = _get_tridiagonal_elements(decomp.T)
    unitary = decomp.Z
    evec = Vector{typeof(B.frame_direction)}(undef, manifold_dimension(M))
    evals = Vector{eltype(B.frame_direction)}(undef, manifold_dimension(M))
    i = 1
    fill_at = Ref(1)
    while i <= N
        if trian_elem[i] == 0
            evs = _ev_zero(trian_elem, unitary, evec, evals, fill_at; i=i)
            i += 1
        else
            evs = _ev_diagonal(trian_elem, unitary, evec, evals, fill_at, i=i)
            j = 1
            while j < i
                # the zero case should have been handled earlier
                @assert trian_elem[j] != 0
                evs = _ev_offdiagonal(trian_elem, unitary, evec, evals, fill_at, i=i, j=j)
                j += 2
            end
            i += 2
        end
    end
    return CachedBasis(B, evals, evec)
end

@doc raw"""
    injectivity_radius(M::Rotations, ::PolarRetraction)

Return the radius of injectivity for the [`PolarRetraction`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions.html#ManifoldsBase.PolarRetraction) on the
[`Rotations`](@ref) `M` which is $\frac{π}{\sqrt{2}}$.
"""
injectivity_radius(::Rotations, ::PolarRetraction)
_injectivity_radius(::Rotations, ::PolarRetraction) = π / sqrt(2.0)

@doc raw"""
    inverse_retract(M, p, q, ::PolarInverseRetraction)

Compute a vector from the tangent space $T_p\mathrm{SO}(n)$
of the point `p` on the [`Rotations`](@ref) manifold `M`
with which the point `q` can be reached by the
[`PolarRetraction`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions.html#ManifoldsBase.PolarRetraction) from the point `p` after time 1.

The formula reads
````math
\operatorname{retr}^{-1}_p(q)
= -\frac{1}{2}(p^{\mathrm{T}}qs - (p^{\mathrm{T}}qs)^{\mathrm{T}})
````

where $s$ is the solution to the Sylvester equation

$p^{\mathrm{T}}qs + s(p^{\mathrm{T}}q)^{\mathrm{T}} + 2I_n = 0.$
"""
inverse_retract(::Rotations, ::Any, ::Any, ::PolarInverseRetraction)

@doc raw"""
    inverse_retract(M::Rotations, p, q, ::QRInverseRetraction)

Compute a vector from the tangent space $T_p\mathrm{SO}(n)$ of the point `p` on the
[`Rotations`](@ref) manifold `M` with which the point `q` can be reached by the
[`QRRetraction`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions.html#ManifoldsBase.QRRetraction) from the point `q` after time 1.
"""
inverse_retract(::Rotations, ::Any, ::Any, ::QRInverseRetraction)

function inverse_retract_polar!(M::Rotations{n}, X, p, q) where {n}
    A = transpose(p) * q
    Amat = A isa StaticMatrix ? A : convert(Matrix, A)
    H = copyto!(allocate(Amat), -2I)
    try
        B = lyap(A, H)
        mul!(X, A, B)
    catch e
        if isa(e, LinearAlgebra.LAPACKException)
            throw(OutOfInjectivityRadiusError())
        else
            rethrow()
        end
    end
    return project!(SkewSymmetricMatrices(n), X, p, X)
end
function inverse_retract_qr!(::Rotations{n}, X, p, q) where {n}
    A = transpose(p) * q
    R = zero(X)
    for i in 1:n
        b = zeros(i)
        b[end] = 1
        b[1:(end - 1)] = -transpose(R[1:(i - 1), 1:(i - 1)]) * A[i, 1:(i - 1)]
        R[1:i, i] = A[1:i, 1:i] \ b
    end
    mul!(X, A, R)
    return project!(SkewSymmetricMatrices(n), X, p, X)
end

@doc raw"""
    normal_rotation_distribution(M::Rotations, p, σ::Real)

Return a random point on the manifold [`Rotations`](@ref) `M`
by generating a (Gaussian) random orthogonal matrix with determinant $+1$. Let

$QR = A$

be the QR decomposition of a random matrix $A$, then the formula reads

$p = QD$

where $D$ is a diagonal matrix with the signs of the diagonal entries of $R$,
i.e.

$D_{ij}=\begin{cases} \operatorname{sgn}(R_{ij}) & \text{if} \; i=j \\ 0 & \, \text{otherwise} \end{cases}.$

It can happen that the matrix gets -1 as a determinant. In this case, the first
and second columns are swapped.

The argument `p` is used to determine the type of returned points.
"""
function normal_rotation_distribution(M::Rotations{N}, p, σ::Real) where {N}
    d = Distributions.MvNormal(zeros(N * N), σ)
    return NormalRotationDistribution(M, d, p)
end

@doc raw"""
    project(M::Rotations, p; check_det = true)

Project `p` to the nearest point on manifold `M`.

Given the singular value decomposition $p = U Σ V^\mathrm{T}$, with the
singular values sorted in descending order, the projection is

````math
\operatorname{proj}_{\mathrm{SO}(n)}(p) =
U\operatorname{diag}\left[1,1,…,\det(U V^\mathrm{T})\right] V^\mathrm{T}
````

The diagonal matrix ensures that the determinant of the result is $+1$.
If `p` is expected to be almost special orthogonal, then you may avoid this
check with `check_det = false`.
"""
project(::Rotations, ::Any)

function project!(::Rotations{N}, q, p; check_det=true) where {N}
    F = svd(p)
    mul!(q, F.U, F.Vt)
    if check_det && det(q) < 0
        d = similar(F.S)
        @inbounds fill!(view(d, 1:(N - 1)), 1)
        @inbounds d[N] = -1
        copyto!(q, F.U * Diagonal(d) * F.Vt)
    end
    return q
end

function Random.rand(
    rng::AbstractRNG,
    d::NormalRotationDistribution{TResult,Rotations{N}},
) where {TResult,N}
    return if N == 1
        convert(TResult, ones(1, 1))
    else
        A = reshape(rand(rng, d.distr), (N, N))
        convert(TResult, _fix_random_rotation(A))
    end
end
function Random.rand!(M::Rotations, pX; vector_at=nothing, σ::Real=one(eltype(pX)))
    if vector_at === nothing
        # Special case: Rotations(1) is just zero-dimensional
        (manifold_dimension(M) == 0) && return fill!(pX, 1)
        A = randn(representation_size(M))
        s = diag(sign.(qr(A).R))
        D = Diagonal(s)
        pX .= qr(A).Q * D
        if det(pX) < 0
            pX[:, [1, 2]] = pX[:, [2, 1]]
        end
    else
        # Special case: Rotations(1) is just zero-dimensional
        (manifold_dimension(M) == 0) && return fill!(pX, 0)
        A = σ .* randn(representation_size(M))
        pX .= triu(A, 1) .- transpose(triu(A, 1))
        normalize!(pX)
    end
    return pX
end
function Random.rand!(
    rng::AbstractRNG,
    M::Rotations,
    pX;
    vector_at=nothing,
    σ::Real=one(eltype(pX)),
)
    if vector_at === nothing
        # Special case: Rotations(1) is just zero-dimensional
        (manifold_dimension(M) == 0) && return fill!(pX, 1)
        A = randn(rng, representation_size(M))
        s = diag(sign.(qr(A).R))
        D = Diagonal(s)
        pX .= qr(A).Q * D
        if det(pX) < 0
            pX[:, [1, 2]] = pX[:, [2, 1]]
        end
    else
        # Special case: Rotations(1) is just zero-dimensional
        (manifold_dimension(M) == 0) && return fill!(pX, 0)
        A = σ .* randn(rng, representation_size(M))
        pX .= triu(A, 1) .- transpose(triu(A, 1))
        normalize!(pX)
    end
    return pX
end

function Distributions._rand!(
    rng::AbstractRNG,
    d::NormalRotationDistribution{TResult,Rotations{N}},
    x::AbstractArray{<:Real},
) where {TResult,N}
    return copyto!(x, rand(rng, d))
end

function _fix_random_rotation(A::AbstractMatrix)
    s = diag(sign.(qr(A).R))
    D = Diagonal(s)
    C = qr(A).Q * D
    if det(C) < 0
        C[:, [1, 2]] = C[:, [2, 1]]
    end
    return C
end

@doc raw"""
    parallel_transport_direction(M::Rotations, p, X, d)

Compute parallel transport of vector `X` tangent at `p` on the [`Rotations`](@ref)
manifold in the direction `d`. The formula, provided in [^Rentmeesters2011], reads:

```math
\mathcal P_{q\gets p}X = q^\mathrm{T}p \operatorname{Exp}(d/2) X \operatorname{Exp}(d/2)
```
where ``q=\exp_p d``.

The formula simplifies to identity for 2-D rotations.

[^Rentmeesters2011]:
    > Rentmeesters Q., “A gradient method for geodesic data fitting on some symmetric
    > Riemannian manifolds,” in 2011 50th IEEE Conference on Decision and Control and
    > European Control Conference, Dec. 2011, pp. 7141–7146. doi: [10.1109/CDC.2011.6161280](https://doi.org/10.1109/CDC.2011.6161280).
"""
parallel_transport_direction(M::Rotations, p, X, d)

function parallel_transport_direction!(M::Rotations, Y, p, X, d)
    expdhalf = exp(d / 2)
    q = exp(M, p, d)
    return copyto!(Y, transpose(q) * p * expdhalf * X * expdhalf)
end
function parallel_transport_direction!(::Rotations{2}, Y, p, X, d)
    return copyto!(Y, X)
end
function parallel_transport_direction(M::Rotations, p, X, d)
    expdhalf = exp(d / 2)
    q = exp(M, p, d)
    return transpose(q) * p * expdhalf * X * expdhalf
end
parallel_transport_direction(::Rotations{2}, p, X, d) = X

function parallel_transport_to!(M::Rotations, Y, p, X, q)
    d = log(M, p, q)
    expdhalf = exp(d / 2)
    return copyto!(Y, transpose(q) * p * expdhalf * X * expdhalf)
end
function parallel_transport_to!(::Rotations{2}, Y, p, X, q)
    return copyto!(Y, X)
end
function parallel_transport_to(M::Rotations, p, X, q)
    d = log(M, p, q)
    expdhalf = exp(d / 2)
    return transpose(q) * p * expdhalf * X * expdhalf
end
parallel_transport_to(::Rotations{2}, p, X, q) = X

function Base.show(io::IO, ::Rotations{n}) where {n}
    return print(io, "Rotations($(n))")
end

Distributions.support(d::NormalRotationDistribution) = MPointSupport(d.manifold)

@doc raw"""
    zero_vector(M::Rotations, p)

Return the zero tangent vector from the tangent space art `p` on the [`Rotations`](@ref)
as an element of the Lie group, i.e. the zero matrix.
"""
zero_vector(::Rotations, p) = zero(p)

zero_vector!(::Rotations, X, p) = fill!(X, 0)
