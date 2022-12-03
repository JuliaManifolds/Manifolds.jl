@doc doc"""
    q, dot_q = dot_exp(M::MetricManifold{ℝ, Stiefel{n,k,ℝ}, <:StiefelSubmersionMetric}, p, X)
    dot_exp!(M::MetricManifold{ℝ, Stiefel{n,k,ℝ}, <:StiefelSubmersionMetric}, q, dot_q, p, X)

Compute the geodesic end point ``q = q(t)`` and its time derivative ``dot\_q = \dot{q}(t)`` on [`Stiefel(n,k)`](@ref) with the [`StiefelSubmersionMetric`](@ref). The geodesic is given by
````math
\operatorname{Exp} tX = \begin{bmatrix} p & X\end{bmatrix}
\{\exp t\begin{bmatrix} (2\bar{\alpha}-1)A & 2(\bar{\alpha}-1)A^2-X^{\mathrm{T}}X \\ I_k & A\end{bmatrix}\}
\begin{bmatrix}\ \exp((1-2\bar{\alpha})tA) \\ 0\end{bmatrix}.
````
with ``\bar{\alpha} = \frac{1}{2(1+\alpha)}, A = p^{\mathrm{T}}X``. This implementation is based on [^Nguyen2022], equation (3).

[^Nguyen2022]:
    ^Nguyen, D. (2022),
    ^"Closed-form Geodesics and Optimization for Riemannian Logarithms of Stiefel and Flag Manifolds".
    ^"J Optim Theory Appl 194, 142–166 (2022)."
    ^doi: https://doi.org/10.1007/s10957-022-02012-3
    ^url https://rdcu.be/c0V2r
"""
function dot_exp(
    M::MetricManifold{ℝ,Stiefel{n,k,ℝ},<:StiefelSubmersionMetric},
    p,
    X,
    t,
) where {n,k}
    q = similar(p)
    dot_q = similar(p)
    dot_exp!(M, q, dot_q, p, X, t)
    return q, dot_q
end

function dot_exp!(
    M::MetricManifold{ℝ,Stiefel{n,k,ℝ},<:StiefelSubmersionMetric},
    q,
    dot_q,
    p,
    X,
    t,
) where {n,k}
    # return the other exp formula, and also
    # time derivative of the exponential map
    α = metric(M).α
    T = Base.promote_eltype(p, X)

    alf = 0.5 / (1 + α)
    A = p' * X
    e_mat = Array{T,2}(undef, 2 * k, 2 * k)
    e_mat[1:k, 1:k] = (2 * alf - 1) * A
    e_mat[1:k, (k + 1):end] = -X' * X - 2 * (1 - alf) * A * A
    e_mat[(k + 1):end, 1:k] = I(k)
    e_mat[(k + 1):end, (k + 1):end] = A
    eE = exp(t * e_mat)
    eA = exp((1 - 2 * alf) * t * A)
    q .= ([p X] * eE)[1:end, 1:k] * eA
    dot_q .=
        (vcat([p X]) * e_mat * eE)[1:end, 1:k] * eA +
        (vcat([p X]) * eE)[1:end, 1:k] * ((1 - 2 * alf) * A * eA)
    return q, dot_q
end

@doc doc"""
    log_lbfgs(Stf::MetricManifold{ℝ,Stiefel{n,k,ℝ},<:StiefelSubmersionMetric}, Y, W;
                       tolerance=sqrt(eps(float(real(Base.promote_eltype(Y, W))))),
                       max_itr=1_000, pretol=1e-3, lbfgs_options=nothing)

Compute the Riemannian logarithm on [`Stiefel(n,k)`](@ref) with the [`StiefelSubmersionMetric`](@ref). Returning ``\eta`` such that ``\operatorname{Exp}_Y\eta = W``.

The logarithmic map is computed by finding the initial velocity ``\eta`` such that if ``\gamma`` is the geodesic with initial conditions ``\gamma(0) = Y, \dot{\gamma}(0) = \eta``, 
the cost function ``|\gamma(1) - W|^2`` (the square Euclidean distance) is minimized. The exact gradient of the cost function is computed using Fréchet derivatives. The function is optimized using an [L-BFGS optimizer `minimize`](@ref).

The vector space spanned by $Y$ and $W$ intersects with the original Stiefel manifold along a smaller Stiefel manifold in ``ℝ^{\dim(span(Y, W))\times k}``. Let $Q$ be the complementary basis of $Y$ in ``span(Y, W)``, so ``[Y|Q]`` is an orthogonal basis for ``span(Y, W)``, we solve for two matrices ``A, R`` representing the tangent vector in this smaller Stiefel manifold (following [^Zimmermann2017]):

```math
 \eta = YA + QR
```

The exact gradient calculation with Fréchet derivatives is relatively expensive, to speed up we only run the exact gradient until the iteration error is less than pretol, then run a simple update on ``A, R`` for the last steps. A pretol=1e-5 is used if the distance is expected to be over $\frac{1}{2}\pi$. Otherwise, use pretol=1e-3.

``lbfgs\_options`` is a dictionary of technical parameters, isolated to a separate group as we do not expect users to modify them. The relevant keys are\

* ``complementary\_rank\_cutoff`` (default = 1e-14): cut off of eigenvalue to determine the rank of the complementary basis ``Q``. We use SVD to determine the size of the complement basis ``Q`` of ``Y`` in ``span(Y, W)``. We keep columns of $Q$ corresponding to eigenvalues higher than this cutoff.\

*  The remaining parameters are related to the [L-BFGS minimizer `minimize`](@ref):\

    +  ``corrections`` (default = 10): default number of stored history in L-BFGS.\
    + ``c1`` (default = 1e-4)  default c1, c2 in the line search.\
    + ``c2`` (default = 0.9).\
    +  ``max\_ls`` (default = 25) max iterations for the line search.
"""
function log_lbfgs(
    Stf,
    Y,
    W;
    tolerance=sqrt(eps(float(real(Base.promote_eltype(Y, W))))),
    max_itr=1_000,
    pretol=1e-3,
    lbfgs_options=nothing,
)
    # these are default values - overide in lbfgs_options.    
    if !isnothing(lbfgs_options)
        complementary_rank_cutoff = get(lbfgs_options, "complementary_rank_cutoff", 1e-14)
        corrections = get(lbfgs_options, "corrections", 10)
        c1 = get(lbfgs_options, "c1", 1e-4)
        c2 = get(lbfgs_options, "c2", .9)
        max_ls = get(lbfgs_options, "max_ls", 25)
    end

    α = metric(Stf).α
    alf = 0.5 / (1 + α)
    T = Base.promote_eltype(Y, W, complementary_rank_cutoff)
    n, p = size(Y)

    Q = _get_complementary_basis(Y, W, complementary_rank_cutoff)
    k = size(Q, 2)

    if k == 0
        # q and p has the same linear span
        A = log(Y' * W)
        return Y * A
    end

    eta0 = project(Stf, Y, W - Y)
    A = asym(Y' * eta0)
    R = Q' * eta0 - (Q' * Y) * (Y' * eta0)
    Adim = div(p * (p - 1), 2)

    # W is Z in the paper.
    WTY = W' * Y
    WTQ = W' * Q

    mat = zeros(T, p + k, p + k)
    E = zeros(T, p + k, p + k)
    v0 = Array{T,1}(undef, Adim + k * p)
    buff = Array{T,2}(undef, 16 * (p + k), p + k)
    buffA = Array{T,2}(undef, 16 * p, p)

    @views begin
        mat11 = mat[1:p, 1:p]
        mat12 = mat[1:p, (p + 1):end]
        mat21 = mat[(p + 1):end, 1:p]
    end

    @views begin
        E11 = E[1:p, 1:p]
        E12 = E[1:p, (p + 1):end]
    end

    WYMQN = Array{T,2}(undef, p, p)
    dA = similar(A)
    dR = similar(R)
    ex1 = similar(A)
    @views begin
        ex2 = buff[1:(k + p), :]
        fe2 = buff[(k + p + 1):(2 * (k + p)), :]
        M = buff[1:p, 1:p]
        N = buff[(p + 1):(p + k), 1:p]
    end

    # unravel v to A and R
    @inline function vec2AR!(A, R, v)
        A .= vec2asym(v[1:Adim])
        return R .= reshape(v[(Adim + 1):end], k, p)
    end

    # compute ex1= exp((1-2*alf)*A) and populate
    # the (p+k) × (p+k) matrix mat in the Exp formula. ex2=exp(mat)
    @inline function setMatForExp!(mat, ex1, ex2, A, R)
        mul!(mat11, 2 * alf, A)
        mul!(mat12, -1, R')
        return copyto!(mat21, R)
    end

    # cost function and gradient in the format expected by minimize
    function cost_and_grad!(F, G, v)
        vec2AR!(A, R, v)
        ex1 .= exp((1 - 2 * alf) * A)
        setMatForExp!(mat, ex1, ex2, A, R)
        mul!(E11, ex1, WTY)
        mul!(E12, ex1, WTQ)

        if isnothing(G)
            ex2 .= exp(mat)
            return -sum((WTY * M + WTQ * N)' .* ex1)
        end
        # ex2, fe2 = Manifolds.expm_frechet(mat, E)
        # expm affects buff, so affect ex2, fe2
        expm_frechet!(buff, mat, E)

        mul!(WYMQN, WTY, M)
        mul!(WYMQN, WTQ, N, 1, 1)

        expm_frechet!(buffA, (1 - 2 * alf) * A, WYMQN)
        dA .=
            asym((1 - 2 * alf) * buffA[(p + 1):(2 * p), :]) .+ 2 * alf * asym(fe2[1:p, 1:p])

        dR .= fe2[(p + 1):end, 1:p] - fe2[1:p, (p + 1):end]'

        G[1:Adim] .= asym2vec(dA)
        G[(1 + Adim):end] .= vec(dR)
        return -sum(WYMQN' .* ex1)
    end
    v0[1:Adim] = asym2vec(A)
    v0[(Adim + 1):end] = vec(R)

    # run full gradient until pretol
    xret, f, exitflag, output = minimize(
        cost_and_grad!,
        v0,
        max_fun_evals=max_fun_evals,
        max_itr=max_itr,
        grad_tol=pretol,
        func_tol=pretol,
        corrections=_g_corrections,
        c1=c1,
        c2=c2,
        max_ls=max_ls,
    )

    vec2AR!(A, R, xret)

    # run simple descent
    function Fnorm(dA, dR)
        return norm(dA, 2) + norm(dR, 2)
    end
    disc = Fnorm(dA, dR)
    for i in 1:max_itr
        ex1 .= exp((1 - 2 * alf) * A)
        setMatForExp!(mat, ex1, ex2, A, R)
        # this operation changes ex2, so change M, N        
        ex2 .= exp(mat)
        dA .= asym(WTY' - M * ex1)
        broadcast!(-, dR, WTQ', N * ex1)
        new_d = Fnorm(dA, dR)
        if new_d < disc
            A .+= dA
            R .+= dR
            disc = new_d
        else
            A .+= dA * disc / new_d
            R .+= dR * disc / new_d
        end
        if (
            maximum(abs.(extrema(dA))) < tolerance &&
            maximum(abs.(extrema(dR))) < tolerance
        )
            break
        end
    end

    eta = similar(Y)
    mul!(eta, Y, A)
    mul!(eta, Q, R, 1, 1)

    # return Y * A + Q*R, ret
    return eta
end

# Note: the following are for internal use
@doc raw"""
    _get_complementary_basis(Y, Y1)
Find a complementary basis Q in the linear span of Y Y1
orthogonal to Y
"""    
function _get_complementary_basis(Y, Y1, complementary_rank_cutoff)
    n, p = size(Y)
    F = svd([Y Y1])
    k = sum(F.S .> complementary_rank_cutoff)
    good = F.U[:, 1:k] * F.Vt[1:k, 1:k]
    qs = nullspace(Y' * good)
    QR = qr(good * qs)
    return QR.Q * vcat(I, zeros((n - k + p, k - p)))
end


@doc raw"""
    sym(mat)
symmetrize mat
"""
@inline sym(mat) = 0.5 * (mat + mat')

@doc raw"""
     asym(mat)
anti-symmetrize mat
"""
@inline asym(mat) = 0.5 * (mat - mat')

@doc raw"""
    asym2vec(mat)
vectorize antisymmetric matrice mat
"""
function asym2vec(mat)
    sz = size(mat)[1]
    ret = zeros(div((sz * (sz - 1)), 2))
    start = 1
    for i in 1:(sz - 1)
        ret[start:(start + sz - i - 1)] .= mat[(i + 1):end, i]
        start += sz - i
    end
    return ret
end

@doc raw"""
    vec2asym(v)
unravel v to an antisymmetric matrices
"""
function vec2asym(v)
    sz = 0.5 * (1 + sqrt(1 + 8 * size(v)[1]))
    sz = Int(round(sz))
    mat = zeros(sz, sz)
    start = 1
    for i in 1:(sz - 1)
        mat[(i + 1):end, i] .= v[start:(start + sz - i - 1)]
        mat[i, (i + 1):end] .= -v[start:(start + sz - i - 1)]
        start += sz - i
    end
    return mat
end
