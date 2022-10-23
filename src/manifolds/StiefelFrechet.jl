import Optim as Opm

_g_rank_cutoff = 1e-14  # cut off of eigenvalue to determine rank of span of two orthogonal basis

@doc raw"""
    q, dot_q = dot_exp(M::MetricManifold{ℝ, Stiefel{n,k,ℝ}, <:StiefelSubmersionMetric}, p, X)
    dot_exp!(M::MetricManifold{ℝ, Stiefel{n,k,ℝ}, <:StiefelSubmersionMetric}, q, dot_q, p, X)

Compute the geodesic end point q = Y(t) and its time derivative the [`Stiefel(n,k)`](@ref) manifold with respect to the
[`StiefelSubmersionMetric`](@ref).

with ````math\bar{\alpha} = \frac{1}{2(1+\alpha)}, A = p^{\mathrm{T}}X```` the geodesic is given by
````math
\operatorname{Exp} tX = \begin{bmatrix} p & X\end{bmatrix}
\{\exp t\begin{bmatrix} (2\bar{\alpha}-1)A & 2(\bar{\alpha}-1)A^2-X^{\mathrm{T}}X \\ I_p & A\end{bmatrix}\}
\begin{bmatrix}\ \exp((1-2\bar{\alpha})tA) \\ 0\end{bmatrix}
````
This implementation is based on [^Nguyen2022].
"""
function dot_exp(
    M::MetricManifold{ℝ,Stiefel{n,k,ℝ},<:StiefelSubmersionMetric},
    p,
    X,
    t
) where {n, k}
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
    t
) where {n, k}
    # return the other exp formula, and also
    # time derivative of the exponential map
    α = metric(M).α
    T = Base.promote_eltype(p, X)
    
    alf = 0.5/(1+α)
    A = p' * X
    e_mat = Array{T, 2}(undef, 2*k, 2*k)
    e_mat[1:k, 1:k] = (2*alf-1)*A 
    e_mat[1:k, k+1:end] = -X'*X - 2*(1-alf)*A*A
    e_mat[k+1:end, 1:k] = I(k)
    e_mat[k+1:end, k+1:end] = A
    eE = exp(t*e_mat)
    eA = exp((1-2*alf)*t*A)
    q .= ([p X] * eE)[1:end, 1:k] * eA
    dot_q .= (vcat([p X]) * e_mat*eE)[1:end, 1:k] * eA +
        (vcat([p X]) * eE)[1:end, 1:k] * ((1-2*alf)*A*eA)
   return q, dot_q
end

@doc raw"""
    log_lbfgs(Stf::MetricManifold{ℝ,Stiefel{n,k,ℝ},<:StiefelSubmersionMetric}, Y, W; kwargs...)

Compute the logarithmic map on the [`Stiefel(n,k)`](@ref) manifold with respect to the [`StiefelSubmersionMetric`](@ref). Returning eta such that ``\operatorname{Exp}_Y\eta = W``

The logarithmic map is computed by minimizing the square norm of [`GradientEuclideanDistance`](@ref) using the LBFGS optimizer.
The exact gradient of the square Euclidean distant is computed using Frechet derivatives

The vector space spanned by $Y$ and $W$ intersects with the Stiefel manifold along a new Stiefel manifold of size $\dim(span(Y, W)*k$.
Express $span(Y, W)$ in an orthogonal basis $[Y, Q]$, we solve for two matrices $A, R$ representing the tangent space in this smaller Stiefel manifold (following Zimmerman 2014).
````math \eta = Y*A + Q*R. ````
The exact gradient calculation with Frechet derivatives is expensive, to speed up we only run the exact gradient until the iteration error is less than pretol, then run a simple update on A, R for the last steps.
If we suspect the distance between p and q is over 0.5pi, use pretol=1e-5.  Otherwise pretol=1e-3 is much faster for short distance.
"""
function log_lbfgs(Stf, Y, W;
                   tolerance=sqrt(eps(float(real(Base.promote_eltype(Y, W))))),
                   max_itr=1_000, pretol=1e-3)
    α = metric(Stf).α
    alf = 0.5/(1+α)
    T = Base.promote_eltype(Y, W)        
    n, p = size(Y)

    Q = _get_Q(Y, W)
    k = size(Q, 2)

    eta0 = project(Stf, Y, W-Y)
    A = asym(Y' * eta0)
    R = Q' * eta0 - (Q' * Y) * (Y' * eta0)
    Adim = div(p*(p-1), 2)

    # W is Z in the paper.
    WTY = W'*Y
    WTQ = W'*Q

    mat = zeros(T, p+k, p+k)
    E = zeros(T, p+k, p+k)
    v0 = Array{T, 1}(undef, Adim + k*p)
    buff = Array{T, 2}(undef, 16*(p+k), p+k)
    buffA = Array{T, 2}(undef, 16*p, p)    

    @views begin
        mat11 = mat[1:p, 1:p]
        mat12 = mat[1:p, p+1:end]
        mat21 = mat[p+1:end, 1:p]
    end

    @views begin
        E11 = E[1:p, 1:p]
        E12 = E[1:p, p+1:end]
    end

    WYMQN = Array{T, 2}(undef, p, p)
    dA = similar(A)
    dR = similar(R)
    ex1 = similar(A)
    @views begin
        ex2 = buff[1:k+p, :]
        fe2 = buff[k+p+1:2*(k+p), :]
        M = buff[1:p, 1:p]
        N = buff[p+1:p+k, 1:p]                
    end

    # unravel v to A and R
    @inline function unvecAR!(A, R, v)
        A .= unveca(v[1:Adim])
        R .= reshape(v[Adim+1:end], k, p)
    end
    
    # compute ex1= exp((1-2*alf)*A) and populate
    # the (p+k) × (p+k) matrix mat in the Exp formula. ex2=exp(mat)
    @inline function setMatForExp!(mat, ex1, ex2, A, R)
        mul!(mat11, 2*alf, A)
        mul!(mat12, -1, R')
        copyto!(mat21, R)
    end
        
    function fun!(F, G, v)
        unvecAR!(A, R, v)
        ex1 .= exp((1-2*alf)*A)        
        setMatForExp!(mat, ex1, ex2, A, R)
        mul!(E11, ex1, WTY)
        mul!(E12, ex1, WTQ)        

        if isnothing(G)
            ex2 .= exp(mat)        
            return -sum(( WTY*M+WTQ*N)' .* ex1)
        end
        # ex2, fe2 = Manifolds.expm_frechet(mat, E)
        # expm affects buff, so affect ex2, fe2
        expm_frechet!(buff, mat, E)        
        
        mul!(WYMQN, WTY, M)
        mul!(WYMQN, WTQ, N, 1, 1)

        expm_frechet!(buffA, (1-2*alf)*A, WYMQN)
        dA .= asym((1-2*alf)*buffA[p+1:2*p, :]) .+ 2*alf*asym(fe2[1:p, 1:p])
        
        dR .= fe2[p+1:end, 1:p] - fe2[1:p, p+1:end]' 
        
        G[1:Adim] .= veca(dA)
        G[1+Adim:end] .= vec(dR)
        return -sum(WYMQN' .* ex1)
    end
    v0[1:Adim] = veca(A)
    v0[Adim+1:end] = vec(R)

    # run full gradient until pretol
    optzer = Opm.LBFGS(linesearch = Opm.LineSearches.MoreThuente(), m=5)
    ret = Opm.optimize(Opm.only_fg!(fun!), v0, optzer,
                   Opm.Options(g_tol=pretol,
                                 iterations=max_itr))
    unvecAR!(A, R, Opm.minimizer(ret))

    # run simple descent
    function Fnorm(dA, dR)
        return norm(dA, 2) + norm(dR, 2)
    end
    disc = Fnorm(dA, dR)
    for i in 1:max_itr
        ex1 .= exp((1-2*alf)*A)        
        setMatForExp!(mat, ex1, ex2, A, R)
        # this operation changes ex2, so change M, N        
        ex2 .= exp(mat)
        dA .= asym(WTY' - M*ex1)
        broadcast!(-, dR, WTQ', N*ex1)
        new_d = Fnorm(dA, dR)
        if new_d < disc
            A .+= dA
            R .+= dR
            disc = new_d
        else
            A .+= dA*disc/new_d
            R .+= dR*disc/new_d
        end
        if (maximum(abs.(extrema(dA))) < tolerance &&
            maximum(abs.(extrema(dR))) < tolerance)
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
    sym(mat)
    symmetrize mat
"""
@inline sym(mat) = 0.5*(mat + mat')

@doc raw"""
     asym(mat)
     anti-symmetrize mat
"""
@inline asym(mat) = 0.5*(mat - mat')

@doc raw"""
    _get_Q(Y, Y1)
    Find a complement basis Q in the linear span of Y Y1
    orthogonal to Y
"""
function _get_Q(Y, Y1)    
    n , p = size(Y)
    F = svd([Y Y1])
    k = sum(F.S .> _g_rank_cutoff)
    good = F.U[:, 1:k]*F.Vt[1:k, 1:k]
    qs = nullspace(Y'*good)
    QR = qr(good*qs)
    return QR.Q * vcat(I, zeros((n-k+p, k - p)))
end

@doc raw"""
    veca(mat)
    vectorize antisymmetric matrice mat
"""
function veca(mat)
    sz = size(mat)[1]
    ret = zeros(div((sz*(sz-1)), 2))
    start = 1
    for i in 1:sz-1
        ret[start:start+sz-i-1] .= mat[i+1:end, i]
        start += sz-i
    end
    return ret
end

@doc raw"""
    unveca(v)
    unravel v to an antisymmetric matrices
"""
function unveca(v)
    sz = .5 * (1 + sqrt(1 + 8 * size(v)[1]))
    sz = Int(round(sz))
    mat = zeros(sz, sz)
    start = 1
    for i in 1:(sz-1)
        mat[i+1:end, i] .= v[start:start+sz-i-1]
        mat[i, i+1:end] .= - v[start:start+sz-i-1]
        start += sz-i
    end
    return mat
end
