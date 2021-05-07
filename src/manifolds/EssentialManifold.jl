@doc raw"""
    EssentialManifold <: AbstractPowerManifold{ℝ}

The essential manifold is the space of the essential matrices which is represented as
a quotient space of the [`Rotations`](@ref) manifold product ``\mathrm{SO}(3)^2``.

Let ``R_x(θ), R_y(θ), R_x(θ) \in ℝ^{x\times 3}`` denote the rotation around the ``z``,
``y``, and ``x`` axis in ``ℝ^3``, respectively, and further the groups

```math
H_z = \bigl\{(R_z(θ),R_z(θ))\ \big|\ θ ∈ [-π,π) \bigr\}
```

and

```math
H_π = \bigl\{ (I,I), (R_x(π), R_x(π)), (I,R_z(π)), (R_x(π), R_y(π))  \bigr\}
```

acting elementwise on the left from ``\mathrm{SO}(3)^2`` (component wise).

Then the unsigned Essential manifold ``\mathcal{M}_{\text{E}}`` can be identified
with the quotient space

````math
\mathcal{M}_{\text{E}} := (\text{SO}(3)×\text{SO}(3))/(H_z × H_π),
````

and for the signed Essential manifold $\mathcal{M}_{\text{Ǝ}}$, the quotient reads

````math
\mathcal{M}_{\text{Ǝ}} := (\text{SO}(3)×\text{SO}(3))/(H_z).
````

An essential matrix is defined as

````math
E = (R'_1)^T [T'_2 - T'_1]_{×} R'_2,
````

where the poses of two cameras ``(R_i', T_i'), i=1,2``, are contained in the space of
rigid body transformations $SE(3)$ and the operator $[⋅]_{×}\colon ℝ^3 \to \operatorname{SkewSym}(3)$
denotes the matrix representation of the cross product operator. For more details see [^TronDaniilidis2017].

# Constructor
    EssentialManifold(is_signed=true)

Generate the manifold of essential matrices, either the signed (`is_signed=true`) or
unsigned (`is_signed=false`) variant.

[^TronDaniilidis2017]:
    > Tron R.; Daniilidis K.; The Space of Essential Matrices as a Riemannian Quotient
    > AbstractManifold.
    > SIAM Journal on Imaging Sciences (2017),
    > DOI: [10.1137/16M1091332](https://doi.org/10.1137/16M1091332),
    > PDF: [https://www.cis.upenn.edu/~kostas/mypub.dir/tron17siam.pdf](https://www.cis.upenn.edu/~kostas/mypub.dir/tron17siam.pdf).
"""
struct EssentialManifold <: AbstractPowerManifold{ℝ,Rotations{3},NestedPowerRepresentation}
    is_signed::Bool
    manifold::Rotations{3}
end

EssentialManifold(is_signed::Bool=true) = EssentialManifold(is_signed, Rotations(3))

@doc raw"""
    check_point(M::EssentialManifold, p; kwargs...)

Check whether the matrix is a valid point on the [`EssentialManifold`](@ref) `M`,
i.e. a 2-element array containing SO(3) matrices.
"""
function check_point(M::EssentialManifold, p; kwargs...)
    if length(p) != 2
        return DomainError(
            length(p),
            "The point $(p) does not lie on $M, since it does not contain exactly two elements.",
        )
    end
    return check_point(
        PowerManifold(M.manifold, NestedPowerRepresentation(), 2),
        p;
        kwargs...,
    )
end

"""
    check_tangent_vector(M::EssentialManifold, p, X; check_base_point = true, kwargs... )

Check whether `X` is a tangent vector to manifold point `p` on the [`EssentialManifold`](@ref) `M`,
i.e. `X` has to be a 2-element array of `3`-by-`3` skew-symmetric matrices.
The optional parameter `check_base_point` indicates, whether to call [`check_point`](@ref)  for `p`.
"""
function check_tangent_vector(M::EssentialManifold, p, X; check_base_point=true, kwargs...)
    if check_base_point
        mpe = check_point(M, p; kwargs...)
        mpe === nothing || return mpe
    end
    if length(X) != 2
        return DomainError(
            length(X),
            "$(X) is not a tangent vector to the manifold $M, since it does not contain exactly two elements.",
        )
    end
    return check_tangent_vector(
        PowerManifold(M.manifold, NestedPowerRepresentation(), 2),
        p,
        X;
        check_base_point=check_base_point,
        kwargs...,
    )
end

@doc raw"""
    distance(M::EssentialManifold, p, q)

Compute the Riemannian distance between the two points `p` and `q` on the [`EssentialManifold`](@ref). This is done by
computing the distance of the equivalence classes $[p]$ and $[q]$ of the points
$p=(R_{p_1},R_{p_2}), q=(R_{q_1},R_{q_2}) ∈ SO(3)^2$, respectively. Two points in $SO(3)^2$ are equivalent iff their
corresponding essential matrices, given by
````math
E = R_1^T [e_z]_{×}R_2,
````
are equal (up to a sign flip). Using the logarithmic map, the distance is given by
````math
\text{dist}([p],[q]) = \| \text{log}_{[p]} [q] \| = \| \log_p (S_z(t_{\text{opt}})q) \|,
````
where $S_z ∈ H_z = \{(R_z(θ),R_z(θ))\colon θ \in [-π,π) \}$ in which $R_z(θ)$ is the rotation around the z axis with angle
$θ$ and $t_{\text{opt}}$ is the minimizer of the cost function
````math
f(t) = f_1 + f_2, \quad f_i = \frac{1}{2} θ^2_i(t), \quad θ_i(t)=d(R_{p_i},R_z(t)R_{b_i}) \text{ for } i=1,2,
````
where $d(⋅,⋅)$ is the distance function in $SO(3)$[^TronDaniilidis2017].
"""
distance(M::EssentialManifold, p, q) = norm(M, p, log(M, p, q))

@doc raw"""
    exp(M::EssentialManifold, p, X)

Compute the exponential map on the [`EssentialManifold`](@ref) from `p` into direction
`X`, i.e.
````math
\text{exp}_p(X) =\text{exp}_g( \tilde X),  \quad g \in \text(SO)(3)^2,
````
where $\tilde X$ is the horizontal lift of $X$[^TronDaniilidis2017].
"""
exp(::EssentialManifold, ::Any...)

function exp!(M::EssentialManifold, q, p, X)
    exp!.(Ref(M.manifold), q, p, X)
    return q
end

get_iterator(::EssentialManifold) = Base.OneTo(2)

function isapprox(M::EssentialManifold, p, q; kwargs...)
    return isapprox(distance(M, p, q), 0.0; kwargs...)
end

@doc raw"""
    log(M::EssentialManifold, p, q)

Compute the logarithmic map on the [`EssentialManifold`](@ref) `M`, i.e. the tangent vector,
whose geodesic starting from `p` reaches `q` after time 1. Here, $p=(R_{p_1},R_{p_2})$ and
$q=(R_{q_1},R_{q_2})$ are elements of $SO(3)^2$. We use that any essential matrix can, up to
scale, be decomposed to
````math
E = R_1^T [e_z]_{×}R_2,
````
where $(R_1,R_2)∈SO(3)^2$. Two points in $SO(3)^2$ are equivalent iff their corresponding
essential matrices are equal (up to a sign flip).
To compute the logarithm, we first move `q` to another representative of its equivalence class.
For this, we find $t= t_{\text{opt}}$ for which the function
````math
f(t) = f_1 + f_2, \quad f_i = \frac{1}{2} θ^2_i(t), \quad θ_i(t)=d(R_{p_i},R_z(t)R_{b_i}) \text{ for } i=1,2,
````
where $d(⋅,⋅)$ is the distance function in $SO(3)$, is minimized. Further, the group $H_z$ acting
on the left on $SO(3)^2$ is defined as
````math
H_z = \{(R_z(θ),R_z(θ))\colon θ \in [-π,π) \},
````
where $R_z(θ)$ is the rotation around the z axis with angle $θ$. Points in $H_z$ are denoted by
$S_z$. Then, the logarithm is defined
as
````math
\log_p (S_z(t_{\text{opt}})q) = [\text{Log}(R_{p_i}^T R_z(t_{\text{opt}})R_{b_i})]_{i=1,2},
````
where $\text{Log}$ is the [`logarithm`](@ref log(::Rotations, ::Any...)) on $SO(3)$. For more
details see [^TronDaniilidis2017].
"""
log(M::EssentialManifold, ::Any, ::Any)

function log!(M::EssentialManifold, X, p, q)
    # compute the closest representative of q
    t = 0
    f_min = Inf
    q2 = deepcopy(q)

    if !M.is_signed
        q2min = deepcopy(q)
        for k in 1:4
            #flip sign in q to get another member of its equivalence class
            if k == 2
                q2[1][2:3, :] = -q[1][2:3, :]
                q2[2][[1 3], :] = -q[2][[1 3], :]
            elseif k == 3
                q2[2][1:2, :] = -q[2][1:2, :]
            elseif k == 4
                q2[1][2:3, :] = -q[1][2:3, :]
                q2[2][2:3, :] = -q[2][2:3, :]
            end
            t_temp, f_temp = dist_min_angle_pair(p, q2)
            if f_temp < f_min
                q2min = deepcopy(q2)
                f_min = f_temp
                t = t_temp
            end
        end
        q2 = q2min
    else
        t, _ = dist_min_angle_pair(p, q2)
    end
    Rz = [cos(t) -sin(t) 0; sin(t) cos(t) 0; 0 0 1]
    representative_q = zeros(size(q2))
    representative_q = [Rz * q2[1], Rz * q2[2]]

    # use the logarithmic map in SO(3)^2 and the corresponding geodesic between p and representative_q to compute the solution
    log!.(Ref(M.manifold), X, p, representative_q)
    return X
end

@doc raw"""
    dist_min_angle_pair(p, q)

This function computes the global minimizer of the function
````math
f(t) = f_1 + f_2, \quad f_i = \frac{1}{2} θ^2_i(t), \quad θ_i(t)=d(R_{p_i},R_z(t)R_{b_i}) \text{ for } i=1,2,
````
for the given values. This is done by finding the discontinuity points $t_{d_i}, i=1,2$ of its derivative
and using Newton's method to minimize the function over the intervals $[t_{d_1},t_{d_2}]$ and $[t_{d_2},t_{d_1}+2π]$
separately. Then, the minimizer for which $f$ is minimal is chosen and given back together with the minimal value.
For more details see Algorithm 1 in [^TronDaniilidis2017].
"""
function dist_min_angle_pair(p, q)
    #compute rotations
    q211 = q[1] * p[1]'
    q212 = q[2] * p[2]'

    #compute values from Proposition 9
    t_break1, c1, m1, Φ1 = dist_min_angle_pair_discontinuity_distance(q211)
    t_break2, c2, m2, Φ2 = dist_min_angle_pair_discontinuity_distance(q212)

    #check if cost is constant
    tol_m_zero = 5 * eps(eltype(q211)) # i.e. promoted from p and q
    tol_break = sqrt(eps(eltype(q211)))
    if (abs(m1) < tol_m_zero) && (abs(m2) < tol_m_zero)
        t_min = 0
        f_min = 2 * pi^2
    else
        if abs(mod(t_break1 - t_break2 + pi, 2 * pi) - pi) < tol_break
            t_min = t_break1 + pi
            f_min = 0
        else
            t_search1 = t_break1
            t_search2 = t_break2
            if t_search1 > t_search2
                t_search1 = t_search1 - 2 * pi
            end

            #compute derivatives of each term at discontinuity points
            df1_break1 = dist_min_angle_pair_compute_df_break(t_break1, q211)
            df2_break2 = dist_min_angle_pair_compute_df_break(t_break2, q212)

            #compute derivative of each term at other's discontinuity
            θ1_break2 = acos(clamp((m1 * sin(t_break2 + Φ1) + c1 - 1) / 2, -1.0, 1.0))
            df1_break2 = -θ1_break2 * (m1 * cos(t_break2 + Φ1)) / (2 * sin(θ1_break2))
            θ2_break1 = acos(clamp((m2 * sin(t_break1 + Φ2) + c2 - 1) / 2, -1.0, 1.0))
            df2_break1 = -θ2_break1 * (m2 * cos(t_break1 + Φ2)) / (2 * sin(θ2_break1))

            #compute left and right derivatives of sum of the two terms
            df_break1n = df1_break1 + df2_break1
            df_break1p = -df1_break1 + df2_break1
            df_break2n = df2_break2 + df1_break2
            df_break2p = -df2_break2 + df1_break2

            f_min = Inf
            if sign(df_break1p) != sign(df_break2n)
                #parabolic prediction of min
                t_min0 =
                    t_search1 -
                    df_break1p * (t_search2 - t_search1) / (df_break2n - df_break1p)
                #use Newton's search to find t_min
                t_min, f_min = dist_min_angle_pair_df_newton(
                    m1,
                    Φ1,
                    c1,
                    m2,
                    Φ2,
                    c2,
                    t_min0,
                    t_search1,
                    t_search2,
                )
            end
            t_search1 = t_search1 + 2 * pi
            if sign(df_break2p) != sign(df_break1n)
                #parabolic prediction of min
                t_min0 =
                    t_search2 -
                    df_break2p * (t_search1 - t_search2) / (df_break1n - df_break2p)
                #use Newton's search to find t_min
                t_min2, f_min2 = dist_min_angle_pair_df_newton(
                    m1,
                    Φ1,
                    c1,
                    m2,
                    Φ2,
                    c2,
                    t_min0,
                    t_search2,
                    t_search1,
                )
                #choose t with lower function values
                if f_min2 < f_min
                    t_min = t_min2
                    f_min = f_min2
                end
            end
        end
    end
    return t_min, f_min
end

@doc raw"""
    dist_min_angle_pair_discontinuity_distance(q)

This function computes the point $t_{\text{di}}$ for which the first derivative of
````math
f(t) = f_1 + f_2, \quad f_i = \frac{1}{2} θ^2_i(t), \quad θ_i(t)=d(R_{p_i},R_z(t)R_{b_i}) \text{ for } i=1,2,
````
does not exist. This is the case for $\sin(θ_i(t_{\text{di}})) = 0$. For more details see Proposition 9
and its proof, as well as Lemma 1 in [^TronDaniilidis2017].
"""
function dist_min_angle_pair_discontinuity_distance(q)
    c1 = q[1, 1] + q[2, 2]
    c2 = q[1, 2] - q[2, 1]
    c3 = q[3, 3]

    m = norm([c1 c2])
    Φ = atan(c1, c2)

    t_break = -0.5 * pi - Φ
    return t_break, c3, m, Φ
end

@doc raw"""
    dist_min_angle_pair_compute_df_break(t_break, q)

This function computes the derivatives of each term $f_i, i=1,2,$ at discontinuity point `t_break`. For more details see [^TronDaniilidis2017].
"""
function dist_min_angle_pair_compute_df_break(t_break, q)
    c = cos(t_break)
    s = sin(t_break)

    q1_r_break_q1 = [c -s 0; s c 0; 0 0 1] * q
    F = svd(q1_r_break_q1 + I(3))
    df_break = pi * abs(F.U[3, 1])
    return df_break
end

@doc raw"""
    dist_min_angle_pair_df_newton(m1, Φ1, c1, m2, Φ2, c2, t_min, t_low, t_high)

This function computes the minimizer of the function
````math
f(t) = f_1 + f_2, \quad f_i = \frac{1}{2} θ^2_i(t), \quad θ_i(t)=d(R_{p_i},R_z(t)R_{b_i}) \text{ for } i=1,2,
````
in the interval $[$`t_low`, `t_high`$]$ using Newton's method. For more details see [^TronDaniilidis2017].
"""
function dist_min_angle_pair_df_newton(m1, Φ1, c1, m2, Φ2, c2, t_min, t_low, t_high)
    tol_dist = sqrt(eps(eltype(t_min)))
    θ1 = zero(t_min)
    θ2 = zero(t_min)
    for i in 1:100
        #compute auxiliary values
        mc1 = m1 * cos(t_min + Φ1)
        mc2 = m2 * cos(t_min + Φ2)

        #compute θi, i=1,2
        θ1 = acos(clamp(((m1 * sin(t_min + Φ1) + c1 - 1) / 2), -1, 1))
        θ2 = acos(clamp(((m2 * sin(t_min + Φ2) + c2 - 1) / 2), -1, 1))

        #compute the first derivatives di, i=1,2
        s1 = 2 * sin(θ1)
        s2 = 2 * sin(θ2)
        d1 = (-θ1 * mc1) / s1
        d2 = (-θ2 * mc2) / s2
        d = d1 + d2

        #compute the second derivatives ddi, i=1,2
        eztuSq1 = (mc1 / s1)^2
        eztuSq2 = (mc2 / s2)^2
        dd1 = eztuSq1 + θ1 / 2 * cot(θ1 / 2) * (1 - eztuSq1)
        dd2 = eztuSq2 + θ2 / 2 * cot(θ2 / 2) * (1 - eztuSq2)
        dd = dd1 + dd2

        #compute the new t_min
        t_old = t_min
        t_min = clamp(t_old - d / dd, t_low + tol_dist, t_high - tol_dist)
        if abs(t_min - t_old) < tol_dist
            break
        end
    end
    f_min = θ1^2 + θ2^2
    return t_min, f_min
end

@doc raw"""
    manifold_dimension(M::EssentialManifold{is_signed, ℝ})

Return the manifold dimension of the [`EssentialManifold`](@ref), which is `5`[^TronDaniilidis2017].
"""
function manifold_dimension(::EssentialManifold)
    return 5
end

@doc raw"""
    project(M::EssentialManifold, p, X)

Project the matrix `X` onto the tangent space
````math
T_{p} \text{SO}(3)^2 = T_{\text{vp}}\text{SO}(3)^2 ⊕ T_{\text{hp}}\text{SO}(3)^2,
````
by first computing its projection onto the vertical space $T_{\text{vp}}\text{SO}(3)^2$ using [`vert_proj`](@ref).
Then the orthogonal projection of `X` onto the horizontal space $T_{\text{hp}}\text{SO}(3)^2$ is defined as
````math
\Pi_h(X) = X - \frac{\text{vert\_proj}_p(X)}{2} \begin{bmatrix} R_1^T e_z \\ R_2^T e_z \end{bmatrix},
````
with $R_i = R_0 R'_i, i=1,2,$ where $R'_i$ is part of the pose of camera $i$ $g_i = (R'_i,T'_i) ∈ \text{SE}(3)$
and $R_0 ∈ \text{SO}(3)$ such that $R_0(T'_2-T'_1) = e_z$.
"""
project(::EssentialManifold, ::Any, ::Any)

function project!(M::EssentialManifold, Y, p, X)
    X_proj = [pe' * Xe for (pe, Xe) in zip(p, X)]
    X_proj .= [0.5 * (Xe - Xe') for Xe in X_proj]
    # projection on the vertical component
    s = vert_proj(M, p, X_proj)
    # orthogonal projection
    copyto!(
        Y,
        [
            Xe - (s / 2) * get_vector(M.manifold, pe, pe[3, :], DefaultOrthogonalBasis())
            for (Xe, pe) in zip(X_proj, p)
        ],
    )
    return Y
end

function Base.show(io::IO, M::EssentialManifold)
    return print(io, "EssentialManifold($(M.is_signed))")
end

function vector_transport_direction(M::EssentialManifold, p, X, d)
    return vector_transport_direction(M, p, X, d, ParallelTransport())
end

function vector_transport_direction!(M::EssentialManifold, Y, p, X, d)
    return vector_transport_direction!(M, Y, p, X, d, ParallelTransport())
end

function vector_transport_direction!(M::EssentialManifold, Y, p, X, d, m::ParallelTransport)
    y = exp(M, p, d)
    return vector_transport_to!(M, Y, p, X, y, m)
end

@doc raw"""
    vector_transport_to(M::EssentialManifold, p, X, q, method::ParallelTransport)

Compute the vector transport of the tangent vector `X` at `p` to `q` on the
[`EssentialManifold`](@ref) `M` using left translation of the ambient group.
"""
vector_transport_to(::EssentialManifold, ::Any, ::Any, ::Any, ::ParallelTransport)

function vector_transport_to(M::EssentialManifold, p, X, q)
    return vector_transport_to(M, p, X, q, ParallelTransport())
end

function vector_transport_to!(M::EssentialManifold, Y, p, X, q)
    return vector_transport_to!(M, Y, p, X, q, ParallelTransport())
end

function vector_transport_to!(::EssentialManifold, Y, p, X, q, ::ParallelTransport)
    # group operation in the ambient group
    pq = [qe' * pe for (pe, qe) in zip(p, q)]
    # left translation
    copyto!(Y, [pqe * Xe * pqe' for (pqe, Xe) in zip(pq, X)])
    return Y
end

@doc raw"""
    vert_proj(M::EssentialManifold, p, X)

Project `X` onto the vertical space $T_{\text{vp}}\text{SO}(3)^2$ with
````math
\text{vert\_proj}_p(X) = e_z^T(R_1 X_1 + R_2 X_2),
````
where $e_z$ is the third unit vector, $X_i ∈ T_{p}\text{SO}(3)$ for $i=1,2,$ and it holds $R_i = R_0 R'_i, i=1,2,$ where $R'_i$ is part of the
pose of camera $i$ $g_i = (R_i,T'_i) ∈ \text{SE}(3)$ and $R_0 ∈ \text{SO}(3)$ such that $R_0(T'_2-T'_1) = e_z$ [^TronDaniilidis2017].

"""
function vert_proj(M::EssentialManifold, p, X)
    return sum(vert_proj.(Ref(M.manifold), p, X))
end
function vert_proj(M::Rotations{3}, p, X)
    return (p[3, :]' * get_coordinates(M, p, X, DefaultOrthogonalBasis()))
end
