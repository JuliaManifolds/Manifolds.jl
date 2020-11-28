@doc raw"""
    EssentialManifold <: AbstractPowerManifold{ℝ}

The [`Manifold`](@ref) is the space of the essential matrices which is represented as 
the quotient space 
````math
\mathcal{M}_{\text{E}} := (\text{SO}(3)×\text{SO}(3))/(\text{SO}(2))\text{[^Tron2014]}.
````

[^Tron2014]:
    > Tron R.; Daniilidis K.; On the Quotient Representation for the Essential 
    > Manifold.
    > IEEE Conference on Computer Vision and Pattern Recognition (2014),
    > [pdf](https://ieeexplore.ieee.org/document/6909600).

# Constructor
    EssentialManifold(is_signed)

Generate the manifold 
"""

struct EssentialManifold <:
    AbstractPowerManifold{ℝ, Rotations{3}, NestedPowerRepresentation} 
    is_signed::Bool
    manifold::Rotations{3}
end

EssentialManifold(is_signed::Bool = true) = Essential(is_signed, Rotations(3))  

@doc raw"""
    check_manifold_point(M::EssentialManifold, p; kwargs...) 

Check whether the matrix is a valid point on the [`EssentialManifold`](@ref) `M`, 
i.e. is a Tuple of two SO(3) matrices.
"""
function check_manifold_point(M::EssentialManifold, p; kwargs...) 
    if length(p)!=2
        return DomainError(
            length(p),
            "The point $(p) does not lie on $M, since it does not contain exactly two elements.",
        )
    end
    return check_manifold_point(PowerManifold(M.manifold, 2), p; kwargs...)
    #Checke noch, ob signed
    return nothing
end

"""
    check_tangent_vector(M::EssentialManifold, p, X; check_base_point = true, kwargs... )

Check whether `X` is a tangent vector to manifold point `p` on the [`EssentialManifold`](@ref) `M`, 
i.e. `X` has to be a Tuple of two `3`-by-`3` skew-symmetric matrices. The optional parameter `check_base_point` 
indicates, whether to call
 [`check_manifold_point`](@ref)  for `p`.

The tolerance for the ... of `p` and `X` can be set using `kwargs...`.
"""
function check_tangent_vector(
    M::EssentialManifold,
    p,
    X;
    check_base_point = true,
    kwargs...,
)
    if check_base_point
        mpe = check_manifold_point(M, p; kwargs...)
        mpe === nothing || return mpe
    end
    if length(X)!=2
        return DomainError(
            length(X),
            "$(X) is not a tangent vector to the manifold $M, since it does not contain exactly two elements.",
        )
    end
    return check_tangent_vector(
        PowerManifold(M.manifold, 2),
        p,
        X;
        check_base_point = check_base_point,
        kwargs...,
    )
    return nothing
end

@doc raw"""
    exp(M::EssentialManifold, p, X)

Compute the exponential map on the [`EssentialManifold`](@ref) from `p` into direction
`X`, i.e.
````math
\text{exp}_p(X) =\text{exp}_g(̃Y),   g \in \text(SO)(3)^2,
````
where $Ỹ$ is the horizontal lift of $Y$.
"""#define Y
exp(::EssentialManifold, ::Any...)

function exp!(M::EssentialManifold, q, p, X)
    exp!.(Ref(M.manifold),q, p, X)
    q .= X.*q
    return q
end


get_iterator(M::EssentialManifold) = Base.OneTo(2)

@doc raw"""
    log(M::EssentialManifold, p, q)

Compute the logarithmic map on the [`EssentialManifold`](@ref) `M`, i.e. the tangent vector,
whose geodesic starting from `p` reaches `q` after time 1.
"""
function log!(M::EssentialManifold, X, p, q)
    # compute the closest representative of q
    t = 0
    f_min = Inf 
    q2 = deepcopy(q)

    if !M.is_signed
        for k = 1:4 
            #flip q
            if k == 2
                q2[1][2:3, :] = -q[1][2:3, :]
                q2[2][[1 3],:] = -q[2][[1 3],:]
            elseif k == 3
                q2[2][1:2, :] = -q[2][1:2, :]
            elseif k == 4
                q2[1][2:3, :] = -q[1][2:3, :]
                q2[2][2:3, :] = -q[2][2:3, :]
            end 
            t_temp, f_temp = dist_min_angle_pair(p, q2)
            if f_temp < f_min
                f_min = f_temp
                t = t_temp
            end
        end
    else
        t, _ = dist_min_angle_pair(p, q2) 
    end
        
    Rz=[cos(t) -sin(t) 0; sin(t) cos(t) 0; 0 0 1]
    representative_q = zeros(size(q2))
    representative_q[1] = Rz * q2[1]
    representative_q[2] = Rz * q2[2]

    # use the logarithmic map in SO(3)^2 and the corresponding geodesic between p and representative_p to compute the solution 
    log!.(Ref(M.manifold), X, p, representative_q)
    return X
end

function dist_min_angle_pair(p, q)
    #compute rotations
    q211 = q[1]*p[1]'
    q212 = q[2]*p[2]'

    #compute values from Proposition ... 
    t_break1, c1, m1, Φ1 = dist_min_angle_pair_discontinuity_distance(q211)
    t_break2, c2, m2, Φ2 = dist_min_angle_pair_discontinuity_distance(q212)
    
    #check if cost is constant
    tolMZero = 1e-15 
    if abs(m1) < tolMZero && abs(m2) < tolMZero
        t_min = 0
        f_min = 2*pi^2
    else
        if abs(mod(t_break1 - t_break2 + pi, 2 * pi) - pi)
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
            θ1_break2 = acos(clamp((m1 * sin(t_break2 + Φ1) + c1 - 1) / 2 , -1.0, 1.0))
            df1_break2= -θ1_break2 * (m1 * cos(t_break2 + Φ1)) / (2 * sin(θ1_break2))
            θ2_break1 = acos(clamp((m2 * sin(t_break1 + Φ2) + c2 - 1) / 2 , -1.0, 1.0))
            df2_break1= -θ2_break1 * (m2 * cos(t_break1 + Φ2)) / (2 * sin(θ2_break1))

            #compute left and right derivatives of sum of the two terms
            df_break1n = df1_break1 + df2_break1
            df_break1p = -df1_break1 + df2_break1
            df_break2n = df2_break2 + df1_break2
            df_break2p = -df2_break2 + df1_break2

            f_min = Inf
            if sign(df_break1p) != sign(df_break2n)
                #parabolic prediction of min
                t_min0 = t_search1 - df_break1p * (t_search2 - t_search1) / (df_break2n - df_break1p)
                #use Newton's search to find t_min
                t_min, f_min = dist_min_angle_pair_df_newton(m1, Φ1, c1, m2, Φ2, c2, t_min0, t_search1, t_search2)
            end
            t_search1 = t_search1 + 2 * pi
            if sign(df_break2p) != sign(df_break1n)
                #parabolic prediction of min
                t_min0 = t_search2 - df_break2p * (t_search1 - t_search2) / (df_break1n - df_break2p)
                #use Newton's search to find t_min
                t_min2, f_min2 = dist_min_angle_pair_df_newton(m1, Φ1, c1, m2, Φ2, c2, t_min0, t_search2, t_search1)
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

function dist_min_angle_pair_discontinuity_distance(q)
    c1 = q[1, 1] + q[2, 2]
    c2 = q[1, 2] - q[2, 1]
    c3 = q[3, 3]

    m = norm([c1 c2])
    Φ = atan(c1, c2) 

    t_break = -0.5 * pi - Φ
    return t_break, c3, m, Φ
end

function dist_min_angle_pair_compute_df_break(t_break, q) #compute derivatives of each term at discontinuity points
    c = cos(t_break)
    s = sin(t_break)

    q1_r_break_q1 = [c -s 0; s c 0; 0 0 1] * q
    F = svd(q1_r_break_q1 + I(3))    
    df_break = pi * abs(F.U[3, 1])
    return df_break
end

function dist_min_angle_pair_df_newton(m1, Φ1, c1, m2, Φ2, c2, t_min, t_low, t_high)
    tol_dist = 1e-8
    for i = 1:100
        #compute auxiliary values
        mc1 = m1 * cos(t_min + Φ1)
        mc2 = m2 * cos(t_min + Φ2)
        s1 = 2 * sin(θ1)
        s2 = 2 * sin(θ2)
        eztuSq1 = (mc1 / s1)^2
        eztuSq2 = (mc2 / s2)^2
        
        #compute θ_i
        θ1 = acos(clamp(((m1 * sin(t_min + Φ1) + c1 - 1) / 2), -1, 1))
        θ2 = acos(clamp(((m2 * sin(t_min + Φ2) + c2 - 1) / 2), -1, 1))

        #compute the first derivatives
        d1 = (-θ1 * mc1) / s1 
        d2 = (-θ2 * mc2) / s2
        d = d1 + d2
        
        #compute the second derivatives
        dd1 = eztuSq1 + θ1 / 2 * cot(θ1 / 2) * (1 - eztuSq1)
        dd2 = eztuSq2 + θ2 / 2 * cot(θ2 / 2) * (1 - eztuSq2)
        dd = dd1 + dd2

        #compute the new t_min
        t_old = t_min;
        t_min = clamp(t_old - d / dd, t_low + tol_dist, t_high - tol-dist)
        if abs(t_min - t_old) < tol_dist
            break
        end
    end
    f_min = θ1^2 + θ2^2
    return t_min, f_min
end


@doc raw"""
    manifold_dimension(M::EssentialManifold{is_signed, ℝ})

Return the manifold dimension of the [`EssentialManifold`](@ref), which is `5`.
"""
function manifold_dimension(::EssentialManifold) 
    return 5
end

@doc raw"""
    project(M::EssentialManifold, p, X)

Project the matrix `X` onto the tangent space
````math
T_{p}\text{SO}(3)^2 = T_{\text{vp}}\text{SO}(3)^2 ⊕ T_{\text{hp}}\text{SO}(3)^2,
````
by first computing its projection onto the vertical space $T_{\text{vp}}\text{SO}(3)^2$ using [`vert_proj`](@ref).
Then the orthogonal projection of `X` onto the horizontal space $T_{\text{hp}}\text{SO}(3)^2$ is defined as
````math
\Pi_h(X) = X - \frac{\text{vert/_proj}_p(X)}{2} \begin{bmatrix} R_1^T e_z \\ R_2^T e_z \end{bmatrix},
````
with $R_i = R_0 R^'_i, i=1,2,$ where $R^'_i$ is part of the pose of camera $i$ $g_i = (R^'_i,T^'_i) ∈ \text{SE}(3)$ 
and $R_0 ∈ \text{SO}(3)$ such that $R_0(T^'_2-T^'_1) = e_z$.
"""
project(::EssentialManifold, ::Any, ::Any)

function project!(M::EssentialManifold, Y, p, X) 
    X_proj = p'.*X
    X_proj_skew = 0.5 * (X_proj - X_proj')
    # projection on the vertical component
    s = vert_proj(M, p, X_proj_skew)
    # orthogonal projection
    copyto!(Y, X_proj_skew - (s/2) .* vcat(get_vector(M.manifold, p[1], p[1][3, :]), get_vector(M.manifold, p[2], p[2][3, :])))  
    return Y 
end

"""
    representation_size(M::EssentialManifold)

Return the array dimensions required to represent an element on the
[`EssentialManifold`](@ref) `M`, i.e. the vector of all array dimensions.
"""
@generated representation_size(::EssentialManifold) = ((3, 3), 2)

function Base.show(io::IO, M::EssentialManifold) 
    return print(io, "EssentialManifold($(M.is_signed))")
end

@doc raw"""
    vector_transport_to(M::EssentialManifold, p, X, q, method::ParallelTransport)

Compute the vector transport of the tangent vector `X` at `p` to `q` on the
[`EssentialManifold`](@ref) `M` using left translation of the ambient group. 
"""
vector_transport_to(::EssentialManifold, ::Any, ::Any, ::Any, ::ParallelTransport)

function vector_transport_to!(::EssentialManifold, Y, p, X, q, ::ParallelTransport)
    # group operation in the ambient group
    pq = (q').*p
    # left translation
    copyto!(Y, pq.*X.*pq')
    return Y
end

@doc raw"""
    vert_proj(M::EssentialManifold, p, X)

Project 'X' onto the vertical space $T_{\text{vp}}\text{SO}(3)^2$ with
````math
\text{vert/_proj}_p(X) = e_z^T(R_1 X_1 + R_2 X_2),
````
where $e_z$ is the third unit vector, $X_i ∈ T_{p}\text{SO}(3) for i=1,2,$ and it holds $R_i = R_0 R^'_i, i=1,2,$ where $R^'_i$ is part of the 
pose of camera $i$ $g_i = (R^'_i,T^'_i) ∈ \text{SE}(3)$ and $R_0 ∈ \text{SO}(3)$ such that $R_0(T^'_2-T^'_1) = e_z$.

"""
vert_proj(M::EssentialManifold, p, X) = vert_proj(M.manifold, p[1], X[1]) + vert_proj(M.manifold, p[2], X[2])
vert_proj(M::Rotations{3}, p, X) = (p*get_coordinates(M, p, X))[3]
