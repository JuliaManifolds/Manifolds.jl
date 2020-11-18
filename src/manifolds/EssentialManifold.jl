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
#=
struct EssentialManifold <: Manifold{ℝ} 
    is_signed::Bool
    rotation::Rotations
end =#

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

    ##berechne t und q2
    t = 0
    fMin = Inf 
    q2_candidate = zeros(size(q))
    q2 = zeros(size(q))

    if !M.is_signed
        for k = 1:4 
            t_temp, f_temp, q2_candidate = distMinAnglePair(p, q, k)
            if f_temp < fMin
                fMin = f_temp
                t = t_temp
                q2 = q2_candidate
            end
        end
    else
        t, f, q2 = distMinAnglePair(p, q, 1) #ignore f 
    end
        
    #t = tMin[1] 
    Rz=[cos(t) -sin(t) 0; sin(t) cos(t) 0; 0 0 1]
    representative_q = zeros(size(q2))
    representative_q[1] = Rz * q2[1]
    representative_q[2] = Rz * q2[2]

    # use the logarithmic map in SO(3)^2 and the corresponding geodesic between p and representative_p to compute the solution 
    log!.(Ref(M.manifold), X, p, representative_q)
    return X
end

function distMinAnglePair(p, q, k)
    #flip q
    if k == 
        #do nothing
        break #insert something better 
    elseif k == 2
        q[1][2:3, :] = -q[1][2:3, :]
        q[2][[1 3],:] = -q[2][[1 3],:]
    elseif k == 3
        q[2][1:2, :] = -q[2][1:2, :]
    elseif k == 4
        q[1][2:3, :] = -q[1][2:3, :]
        q[2][2:3, :] = -q[2][2:3, :]
    else 
        return DomainError(
            k,
            "Value of $k is invalid.",
        )
    end 
    #compute rotations
    q211 = q[1]*p[1]'
    q212 = q[2]*p[2]'

    #compute values from Proposition ...
    t_break1, c1, m1, p1 = distMinAnglePair_discontinuityDistance(q211)
    t_break2, c2, m2, p2 = distMinAnglePair_discontinuityDistance(q212)

    #check if cost is constant
    tolMZero = 1e-15
    if abs(m1) < tolMZero && abs(m2) < tolMZero
        t = 0
        f = 2*pi^2
        tMinAll = 0
    else
        if abs(mod(t_break1 - t_break2 + pi, 2 * pi) - pi)
            t = t_break1 + pi
            f = 0
        else
            

    

    return t, f, q2_candidate
end

function distMinAnglePair_discontinuityDistance(q)
    c1 = q[1, 1] + q[2, 2]
    c2 = q[1, 2] - q[2, 1]
    c3 = q[3, 3]

    m = norm[c1 c2]
    p = atan(c1, c2) #order

    t_break = -0.5 * pi - p #why different than in paper? 
    return t_break, c3, m, p
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
    pq = (q.').*p
    # left translation
    copyto!(Y, pq.*X.*pq.')
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
