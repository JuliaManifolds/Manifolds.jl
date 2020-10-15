@doc raw"""
    EssentialManifold <: Manifold{ℝ}

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
struct EssentialManifold <: Manifold{ℝ} 
    is_signed::Bool
    rotation::Rotations
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
    for i in 1:2    
        mpv = check_manifold_point(M.rotation, p[i]; kwargs...)
        mpv === nothing || return mpv
    end
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
    for i in 1:2    
        mpv = check_tangent_vector(M.rotation, p[i], X[i]; kwargs...)
        mpv === nothing || return mpv
    end
    return nothing
end

"""
    distance(M::EssentialManifold, p, q)

Compute the Riemannian distance between two points on the [`EssentialManifold`](@ref)
manifold `M`.
"""
distance(::EssentialManifold, p, q) =

@doc raw"""
    manifold_dimension(M::EssentialManifold{is_signed, ℝ})

Return the manifold dimension of the [`EssentialManifold`](@ref) over the number system `ℝ`.
"""
function manifold_dimension(::EssentialManifold) 
    return 5
end