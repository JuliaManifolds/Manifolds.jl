@doc raw"""
    SubmersionManifold{F1, F2, F3} <: AbstractManifold{ℝ}

Model a manifold ``\mathcal M`` as a submersion and another manifold ``\mathcal N`` as the domain of the submersion, i.e.
we have a function $h: \mathcal N \to ℝ^k`` such that

```math
\mathcal M := \bigl\{ p \in \mathcal N : h(p) = 0_k \bigr\},
````

where ``0_k`` is the zero vector in ``ℝ^k``.
The function ``h`` is called a submersion if its differential is surjective at every point,
i.e. for every point ``p ∈ \mathcal N`` the linear map ``D h(p)`` is surjective.

And the tangent spaces for any point ``p ∈ \mathcal M`` are given by the kernel of the differential of ``h`` at ``p``, i.e.

```math
T_p \mathcal M := \bigl\{ X \in T_p \mathcal N : D h(p)[X] = 0 \bigr\}.
```

See also [Boumal:2023, Section 8.14](@cite), which is here simplified to a single submersion function ``h``.

## Fields
* `h::H`: The submersion function ``h`` with signature `(N, p) -> a`, where `N` is the `embedding`.
* `Dh::DH`: The differential ``Dh(p)[X]`` of the submersion function ``h`` with signature `(N, p, X) -> a`.
* `D²h::DDH`: The second differential of the submersion function ``h``.
* `embedding::M`: The manifold ``\mathcal N`` into which the submersion is defined.
"""
struct SubmersionManifold{M<:AbstractManifold{ℝ}, H, DH, DDH} <: AbstractManifold{ℝ}
    h::H
    Dh::DH
    D²h::DDH
    embedding::M
    function SubmersionManifold(h, Dh, embedding; D²h = nothing)
        return new{typeof(embedding), typeof(h), typeof(Dh), typeof(D²h)}(
            h, Dh, D²h, embedding,
        )
    end
end

"""
    check_point(M::SubmersionManifold, p; kwargs...)

Check if the point `p` lies on the submersion manifold `M`, i.e. fulfills ``h(p) = 0_k``.
"""
function check_point(M::SubmersionManifold, p; kwargs...)
    a = M.h(M.embedding, p)
    if !isapprox(norm(a), 0; kwargs...)
        return DomainError(
            a, "The point $(p) does not lie on $(M), since h(p) is not the zero vector.",
        )
    end
    return nothing
end


"""
    check_vector(M::SubmersionManifold, p, X; kwargs...)

Check if the vector `X` lies in the tangent space of the submersion manifold `M` at the point `p`,
i.e. fulfills ``Dh(p)[X] = 0_k``.
"""
function check_vector(M::SubmersionManifold, p, X; kwargs...)
    #TODO: Why just the very first entry? Dc(p)[X] should be the zero vector (the kernel of the differential is the tangent space)
    a = M.Dh(M.embedding, p, X)
    if !isapprox(norm(a), 0; kwargs...)
        return DomainError(
            a, "The vector $(X) does not lie in the tangent space at $(p) of $(M), since Dh(p)[X] is not the zero vector.",
        )
    end
    return nothing
end

"""
    get_embedding(M::SubmersionManifold)

Returns the embedding manifold ``\mathcal N`` of the submersion manifold ``\mathcal M``.
"""
function get_embedding(M::SubmersionManifold)
    return M.embedding
end

"""
    representation_size(M::SubmersionManifold)

Returns the size of the representation of points on the submersion manifold ``\mathcal M``,
which is the same as the size of the representation of points on the embedding manifold ``\mathcal N``.
"""
function representation_size(M::SubmersionManifold)
    return representation_size(M.embedding)
end

# TODO: Continue here

#TODO: Docs
function manifold_dimension(M::SubmersionManifold)
    return M.dim_domain - M.dim_codomain #TODO: When storing n we could also return n
end

#TODO: Docs – but we could also just set the right embedding mode as for the sphere?
function inner(M::SubmersionManifold, p, v, w)
    return dot(v, w)
end


#Hilsmethode: Erstellt die Matrix für das Sattelpunktprobelm für die nichtlineare Projection
#TODO: Docs – approach unclear, source unclear
function erstelle_Matrix_Sattel(M::SubmersionManifold, p)
    A = zeros(M.dim_domain + M.dim_codomain, M.dim_domain + M.dim_codomain)
    A = vcat(hcat(Matrix(I, M.dim_domain, M.dim_domain), transpose(M.c_prime(p))), hcat(M.c_prime(p), zeros(M.dim_codomain, M.dim_codomain)))
    return A
end

#Punkt aus R^n wird auf M projiziert (nichtlineare Projection)
#TODO: Docs – especially where it is from and that we should probably add keywords?
function project!(M::SubmersionManifold, q, p)
    p_curr = p
    c_val = M.c(p_curr)
    while LinearAlgebra.norm(c_val) > 1.0e-10
        R_i = erstelle_Matrix_Sattel(M, p_curr)
        br_i = vcat(zeros(M.dim_domain), c_val)
        sol_r = R_i \ -br_i
        delta_pi = sol_r[1:M.dim_domain]
        p_curr += delta_pi
        c_val = M.c(p_curr)
    end
    q .= p_curr
    return q
end

#lineare Projektion eines Vektors v \in R^n nach  T_p_M --> Vektortransport (RB: Wieso ist das ein VT?)
#TODO: Docs
function project!(M::SubmersionManifold, w, p, v)
    a = v - (transpose(M.c_prime(p)) * inv(M.c_prime(p) * transpose(M.c_prime(p))) * M.c_prime(p) * v)
    w .= vec(a)
    return w
end

#TODO: Docs
default_retraction_method(::SubmersionManifold) = ProjectionRetraction()

# TODO: Docs – also this would only work for the Euclidean case, when we do it more general we should probably have
# either a keyword to specify an inverse retraction in the embedding or so.
function inverse_retract_project!(M::SubmersionManifold, v, p, q)
    w = q .- p
    v .= project(M, p, w)
    return v
end

#TODO: Docs
function get_vectors(M::SubmersionManifold, p, ::DefaultOrthonormalBasis)
    N = nullspace(M.c_prime(p))
    return [N[:, j] for j in axes(N, 2)]
end

#TODO: Docs
# also: We could use the vectors from the previous function.
function get_vector_orthonormal!(M::SubmersionManifold, v, p, c, ::ManifoldsBase.RealNumbers)
    N = nullspace(M.c_prime(p))
    v .= 0.0
    for i in 1:length(c)
        v .+= c[i] .* N[:, i]
    end
    return v
end

# TODO: Docs
# also: why the atol and rtol? They are not used. We should also provide the case where `vector_at` is a point and produce a valid random tangent vector then.
function rand!(M::SubmersionManifold, x; vector_at = nothing, atol = 1.0e-8, rtol = 1.0e-8)
    x = vcat([1], zeros(M.dim_domain - 1))
    x = project(M, x)
    return x
end

# TODO: This should become ab ApproximateInverseRetraction and we should set the default_inverse_retraction_method to that.
function log!(M::SubmersionManifold, X, p, q)
    i = 0
    v_list = []
    x = []
    push!(v_list, q - p)
    while i < 10000
        v_list[end] = project(M, v_list[end], p)
        x = retract(M, p, v_list[end])
        w = q - x
        w_T_p_M = project(M, w, p)
        push!(v_list, v_list[end] + w_T_p_M)
        if LinearAlgebra.norm(w_T_p_M) < 1.0e-11
            break
        end
        i += 1
    end
    X .= v_list[end]
    return X
end

#TODO: Docs
# ... and adapt one we have the points above discussed
function Base.show(io::IO, M::SubmersionManifold)
    return print(io, "SubmersionManifold($(M.c), $(M.c_prime), $(M.c_prime_2), $(M.n), $(M.dim_domain), $(M.dim_codomain))")
end