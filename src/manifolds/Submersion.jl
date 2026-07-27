# TODO: doku
struct SubmersionManifold{F1, F2, F3} <: AbstractManifold{ℝ}
    c::F1 #TODO: Following Boumal I would probably call it h?
    c_prime::F2 # TODO: Domain unclear: A differential? Dc(p) is a matrix? or is Dc(p)[X] a linear operator?
    c_prime_2::F3 # TODO: and how is this meant to be implemented? It is also not used at all?
    n::Int
    dim_domain::Int
    dim_codomain::Int
end

# n ist dim(M) = (dim_domain - dim_codomain). Muss eigentlich nicht übergeben werden, aber der Konstruktor funktioniert in der Datei, in welcher diese Included wird, nur mit n (auch nach mehrmaligem Neustart)
#TODO: Without seeing that script, I can not comment in this.

#TODO: Docs
function check_point(M::SubmersionManifold, p; kwargs...)
    if !isapprox(M.c(p), 0; kwargs...)
        return DomainError(
            M.c(p),
            "The point $(p) does not lie on $(M), since c(p) is not 0.",
        )
    end
    return nothing
end

#TODO: Docs
function check_vector(M::SubmersionManifold, p, v; kwargs...)
    #TODO: Why just the very first entry? Dc(p)[X] should be the zero vector (the kernel of the differential is the tangent space)
    if !isapprox((M.c_prime(p) * v)[1], 0; kwargs...)
        return DomainError(
            M.c(p), # TODO: This is the wrong term?
            "The vector $(v) does not lie in the tangent space at $(p) of $(M), since Dc(p)[v](p) is not 0.",
        )
    end
    return nothing
end

#TODO: Docs
function get_embedding(M::SubmersionManifold)
    #TODO: One could actually submerse into _any_ manifold if we wanted to.
    #...we would just have to store the manifold in the type (and would no longer need dim_codomain?)
    return Euclidean(manifold_dimension(M); field = ℝ)
end

#TODO: Docs
function representation_size(M::SubmersionManifold)
    # TODO: Not sure why this would be the case? We would represent them in the manifold we submerse into.
    return (M.dim_domain,)
end

#TODO: Docs
function manifold_dimension(M::SubmersionManifold)
    return M.dim_domain - M.dim_codomain #TODO: When storing n we could also return n
end

#TODO: Docs – but we could also just set the right embedding mode as for the sphere?
function inner(M::SubmersionManifold, p, v, w)
    return dot(v, w)
end


#Hilsmethode: Erstellt die Matrix für das Sattelpunktprobelm für die nichtlineare Projektion
#TODO: Docs – approach unclear, source unclear
function erstelle_Matrix_Sattel(M::SubmersionManifold, p)
    A = zeros(M.dim_domain + M.dim_codomain, M.dim_domain + M.dim_codomain)
    A = vcat(hcat(Matrix(I, M.dim_domain, M.dim_domain), transpose(M.c_prime(p))), hcat(M.c_prime(p), zeros(M.dim_codomain, M.dim_codomain)))
    return A
end

#Punkt aus R^n wird auf M projiziert (nichtlineare Projektion)
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
# either a kwyord to specify an inverse retraction in the embedding or so.
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