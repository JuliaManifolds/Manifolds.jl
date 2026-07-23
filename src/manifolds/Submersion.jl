
# doku
struct SubmersionManifold{F1, F2, F3} <: AbstractManifold{ℝ} 
	   	c::F1
		c_prime::F2
		c_prime_2::F3
		n::Int
		dim_domain::Int
		dim_codomain::Int 
end

# n ist dim(M) = (dim_domain - dim_codomain). Muss eigentlich nicht übergeben werden, aber der Konstruktor funktioniert in der Datei, in welcher diese Included wird, nur mit n (auch nach mehrmaligem Neustart)


function check_point(M::Submersion, p; kwargs...)
    if !isapprox(M.c(p), 0; kwargs...)
        return DomainError(M.c(p),
            "Punkt liegt nicht auf der Untermannigfaltigkeit")
    end
	return nothing
end

function check_vector(M::Submersion, p, v; kwargs...)
	is_point(M,p; kwargs...)
	if !isapprox((M.c_prime(p)*v)[1],0; kwargs...)
		return DomainError(M.c(p),
            "Vektor liegt nicht in T_p_M")
    end
		return nothing
end

function get_embedding(M::Submersion)
	return Euclidean(manifold_dimension(M); field = ℝ)
end

function representation_size(M::Submersion)
	return (M.dim_domain,)
end
	
function manifold_dimension(M::Submersion)
	return M.dim_domain - M.dim_codomain
end


function inner(M::Submersion,p,v,w)
    return dot(v, w)
end

#Hilsmethode: Erstellt die Matrix für das Sattelpunktprobelm für die nichtlineare Projektion
function erstelle_Matrix_Sattel(M::Submersion,p)
			A = zeros(M.dim_domain + M.dim_codomain,M.dim_domain + M.dim_codomain)
			A = vcat(hcat(Matrix(I,M.dim_domain,M.dim_domain),transpose(M.c_prime(p))),hcat(M.c_prime(p),zeros(M.dim_codomain,M.dim_codomain)))
		return A
	end

#Punkt aus R^n wird auf M projiziert (nichtlineare Projektion)
function project!(M::Submersion, q, p)
    p_curr = p
    c_val = M.c(p_curr)
    while LinearAlgebra.norm(c_val) > 1e-10
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

function project(M::Submersion, p)
    q = copy(p)
    return project!(M, q, p)
end

#lineare Projektion eines Vektors v \in R^n nach  T_p_M --> Vektortransport
function project!(M::Submersion, w, p, v)
    a = v -(transpose(M.c_prime(p))*inv(M.c_prime(p)*transpose(M.c_prime(p)))*M.c_prime(p)*v)
    w .= vec(a)
    return w
end

function project(M::Submersion, p, v)
    w = copy(v)
    return project!(M, w, p, v)
end

# Setze die Retraktion default auf pr(M,p+x)
default_retraction_method(::Submersion) = ProjectionRetraction()

function retract_project!(M::Submersion,q,p,v)
    q .= project(M, p .+ v)
    return q
end

function retract_project(M::Submersion,p,v;method::AbstractRetractionMethod=default_retraction_method(M, typeof(p)),kwargs...)
    q = copy(M, p)
    return retract_project!(M,q,p,v)
end

function inverse_retract_project!(M::Submersion,v,p,q)
    w = q .- p
    v .= project(M, p, w)
    return v
end

#Methoden für Basis
	

function get_basis(M::Submersion,p,B::DefaultOrthonormalBasis{ℝ, 		TangentSpaceType};kwargs...,)
    return CachedBasis(B, get_vectors(M, p, B))
end

	
#Berechne Basis von Ker(c'(p)) = T_p_M durch nullspace.
function get_vectors(M::Submersion,p,B::DefaultOrthonormalBasis{ℝ,TangentSpaceType})
    is_point(M,p; kwargs...)
	N = nullspace(M.c_prime(p))
    return [N[:,j] for j in axes(N,2)]
end

#berechnet nur die Linearkombination eines Koordinatenvektors bzgl. einer Basis von T_p_M
function get_vector_orthonormal!(M::Submersion, v, p, c,::ManifoldsBase.RealNumbers)
	N = nullspace(M.c_prime(p)) 
	v .= 0.0
	for i in 1:length(c)
	    v .+= c[i] .* N[:, i]
	end
	return v
end

#Macht genau dasselbe wie get_vector_orthonormal!
function get_vector(M::Submersion, p, c, B::DefaultOrthonormalBasis)
    basis_vec = get_vectors(M, p, B)  
    n = length(T)
    v = [zeros(3) for i in 1:length(p)]
	for i in 1:n
        v .+= c[i] .* basis_vec[i]
    end
    return v
end
	

#Methoden die nicht im Newton verwendet werden, jedoch implementiert werden müssen
#(log! ist nur ein Versuch)

function rand!(M::Submersion,x;vector_at=nothing,atol=1e-8,rtol=1e-8)
	x = vcat([1],zeros(M.dim_domain - 1))
    x =  project(M, x)
    return x
end
	
function log!(M::Submersion,X,p,q)
	i = 0
    v_list = []
	x = []
	push!(v_list,q-p)
    while i < 10000
        v_list[end] = project(M,v_list[end],p)
        x = retract(M,p,v_list[end])
        w = q - x
        w_T_p_M = project(M,w,p)
        push!(v_list,v_list[end] + w_T_p_M)
		if LinearAlgebra.norm(w_T_p_M) < 1e-11
			break
		end
        i += 1
    end
	X .= v_list[end]
    return X
end

# brauchen wir das wirklich?
function exp!(M::Submersion,q,p,X)
	q.= project(M,p,X)
	return q
end
