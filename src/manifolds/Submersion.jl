### A Pluto.jl notebook ###
# v0.20.25

using Markdown
using InteractiveUtils
using Manopt
using Manifolds
using ManifoldsBase
using LinearAlgebra 


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


import ManifoldsBase: check_point
	function check_point(M::Submersion, p; kwargs...)
	    if !isapprox(M.c(p), 0; kwargs...)
	        return DomainError(M.c(p),
	            "Punkt liegt nicht auf der Untermannigfaltigkeit")
	    end
		return nothing
	end

import ManifoldsBase: check_vector
	function check_vector(M::Submersion, p, v; kwargs...)
		is_point(M,p; kwargs...)
		if !isapprox((M.c_prime(p)*v)[1],0; kwargs...)
			return DomainError(M.c(p),
	            "Vektor liegt nicht in T_p_M")
	    end
			return nothing
	end

import ManifoldsBase: get_embedding
	function get_embedding(M::Submersion)
		return Euclidean(manifold_dimension(M); field = ℝ)
	end


import ManifoldsBase: representation_size
	function representation_size(M::Submersion)
		return (M.dim_domain,)
	end

	
import ManifoldsBase:manifold_dimension
	function manifold_dimension(M::Submersion)
		return M.dim_domain - M.dim_codomain
	end

import ManifoldsBase:inner
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
import ManifoldsBase: project!
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

import ManifoldsBase:project
function project(M::Submersion, p)
    q = copy(p)
    return project!(M, q, p)
end

#lineare Projektion eines Vektors v \in R^n nach  T_p_M --> Vektortransport
import ManifoldsBase: project!
function project!(M::Submersion, w, p, v)
    a = v -(transpose(M.c_prime(p))*inv(M.c_prime(p)*transpose(M.c_prime(p)))*M.c_prime(p)*v)
    w .= vec(a)
    return w
end

import ManifoldsBase: project
function project(M::Submersion, p, v)
    w = copy(v)
    return project!(M, w, p, v)
end

# Setze die Retraktion default auf pr(M,p+x)
default_retraction_method(::Submersion) = ProjectionRetraction()

import ManifoldsBase:retract_project!
function retract_project!(M::Submersion,q,p,v)
    q .= project(M, p .+ v)
    return q
end

import ManifoldsBase:retract_project
function retract_project(M::Submersion,p,v;method::AbstractRetractionMethod=default_retraction_method(M, typeof(p)),kwargs...)
    q = copy(M, p)
    return retract_project!(M,q,p,v)
end

import ManifoldsBase: inverse_retract_project!
function inverse_retract_project!(M::Submersion,v,p,q)
    w = q .- p
    v .= project(M, p, w)
    return v
end

#Methoden für Basis
	
import ManifoldsBase:get_basis
	function get_basis(M::Submersion,p,B::DefaultOrthonormalBasis{ℝ, 		TangentSpaceType};kwargs...,)
	    return CachedBasis(B, get_vectors(M, p, B))
	end

	
#Berechne Basis von Ker(c'(p)) = T_p_M durch nullspace.
import ManifoldsBase:get_vectors
function get_vectors(M::Submersion,p,B::DefaultOrthonormalBasis{ℝ,TangentSpaceType})
    is_point(M,p; kwargs...)
	N = nullspace(M.c_prime(p))
    return [N[:,j] for j in axes(N,2)]
end

#berechnet nur die Linearkombination eines Koordinatenvektors bzgl. einer Basis von T_p_M
import ManifoldsBase: get_vector_orthonormal!
function get_vector_orthonormal!(M::Submersion, v, p, c,::ManifoldsBase.RealNumbers)
	N = nullspace(M.c_prime(p)) 
	v .= 0.0
	for i in 1:length(c)
	    v .+= c[i] .* N[:, i]
	end
	return v
end

#Macht genau dasselbe wie get_vector_orthonormal!
import ManifoldsBase:get_vector
function get_vector(M::Submersion, p, c, B::DefaultOrthonormalBasis)
    basis_vec = get_vectors(M, p, B)  
    n = length(T)
    v = [zeros(3) for i in 1:length(p)]
	for i in 1:n
        v .+= c[i] .* basis_vec[i]
    end
    return v
end

# Hier wurde die DefaultStepsize gesetzt, es wurde aber nicht für die Schrittweite übernommen --> musste direkt in Newton gesetzt werden
# siehe fehler 
import Manopt:DefaultStepsize
function DefaultStepsize(M::Submersion)
	 return ConstantLength(1)
end

	

#Methoden die nicht im Newton verwendet werden, jedoch implementiert werden müssen
#(log! ist nur ein Versuch)
import ManifoldsBase: rand!
	function rand!(M::Submersion,x;vector_at=nothing,atol=1e-8,rtol=1e-8)
		x = vcat([1],zeros(M.dim_domain - 1))
	    x =  project(M, x)
	    return x
	end
	

import ManifoldsBase:log!
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

import ManifoldsBase:exp!
	function exp!(M::Submersion,q,p,X)
		q.= project(M,p,X)
		return q
	end

# Hier wird nicht der echte Wert übergeben sondern 1, da die Schrittweite im Newton
# hiervon abhängt 
import ManifoldsBase:injectivity_radius 
function injectivity_radius(M::Submersion)
	return 1
end

	
#Methoden die nicht von ManifoldsBase erben und spezifisch für dieses Prolem benötigt werden


#Berechnen von Startkurven 
function anfangskurve_Retraktion(M::Submersion,x_start,x_end,n)
	t_list = range(0,1,n)
	strecke = []
	gamma = []
	push!(gamma,x_start)
	for t in t_list
		push!(strecke,(1-t)*x_start + t*x_end)
	end
	for i in range(1,length(t_list))
		push!(gamma,project(M,strecke[i]))
	end
	gamma = Vector{Vector{Float64}}(gamma)
	return gamma
end

#Hilfsmethode für anfangskurve_vektortransport_update
	function cut_anfangskurve(M, gamma, q)
   		dist = abstand_kurve_punkt(M, gamma, q)[2]
    	min_first = argmin(dist)
		gamma = gamma[1:min_first]
		push!(gamma,q)
    	return gamma
	end
	
#Erstellt Anfangskurve Mittels Retraktion schlauer Richtungen
	function anfangskurve_vektortransport_update(M::Submersion,p,q,h,max_iter = 1000)
		tol = 0.01
	    gamma = Vector{Vector{Float64}}()
		d = []
	    push!(gamma, p)
	    v = q - p
	    i = 1
	    while i < max_iter
		#Dieser Teil ist zum Erzwingen einer Ersten Richtung für die Iteration
		#if i <= 1
    	#Vorgegebene Anfangsrichtung
        #    v = [0.1,-0.1,0]
    	#else
        #    Danach wie bisher
        	v = q - gamma[end]
        #end
		push!(d,LinearAlgebra.norm(v))
		#Überprüfen ob Toleranz erreicht(nahe an q)
	    if LinearAlgebra.norm(v) < tol
			println("Abbruch wegen tol")
			push!(gamma,q)
	        return gamma
	    end
	    v_T_M = project(M, gamma[end], v)
		#Überprüfen od die Projektion von v zu klein
	        if norm(v_T_M) < 1e-12
	            println("Tangentialvektor zu klein")
	           	random_point = rand(3)
				random_v_T_p_M = project(M,p,random_point - q)
				random_v_T_p_M_normalized = normalize(random_v_T_p_M )
				v_T_M = random_v_T_p_M_normalized
	            return gamma
	        end
		#Schritt
	        δv = normalize(v_T_M)
	        δx = gamma[end] + h * δv
	        new_point = project(M, δx)
	        push!(gamma, new_point)
	        i += 1
	    end
		#Abschneiden, falls vorher nicht abgebrochen
	    println("max_iter erreicht")
		final = cut_anfangskurve(M,gamma,q)
		println("Abstand gamma,q",LinearAlgebra.norm(q - gamma[end - 1]))
	    return final
	end
	
#Spezialfall R^3 \to R
#Wird nicht im Newton verwendet
function Vektortransport2_prime(M::Submersion,q,v_0,δx)
	alpha1 = dot(transpose(M.c_prime(q)),transpose(M.c_prime(q)))
	alpha2 = transpose(δx) * M.c_prime_2(q) * v_0
	alpha3 = 2*(only(M.c_prime(q)*v_0))
	alpha4 = (transpose(vec(M.c_prime(q)))*M.c_prime_2(q)*δx)
	alpha5 = (dot(transpose(M.c_prime(q)),transpose(M.c_prime(q))))^2
	alpha = - (alpha1 * alpha2 - alpha3*alpha4)/(alpha5)

	beta1 = only(M.c_prime(q)*v_0)
	beta2 = dot(transpose(M.c_prime(q)),transpose(M.c_prime(q)))
	beta = beta1/beta2
	return alpha * transpose(M.c_prime(q)) - beta * M.c_prime_2(q) * δx
end	

#Spezialfall R^n \to R
#wird nicht im Newton verwendet
function Vektortransport2_prime_2(M::Submersion,x,v,δx)
	A = M.c_prime(x)*transpose(M.c_prime(x))
	term1 = transpose(M.c_prime_2(x))*δx*inv(A)*M.c_prime(x)*v
	klammer = M.c_prime(x)*transpose(M.c_prime_2(x))*δx + transpose(δx)*M.c_prime_2(x)*transpose(M.c_prime(x))
	term2 = transpose(M.c_prime(x))*inv(A)*klammer*inv(A)*M.c_prime(x)*v
	term3 = transpose(M.c_prime(x))*inv(A)*transpose(v)*M.c_prime_2(x)*δx
	return -term1 + term2 -term3
end

#Spezialfall R^n \to R. Implementiert als Ableitung des Sattelpunktproblems
#wird im Newton verwendet
function Vektortransport2_prime_3(M::Submersion,x,v_0,δx)
	A = erstelle_Matrix_Sattel(M,x)
	sol = - inv(A)*vcat(v_0,zeros(M.dim_codomain))
	v_x = sol[1:M.dim_domain]
	λ_x = sol[(M.dim_domain + 1) : M.dim_domain + M.dim_codomain]
	sol_prime =  inv(A)*vcat(transpose(transpose(δx)*M.c_prime_2(x)*only(λ_x)), [transpose(δx)*M.c_prime_2(x)*v_x])
	return sol_prime[1:M.dim_domain]
end


