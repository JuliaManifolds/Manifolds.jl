function c_sphaere(x) # hier noch konkret 
	return x[1]^2 + x[2]^2 + x[3]^2- 1
end
	
function c_prime_sphaere(x) # hier noch konkret
	return [ 2*x[1] 2*x[2]  2*x[3] ]
end
	
function c_prime_2_sphaere(x)
	return [2 0 0;
		    0 2 0;
		    0 0 2]
end

function c_sphaere_5(x) # hier noch konkret 
	return x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2 + x[5]^2- 1
end
	
function c_prime_sphaere_5(x) # hier noch konkret
	return [ 2*x[1] 2*x[2]  2*x[3] 2*x[4] 2*x[5] ]
end
	
function c_prime_2_sphaere_5(x)
	return [2 0 0 0 0;
		    0 2 0 0 0;
		    0 0 2 0 0;
		    0 0 0 2 0;
			0 0 0 0 2]
end


#Test auf Übertragung für die Produktmannigfaltigkeit
S = Submersion(c_sphaere,c_prime_sphaere,c_prime_2_sphaere,2,3,1)
powerS = PowerManifold(S, NestedPowerRepresentation(), 3)
p = [10.,5.,0.]
p_power = [[0.,0.,9.],[1.,35.,0.],[1.,23.,0.]]
v_power = [[0.,9.,0.],[0.,0.,1.],[0.,0.,1.]]

test = [project(S,p_power[1]+v_power[1]),project(S,p_power[2]+v_power[2]),project(S,p_power[3]+v_power[3])]
print(test - retract(powerS, p_power, v_power, ProjectionRetraction()))


#Test für R^n \to R (n=5)
S5 = Submersion(c_sphaere_5,c_prime_sphaere_5,c_prime_2_sphaere_5,4,5,1)
powerS5 = PowerManifold(S5, NestedPowerRepresentation(), 3)
p5 = [1.,2.,3.,4.,5.]
p5 = project(S5,p5)
println(LinearAlgebra.norm(p5))
println(S5.c(p5))


x5 = [1.,0.,0.,0.,0.]
v5 = [0.,2.,2.,2.,2.]
check_vector(S5,x5,v5)

x_neu_5 = project(S5,[1.2,0.2,0.1,0.,0.])
println(x_neu_5)
v_neu_5 = project(S5,x_neu_5,v5)
println(S5.c_prime(x_neu_5)*v_neu_5)


#Beispiel für Abbildung R^4 \to R^3
function c(x)
    x1, x2, x3, x4 = x
    return [x1^2 + x2^2 + x3^2 + x4^2 - 1,
            x1 + x2 + x3 + x4,
            x1 - x2 + x3 - x4]
end
	
function c_prime(x)
    x1, x2, x3, x4 = x
    return [2x1  2x2  2x3  2x4;
             1    1    1    1;
             1   -1    1   -1]
end


general = Submersion(c,c_prime,c_prime_2_sphaere,1,4,3)

p_test = [1.,2.,3.,4.]

#test nichtlineare Projektio
println("Punkt:",p_test)
println("c(punkt):",c(p_test))

p_neu = project(general,p_test)
println("Punkt nach Projektion:",p_neu)
println("c angewendet auf neuen Punkt",c(p_neu))


#test lineare Projektion
v_test = [1.,0.,2.,3.]
v_T_M = project(general,p_neu,v_test)
println("Vektor:",v_test)
println("Projizierter Vektor:",v_T_M)
println("c'(p)*v_T_M (sollte [0,0,0] sein):",general.c_prime(p_neu)*v_T_M)
