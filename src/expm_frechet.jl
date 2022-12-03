
@doc raw"""
based on algorithm 6.3 of [^AlMohyHigham2009].
"""

# From table 6.1 of [^AlMohyHigham2009]. A list of cut off values $l_m$ used to determine how many terms of Padé approximant is used. To compute expm_frechet(A, E), look for the largest index $m\in [3, 5, 7, 9, 13]$ just exceed $|A|_1$.
ell_table_61 = (
    nothing,
    # 1
    2.11e-8,
    3.56e-4,
    1.08e-2,
    6.49e-2,
    2.00e-1,
    4.37e-1,
    7.83e-1,
    1.23e0,
    1.78e0,
    2.42e0,
    # 11
    3.13e0,
    3.90e0,
    4.74e0,
    5.63e0,
    6.56e0,
    7.52e0,
    8.53e0,
    9.56e0,
    1.06e1,
    1.17e1,
)

@doc raw"""
    _diff_pade3(A, E)
    Compute U, V, LU, LV from the first 6 lines of the for loop in
    algorithm 6.4 of [^AlMohyHigham2009] using 3-term Padé approximant
    the tuple b contains the Padé coefficients of the numerator
"""
@inline function _diff_pade3!(buff, A, E)
    b = (120.0, 60.0, 12.0, 1.0)
    k = size(A)[1]

    @views begin
        U = buff[1:k, 1:end]
        V = buff[(k + 1):(2 * k), 1:end]
        Lu = buff[(2 * k + 1):(3 * k), 1:end]
        Lv = buff[(3 * k + 1):(4 * k), 1:end]

        A2 = buff[(4 * k + 1):(5 * k), 1:end]
        M2 = buff[(5 * k + 1):(6 * k), 1:end]
    end

    mul!(A2, A, A)
    mul!(M2, A, E)
    mul!(M2, E, A, 1, 1)
    mul!(U, A, b[2])
    mul!(U, A, A2, b[4], 1)
    V .= b[3] * A2 + UniformScaling(b[1])
    mul!(Lu, E, V)
    mul!(Lu, A, M2, b[3], 1)
    return mul!(Lv, b[3], M2)
end
@inline function _diff_pade3(A, E)
    b = (120.0, 60.0, 12.0, 1.0)
    A2 = A * A
    M2 = A * E + E * A
    U = A * (b[4] * A2 + UniformScaling(b[2]))
    V = b[3] * A2 + UniformScaling(b[1])
    Lu = A * (b[3] * M2) + E * (b[3] * A2 + UniformScaling(b[1]))
    Lv = b[3] .* M2
    return U, V, Lu, Lv
end

@doc raw"""
    _diff_pade5(A, E)
    Compute U, V, LU, LV from the first 6 lines of the for loop in
    algorithm 6.4 of [^AlMohyHigham2009] using 5-term Padé approximant
    the tuple b contains the Padé coefficients of the numerator
"""
@inline function _diff_pade5!(buff, A, E)
    b = (30240.0, 15120.0, 3360.0, 420.0, 30.0, 1.0)
    k = size(A)[1]
    @views begin
        U = buff[1:k, 1:end]
        V = buff[(k + 1):(2 * k), 1:end]
        Lu = buff[(2 * k + 1):(3 * k), 1:end]
        Lv = buff[(3 * k + 1):(4 * k), 1:end]

        A2 = buff[(4 * k + 1):(5 * k), 1:end]
        M2 = buff[(5 * k + 1):(6 * k), 1:end]

        A4 = buff[(6 * k + 1):(7 * k), 1:end]
        M4 = buff[(7 * k + 1):(8 * k), 1:end]
    end

    mul!(A2, A, A)
    mul!(M2, A, E)
    mul!(M2, E, A, 1, 1)

    mul!(A4, A2, A2)
    mul!(M4, A2, M2)
    mul!(M4, M2, A2, 1, 1)
    Z = b[6] * A4 + b[4] * A2 + UniformScaling(b[2])
    mul!(U, A, Z)
    V .= b[5] * A4 + b[3] * A2 + UniformScaling(b[1])
    mul!(Lu, E, Z)
    mul!(Lu, A, M4, b[6], 1)
    mul!(Lu, A, M2, b[4], 1)

    return Lv .= b[5] * M4 + b[3] * M2
end
@inline function _diff_pade5(A, E)
    b = (30240.0, 15120.0, 3360.0, 420.0, 30.0, 1.0)
    A2 = A * A
    M2 = A * E + E * A
    A4 = A2 * A2
    M4 = A2 * M2 + M2 * A2
    U = A * (b[6] * A4 + b[4] * A2 + UniformScaling(b[2]))
    V = b[5] * A4 + b[3] * A2 + UniformScaling(b[1])
    Lu = (A * (b[6] * M4 + b[4] * M2) + E * (b[6] * A4 + b[4] * A2 + UniformScaling(b[2])))
    Lv = b[5] * M4 + b[3] * M2
    return U, V, Lu, Lv
end

@doc raw"""
    _diff_pade7(A, E)
    Compute U, V, LU, LV from the first 6 lines of the for loop in
    algorithm 6.4 of [^AlMohyHigham2009] using 7-term Padé approximant
    the tuple b contains the Padé coefficients of the numerator
"""
@inline function _diff_pade7!(buff, A, E)
    b = (17297280.0, 8648640.0, 1995840.0, 277200.0, 25200.0, 1512.0, 56.0, 1.0)
    k = size(A)[1]
    @views begin
        U = buff[1:k, 1:end]
        V = buff[(k + 1):(2 * k), 1:end]
        Lu = buff[(2 * k + 1):(3 * k), 1:end]
        Lv = buff[(3 * k + 1):(4 * k), 1:end]

        A2 = buff[(4 * k + 1):(5 * k), 1:end]
        M2 = buff[(5 * k + 1):(6 * k), 1:end]

        A4 = buff[(6 * k + 1):(7 * k), 1:end]
        M4 = buff[(7 * k + 1):(8 * k), 1:end]

        A6 = buff[(8 * k + 1):(9 * k), 1:end]
        M6 = buff[(9 * k + 1):(10 * k), 1:end]
    end

    mul!(A2, A, A)
    mulsym!(M2, A, E)
    mul!(A4, A2, A2)
    mulsym!(M4, A2, M2)

    mul!(A6, A2, A4)
    mul!(M6, A4, M2)
    mul!(M6, M4, A2, 1, 1)

    Z = b[8] * A6 + b[6] * A4 + b[4] * A2 + UniformScaling(b[2])
    mul!(U, A, Z)
    V .= b[7] * A6 + b[5] * A4 + b[3] * A2 + UniformScaling(b[1])
    mul!(Lu, E, Z)
    mul!(Lu, A, b[8] * M6 + b[6] * M4 + b[4] * M2, 1, 1)
    return Lv .= b[7] * M6 + b[5] * M4 + b[3] * M2
end
@inline function _diff_pade7(A, E)
    b = (17297280.0, 8648640.0, 1995840.0, 277200.0, 25200.0, 1512.0, 56.0, 1.0)
    A2 = A * A
    M2 = A * E + E * A
    A4 = A2 * A2
    M4 = A2 * M2 + M2 * A2
    A6 = A2 * A4
    M6 = A4 * M2 + M4 * A2
    U = A * (b[8] * A6 + b[6] * A4 + b[4] * A2 + UniformScaling(b[2]))
    V = b[7] * A6 + b[5] * A4 + b[3] * A2 + UniformScaling(b[1])
    Lu = (
        A * (b[8] * M6 + b[6] * M4 + b[4] * M2) +
        E * (b[8] * A6 + b[6] * A4 + b[4] * A2 + UniformScaling(b[2]))
    )
    Lv = b[7] * M6 + b[5] * M4 + b[3] * M2
    return U, V, Lu, Lv
end

@doc raw"""
    _diff_pade9(A, E)
    Compute U, V, LU, LV from the first 6 lines of the for loop in
    algorithm 6.4 of [^AlMohyHigham2009] using 9-term Padé approximant
    the tuple b contains the Padé coefficients of the numerator
"""
@inline function _diff_pade9!(buff, A, E)
    b = (
        17643225600.0,
        8821612800.0,
        2075673600.0,
        302702400.0,
        30270240.0,
        2162160.0,
        110880.0,
        3960.0,
        90.0,
        1.0,
    )

    k = size(A)[1]
    @views begin
        U = buff[1:k, 1:end]
        V = buff[(k + 1):(2 * k), 1:end]
        Lu = buff[(2 * k + 1):(3 * k), 1:end]
        Lv = buff[(3 * k + 1):(4 * k), 1:end]

        A2 = buff[(4 * k + 1):(5 * k), 1:end]
        M2 = buff[(5 * k + 1):(6 * k), 1:end]

        A4 = buff[(6 * k + 1):(7 * k), 1:end]
        M4 = buff[(7 * k + 1):(8 * k), 1:end]

        A6 = buff[(8 * k + 1):(9 * k), 1:end]
        M6 = buff[(9 * k + 1):(10 * k), 1:end]

        A8 = buff[(10 * k + 1):(11 * k), 1:end]
        M8 = buff[(11 * k + 1):(12 * k), 1:end]
    end

    mul!(A2, A, A)
    mulsym!(M2, A, E)
    mul!(A4, A2, A2)
    mulsym!(M4, A2, M2)
    mul!(A6, A2, A4)
    mul!(M6, A4, M2)
    mul!(M6, M4, A2, 1, 1)
    mul!(A8, A4, A4)
    mulsym!(M8, A4, M4)
    Z = b[10] * A8 + b[8] * A6 + b[6] * A4 + b[4] * A2 + UniformScaling(b[2])
    mul!(U, A, Z)
    V .= b[9] * A8 + b[7] * A6 + b[5] * A4 + b[3] * A2 + UniformScaling(b[1])

    mul!(Lu, E, Z)
    mul!(Lu, A, b[10] * M8 + b[8] * M6 + b[6] * M4 + b[4] * M2, 1, 1)
    return Lv .= b[9] * M8 + b[7] * M6 + b[5] * M4 + b[3] * M2
end
@inline function _diff_pade9(A, E)
    b = (
        17643225600.0,
        8821612800.0,
        2075673600.0,
        302702400.0,
        30270240.0,
        2162160.0,
        110880.0,
        3960.0,
        90.0,
        1.0,
    )
    A2 = A * A
    M2 = A * E + E * A
    A4 = A2 * A2
    M4 = A2 * M2 + M2 * A2
    A6 = A2 * A4
    M6 = A4 * M2 + M4 * A2
    A8 = A4 * A4
    M8 = A4 * M4 + M4 * A4
    U = A * (b[10] * A8 + b[8] * A6 + b[6] * A4 + b[4] * A2 + UniformScaling(b[2]))
    V = b[9] * A8 + b[7] * A6 + b[5] * A4 + b[3] * A2 + UniformScaling(b[1])
    Lu = (
        A * (b[10] * M8 + b[8] * M6 + b[6] * M4 + b[4] * M2) +
        E * (b[10] * A8 + b[8] * A6 + b[6] * A4 + b[4] * A2 + UniformScaling(b[2]))
    )
    Lv = b[9] * M8 + b[7] * M6 + b[5] * M4 + b[3] * M2
    return U, V, Lu, Lv
end

@doc raw"""
    _diff_pade13(A, E)
    Compute U, V, LU, LV from the lines 10-25 of the for loop in
    algorithm 6.4 of [^AlMohyHigham2009] using 9-term Padé approximant
    The returns U, V, Lu, Lv are stored in the first four blocks of buff
    the tuple b contains the Padé coefficients of the numerator
"""
@inline function _diff_pade13!(buff, A, E)
    b = (
        64764752532480000.0,
        32382376266240000.0,
        7771770303897600.0,
        1187353796428800.0,
        129060195264000.0,
        10559470521600.0,
        670442572800.0,
        33522128640.0,
        1323241920.0,
        40840800.0,
        960960.0,
        16380.0,
        182.0,
        1.0,
    )
    k = size(A)[1]

    @views begin
        U = buff[1:k, 1:end]
        V = buff[(k + 1):(2 * k), 1:end]
        Lu = buff[(2 * k + 1):(3 * k), 1:end]
        Lv = buff[(3 * k + 1):(4 * k), 1:end]

        A2 = buff[(4 * k + 1):(5 * k), 1:end]
        M2 = buff[(5 * k + 1):(6 * k), 1:end]

        A4 = buff[(6 * k + 1):(7 * k), 1:end]
        M4 = buff[(7 * k + 1):(8 * k), 1:end]

        A6 = buff[(8 * k + 1):(9 * k), 1:end]
        M6 = buff[(9 * k + 1):(10 * k), 1:end]

        W1 = buff[(10 * k + 1):(11 * k), 1:end]
        Z1 = buff[(11 * k + 1):(12 * k), 1:end]
        W = buff[(12 * k + 1):(13 * k), 1:end]
        Lw1 = buff[(13 * k + 1):(14 * k), 1:end]
        Lz1 = buff[(14 * k + 1):(15 * k), 1:end]
        Lw = buff[(15 * k + 1):(16 * k), 1:end]
    end

    mul!(A2, A, A)
    mul!(M2, A, E)
    mul!(M2, E, A, 1, 1)
    mul!(A4, A2, A2)
    mul!(M4, A2, M2)
    mul!(M4, M2, A2, 1, 1)
    mul!(A6, A2, A4)
    mul!(M6, A4, M2)
    mul!(M6, M4, A2, 1, 1)
    mul!(W1, b[14], A6)
    W1 .+= b[12] .* A4
    W1 .+= b[10] .* A2

    mul!(Z1, b[13], A6)
    Z1 .+= b[11] .* A4
    Z1 .+= b[9] .* A2

    mul!(W, A6, W1)
    W .+= b[8] * A6 + b[6] * A4 + b[4] * A2 + UniformScaling(b[2])

    mul!(U, A, W)
    mul!(V, A6, Z1)
    V .+= b[7] * A6 + b[5] * A4 + b[3] * A2 + UniformScaling(b[1])

    Lw1 .= b[14] * M6 + b[12] * M4 + b[10] * M2
    mul!(Lz1, b[13], M6)
    Lz1 .+= b[11] * M4 + b[9] * M2

    mul!(Lw, A6, Lw1)
    mul!(Lw, M6, W1, 1, 1)
    Lw .+= b[8] * M6 + b[6] * M4 + b[4] * M2

    mul!(Lu, A, Lw)
    mul!(Lu, E, W, 1, 1)
    mul!(Lv, A6, Lz1)
    mul!(Lv, M6, Z1, 1, 1)
    return Lv .+= b[7] * M6 + b[5] * M4 + b[3] * M2
end
@inline function _diff_pade13(A, E)
    b = (
        64764752532480000.0,
        32382376266240000.0,
        7771770303897600.0,
        1187353796428800.0,
        129060195264000.0,
        10559470521600.0,
        670442572800.0,
        33522128640.0,
        1323241920.0,
        40840800.0,
        960960.0,
        16380.0,
        182.0,
        1.0,
    )

    A2 = A * A

    M2 = A * E + E * A
    A4 = A2 * A2
    M4 = A2 * M2 + M2 * A2
    A6 = A2 * A4
    M6 = A4 * M2 + M4 * A2
    W1 = b[14] * A6 + b[12] * A4 + b[10] * A2
    W2 = b[8] * A6 + b[6] * A4 + b[4] * A2 + UniformScaling(b[2])
    Z1 = b[13] * A6 + b[11] * A4 + b[9] * A2
    Z2 = b[7] * A6 + b[5] * A4 + b[3] * A2 + UniformScaling(b[1])
    W = A6 * W1 + W2
    U = A * W
    V = A6 * Z1 + Z2
    Lw1 = b[14] * M6 + b[12] * M4 + b[10] * M2
    Lw2 = b[8] * M6 + b[6] * M4 + b[4] * M2
    Lz1 = b[13] * M6 + b[11] * M4 + b[9] * M2
    Lz2 = b[7] * M6 + b[5] * M4 + b[3] * M2
    Lw = A6 * Lw1 + M6 * W1 + Lw2
    Lu = A * Lw + E * W
    Lv = A6 * Lz1 + M6 * Z1 + Lz2
    return U, V, Lu, Lv
end

@doc raw"""
    expm_frechet(A, E)
    expm_frechet!(buff, A, E)

Compute Frechet derivative of expm(A) in direction E using algorithm 6.4 of [^AlMohyHigham2009]
For sufficiently small $|A|_1$ norm, we use Padé with appropriate number of terms
(3, 5, 7, 9, 13), with 13 terms is the default. Otherwise we use the formula
exp(A) = (exp(A/2^s))^{2^s}, where s is used to scale so $|A|_1$ is smaller than ell_table_61[14] of tbale ell_table_61.

For expm_frechet!, buff is a matrix of size 16*k times k
the returns, eA = exp(A), eAf = dexp(A, E) are stored in the first two blocks
the remaining blocks are used as temporary storage

[^AlMohyHigham2009]:
    >Al-Mohy, A. H., Higham, N. J. (2009)
    >"Computing the Frechet Derivative of the Matrix Exponential, with an application to Condition Number Estimation."
    >SIAM Journal On Matrix Analysis and Applications., 30 (4). pp. 1639-1657. ISSN 1095-7162
    >doi: [https://doi.org/10.1137/080716426](https://doi.org/10.1137/080716426)
"""
function expm_frechet(A, E)
    n = size(A, 1)
    s = nothing
    A_norm_1 = maximum(sum(abs.(A), dims=1))
    m_pade_pairs = ((3, _diff_pade3), (5, _diff_pade5), (7, _diff_pade7), (9, _diff_pade9))

    for m_pade in m_pade_pairs
        m, pade = m_pade
        if A_norm_1 <= ell_table_61[m]
            U, V, Lu, Lv = pade(A, E)
            s = 0
            break
        end
    end
    if isnothing(s)
        # scaling
        s = max(0, Int(ceil(log2(A_norm_1 / ell_table_61[14]))))
        # pade order 13
        U, V, Lu, Lv = _diff_pade13((2.0^-s) * A, (2.0^-s) * E)
    end
    # factor once and solve twice    
    lu_piv = lu(-U + V)
    eA = lu_piv \ (U + V)
    eAf = lu_piv \ (Lu + Lv + (Lu - Lv) * eA)

    # squaring
    for k in 1:s
        eAf = eA * eAf + eAf * eA
        eA = eA * eA
    end

    return eA, eAf
end

@doc raw"""
    expm_frechet!(buff, A, E)

Compute Frechet derivative of expm(A) in direction E using algorithm 6.4 of [^AlMohyHigham2009]
For sufficiently small $|A|_1$ norm, we use Padé with appropriate number of terms
(3, 5, 7, 9, 13), with 13 terms is the default. Otherwise we use the formula
exp(A) = (exp(A/2^s))^{2^s}, where s is used to scale so $|A|_1$ is smaller than ell_table_61[14] of tbale ell_table_61.

For expm_frechet!, buff is a matrix of size 16*k times k
the returns, eA = exp(A), eAf = dexp(A, E) are stored in the first two blocks
the remaining blocks are used as temporary storage
"""
function expm_frechet!(buff, A, E)
    n = size(A, 1)
    s = nothing
    A_norm_1 = maximum(sum(abs.(A), dims=1))
    k = size(A)[1]
    m_pade_pairs =
        ((3, _diff_pade3!), (5, _diff_pade5!), (7, _diff_pade7!), (9, _diff_pade9!))

    for m_pade in m_pade_pairs
        m, pade = m_pade
        if A_norm_1 <= ell_table_61[m]
            U, V, Lu, Lv = pade(buff, A, E)
            s = 0
            break
        end
    end
    if isnothing(s)
        # scaling
        s = max(0, Int(ceil(log2(A_norm_1 / ell_table_61[14]))))
        # pade order 13
        _diff_pade13!(buff, (2.0^-s) * A, (2.0^-s) * E)
    end
    buff[(4 * k + 1):(8 * k), :] .= buff[1:(4 * k), :]
    @views begin
        eA = buff[1:k, :]
        eAf = buff[(k + 1):(2 * k), :]
        tmp = buff[(2 * k + 1):(3 * k), :]

        U = buff[(4 * k + 1):(5 * k), :]
        V = buff[(5 * k + 1):(6 * k), :]
        Lu = buff[(6 * k + 1):(7 * k), :]
        Lv = buff[(7 * k + 1):(8 * k), :]
    end

    # factor once and solve twice    
    lu_piv = lu(-U + V)
    broadcast!(+, eA, U, V)
    ldiv!(lu_piv, eA)
    broadcast!(-, tmp, Lu, Lv)
    mul!(eAf, tmp, eA)
    eAf .+= Lu .+ Lv
    ldiv!(lu_piv, eAf)

    # squaring
    for k in 1:s
        mulsym!(tmp, eA, eAf)
        eAf .= tmp
        mul!(tmp, eA, eA)
        eA .= tmp
    end
end

@doc raw"""
    mulsym!(C, A, E)
    Compute C = A*E + E*A by mul    
"""
@inline function mulsym!(C, A, B)
    mul!(C, A, B)
    return mul!(C, B, A, 1, 1)
end
