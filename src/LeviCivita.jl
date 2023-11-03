"""
    LeviCivita{dim}()

Iterate over the non-zero indices in the Levi-Civita symbol, ε.
Used to calculate the determinant for higher-order tensors. 

Example
```julia
for ((i,j), ε) in LeviCivita{2}()
    println("ε($i,$j) = $ε")
end
for ((i,j,k), ε) in LeviCivita{3}()
    println("ε($i,$j,$k) = $ε")
end
```
"""
struct LeviCivita{dim} end
LeviCivita{1}() = (((1,), 1))
LeviCivita{2}() = (((1,2), 1), ((2,1), -1))
LeviCivita{3}() = (((1,2,3), 1), ((2,3,1), 1), ((3,1,2), 1), ((3,2,1), -1), ((2,1,3), -1), ((1,3,2), -1))

LinearAlgebra.det(A::Tensor{4,1,T}) = get_data(A)[1]
LinearAlgebra.det(A::SymmetricTensor{4,1,T}) = get_data(A)[1]
LinearAlgebra.det(::SymmetricTensor{4,<:Any,T}) = zero(T)

function LinearAlgebra.det(A::Tensor{4,2,T}) where T
    det_A = zero(T)
    for ((i1,j1), ε1) in LeviCivita{2}()
        for ((i2,j2), ε2) in LeviCivita{2}()
            for ((i3,j3), ε3) in LeviCivita{2}()
                for ((i4,j4), ε4) in LeviCivita{2}()
                    det_A += ε1*ε2*ε3*ε4*A[i1,i2,i3,i4]*A[j1,j2,j3,j4]
                end
            end
        end
    end
    return det_A/2
end

function LinearAlgebra.det(A::Tensor{4,3,T}) where T
    det_A = zero(T)
    for ((i1,j1,k1), ε1) in LeviCivita{3}()
        for ((i2,j2,k2), ε2) in LeviCivita{3}()
            for ((i3,j3,k3), ε3) in LeviCivita{3}()
                for ((i4,j4,k4), ε4) in LeviCivita{3}()
                    det_A += ε1*ε2*ε3*ε4*A[i1,i2,i3,i4]*A[j1,j2,j3,j4]*A[k1,k2,k3,k4]
                end
            end
        end
    end
    return det_A/6
end