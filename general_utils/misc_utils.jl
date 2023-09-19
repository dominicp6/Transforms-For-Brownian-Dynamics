module MiscUtils
export init_q0, assert_isotropic_diagonal_diffusion, is_identity_diffusion, create_directory_if_not_exists

function init_q0(q0; dim::Int = 1) 
    if q0 === nothing
        q0 = randn(dim)
    end
    if dim == 1
        q0 = q0[1]
    end
    return q0
end

"""
Assert that D is a diagonal, isotropic matrix
"""
function assert_isotropic_diagonal_diffusion(D) 
    D1 = (x,y) -> D(x,y)[1,1]
    D2 = (x,y) -> D(x,y)[2,2]
    Doff1 = (x,y) -> D(x,y)[1,2]
    Doff2 = (x,y) -> D(x,y)[2,1]
    @assert Doff1(0.123,-0.736) == Doff2(0.123,-0.736) == 0 "D must be diagonal"
    @assert D1(0.123,-0.736) == D2(0.123,-0.736) "D must be isotropic"
end

function is_identity_diffusion(D)
    if D(0,0) == D(1,1) == D(-0.2354345, 0.21267) == I
        identity_diffusion = true
    else
        identity_diffusion = false
    end

    return identity_diffusion
end

function create_directory_if_not_exists(dir_path)
    if !isdir(dir_path)
        mkpath(dir_path)
        @info "Created directory $dir_path"
    end
end

end # module MiscUtils