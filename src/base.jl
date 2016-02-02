
# Load & initialize CUDA driver

# const libcuda = dlopen("libcuda")

macro cucall(fv, argtypes, args...)
    f = eval(fv)
    quote
        _curet = ccall( ($(Meta.quot(f)), CUDA_LIB), Cint, $argtypes, $(args...) )
        if _curet != 0
            throw(CuDriverError(Int(_curet)))
        end
    end
end

function initialize()
    @cucall(cuInit, (Cint,), 0)
    println("CUDA Driver Initialized")
end

initialize()


# Get driver version

function driver_version()
    a = Cint[0]
    @cucall(cuDriverGetVersion, (Ptr{Cint},), a)
    return Int(a[1])
end

const DriverVersion = driver_version()

if DriverVersion < 4000
    error("CUDA of version 4.0 or above is required.")
end


# box a variable into array

cubox{T}(x::T) = T[x]

