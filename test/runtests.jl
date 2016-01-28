using CUDA, Base.Test

#Initialize device
dev = CuDevice(0)

#Create context
ctx = create_context(dev)

#Modules include all functions
md = CuModule("thing.ptx")

#Load function
vadd = CuFunction(md, "vadd")
vmul = CuFunction(md, "vmul")
vsub = CuFunction(md, "vsub")
vdiv = CuFunction(md, "vdiv")

#Init
a = rand(Float32, 10)
b = rand(Float32, 10)
ad = CuArray(a)
bd = CuArray(b)

#Addition
c = zeros(Float32, 10)
cd = CuArray(c)
CUDA.launch(vadd, 10, 1, (ad,bd,cd))
c = to_host(cd)
@test_approx_eq c a+b 

#Subtraction
c = zeros(Float32, 10)
cd = CuArray(c)
CUDA.launch(vsub, 10, 1, (ad,bd,cd))
c = to_host(cd)
@test_approx_eq c a-b 

#Multiplication
c = zeros(Float32, 10)
cd = CuArray(c)
CUDA.launch(vmul, 10, 1, (ad,bd,cd))
c = to_host(cd)
@test_approx_eq c a.*b 

#Division
c = zeros(Float32, 10)
cd = CuArray(c)
CUDA.launch(vdiv, 10, 1, (ad,bd,cd))
c = to_host(cd)
@test_approx_eq c a./b 

#Negative test cases
a = rand(Float32, 10)
ad = CuArray(Float32, 5)
try 
	copy!(ad, a)
catch e
	@test typeof(e) == ArgumentError
end
try 
	copy!(a, ad)
catch e
	@test typeof(e) == ArgumentError
end

#Utility
@test CUDA.ndims(ad) == 1
@test CUDA.eltype(ad) == Float32

#Not sure what this does
push(ctx)
pop(ctx)

#Count number of devices
CUDA.devcount()

#Device info
println("Name of device = $(name(dev))")
println("Device capability = $(capability(dev))")
println("Device memory = $(totalmem(dev))")

#Free memory and finish up
free(ad)
free(bd)
free(cd)
unload(md)
destroy(ctx)
