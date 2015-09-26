## CUDA.jl
<!-- Version translated into Portuguese the README -->

Interface de programação Julia para CUDA.

Este pacote envolve as principais funções do Driver CUDA API para Julia. Embora esta é uma obra inacabada, o uso simples está pronto. Veja também o pacote [CUDArt package](https://github.com/JuliaGPU/CUDArt.jl). 

**Observação:** Este pacote foi testado em Ubuntu (13.04 ou superior) e MAC OS X (10.8+). Ele não foi testado completamente no Windows.

### Configuração

1. Instalar driver CUDA, e certifique-se que ``libcuda`` está em no caminho correto da biblioteca. 

   **Observação:** ``libcuda`` é uma biblioteca compartilhada para o driver CUDA. 

2. Verificação do pacote em Julia:

	```julia
	Pkg.add("CUDA")
	```

3. Teste se funciona através da execução do script de exemplo em ``examples/ex1.jl``.

4. Pronto.


### Exemplo

O exemplo a seguir mostra como é possível usar este pacote para adicionar duas matrizes em GPU

##### Write CUDA Kernel

Primeiro você tem que escrever o kernel em CUDA que é desenvolvido em C e salvá-lo em um arquivo .cu. Aqui está
o kernel como exemplo:

```C
// filename: vadd.cu
// um kernel simples em CUDA para adicionar dois vetores.

extern "C"   // assegurar que o nome da função seja "vadd"
{
	__global__ void vadd(const float *a, const float *b, float *c)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		c[i] = a[i] + b[i];
	}
}

```
Você pode compilar o kernel para um arquivo PTX usando ``nvcc``, através do seguinte comando

```
nvcc -ptx vadd.cu
```
Este comando gerará um arquivo chamado ``vadd.ptx``.

##### Executando o kernel in Julia

O script a seguir demonstra como se pode carregar o kernel e executá-lo em Julia.

```julia
using CUDA

# selecione o dispositivo CUDA
dev = CuDevice(0)

# criar um contexto (como um processo no CPU) no dispositivo selecionado
ctx = create_context(dev)

# carregar o módulo PTX (cada módulo pode conter múltiplas funções do kernel)
md = CuModule("vadd.ptx")

# recupera a função do kernel "vadd" do módulo
vadd = CuFunction(md, "vadd")

# geração de matrizes aleatórias e carregá-los para GPU
a = round(rand(Float32, (3, 4)) * 100)
b = round(rand(Float32, (3, 4)) * 100)
ga = CuArray(a)
gb = CuArray(b)

# criação de uma matriz em GPU para armazenar os resultados
gc = CuArray(Float32, (3, 4))

# execução do kernel vadd
# sintaxe: launch(kernel, grid_size, block_size, arguments)
# aqui, grid_size e block_size pode ser um inteiro ou uma tupla de inteiros
launch(vadd, 12, 1, (ga, gb, gc))

# download dos resultados que vieram da GPU
c = to_host(gc)   # c is a Julia array on CPU (host)

# comando para limpar a memória da GPU
free(ga)
free(gb)
free(gc)

# impressão dos resultados
println("Results:")
println("a = \n$a")
println("b = \n$b")
println("c = \n$c")

# Término: módulo de finalizar e de destruir o contexto
unload(md)
destroy(ctx)
```

Esta é uma API relativamente de baixo nível e é projetado para pessoas que tem algum conhecimento de programação CUDA para escrever / migrarcódigos CUDA em Julia. 
Em comparação com CUDA desenvolvido em C, a interface foi muito simplicada.
