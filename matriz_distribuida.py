from mpi4py import MPI
import random
import time
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 1000
if len(sys.argv) > 1:
    N = int(sys.argv[1])

A = None
B = None

if rank == 0:
    A = [[random.random() for _ in range(N)] for _ in range(N)]
    B = [[random.random() for _ in range(N)] for _ in range(N)]

inicio = time.time()

# Broadcast das matrizes A e B para todos os ranks
A = comm.bcast(A, root=0)
B = comm.bcast(B, root=0)

# Divisão das linhas entre os processos
linhas_por_processo = N // size
inicio_linha = rank * linhas_por_processo
fim_linha = N if rank == size - 1 else (rank + 1) * linhas_por_processo

sub_C = []
for i in range(inicio_linha, fim_linha):
    linha = [0] * N
    for j in range(N):
        soma = 0
        for k in range(N):
            soma += A[i][k] * B[k][j]
        linha[j] = soma
    sub_C.append(linha)

# Gather para reunir todas as submatrizes no processo 0
partes_C = comm.gather(sub_C, root=0)

if rank == 0:
    C = []
    for parte in partes_C:
        C.extend(parte)
    fim = time.time()
    print(f"Matriz {N}x{N} Distribuída ({size} procs): {(fim - inicio) * 1000:.2f} ms")