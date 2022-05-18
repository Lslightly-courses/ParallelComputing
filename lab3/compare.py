import subprocess
from os import system


system('make all')
LOOPS = 3
serial_exec = subprocess.run('./serialSpMV {0}'.format(LOOPS), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True)
serial_result = serial_exec.stdout.decode(encoding='utf-8')

list_perf = []
for procs in range(1, 13):
    print(procs)
    mpi_exec = subprocess.run('mpiexec -n {1} ./parallelSpMV {0} {1}'.format(LOOPS, procs), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True)
    mpi_result = mpi_exec.stdout.decode(encoding='utf-8')
    perf_str = mpi_exec.stderr.decode(encoding='utf-8').removesuffix('\n')
    print(perf_str)
    list_perf.append(perf_str.split(' '))
    if mpi_result != serial_result:
        print('结果不同')
    else:
        print('结果相同')
with open('serial.txt', 'w') as f:
    f.write(serial_result)
with open('mpi.txt', 'w') as f:
    f.write(mpi_result)
with open('perf.txt', 'w') as f:
    f.write('进程数 总执行时间/ms 计算比例 通信比例\n')
    for procs_1, perf in enumerate(list_perf):
        f.write('{} {} {} {}\n'.format(procs_1+1, perf[0], perf[1], perf[2]))