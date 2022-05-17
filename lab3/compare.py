import subprocess
from os import system
import time

system('make all')
start = time.time()
serial_exec = subprocess.run('./serialSpMV', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True)
end = time.time()
duration = end-start
print('串行执行时间 {}ms'.format(duration*1000))
print(serial_exec.stderr.decode(encoding='utf-8'))
with open('perf.txt', 'w') as f:
    f.write('{} {}\n'.format('算法', '执行时间/ms'))
    f.write('{} {}\n'.format('串行', duration*1000))
serial_result = serial_exec.stdout.decode(encoding='utf-8')

for procs in range(1, 12):
    print(procs)
    start = time.time()
    mpi_exec = subprocess.run('mpiexec -n {0} ./parallelSpMV {0}'.format(procs), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True)
    end = time.time()
    duration = end-start
    # print('MPI执行时间 {}ms'.format(duration*1000))
    mpi_result = mpi_exec.stdout.decode(encoding='utf-8')
    print(mpi_exec.stderr.decode(encoding='utf-8'))
    if mpi_result != serial_result:
        print('结果不同')
    else:
        print('结果相同')
    with open('serial.txt', 'w') as f:
        f.write(serial_result)
    with open('mpi.txt', 'w') as f:
        f.write(mpi_result)
    with open('perf.txt', 'a') as f:
        f.write('{} {}\n'.format(procs, duration*1000))
