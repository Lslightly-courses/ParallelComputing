import subprocess
from os import system


system('make all')
LOOPS = 3   # 循环次数

# 串行执行
serial_exec = subprocess.run('./serialSpMV {0}'.format(LOOPS), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True)
# 结果通过stdout输出
serial_result = serial_exec.stdout.decode(encoding='utf-8')

# 并行执行，不同的进程数
list_perf = []
for procs in range(1, 13):
    print(procs)
    mpi_exec = subprocess.run('mpiexec -n {1} ./parallelSpMV {0} {1}'.format(LOOPS, procs), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True)
    # 结果通过stdout输出
    mpi_result = mpi_exec.stdout.decode(encoding='utf-8')

    # 时间统计通过stderr输出
    perf_str = mpi_exec.stderr.decode(encoding='utf-8').removesuffix('\n')
    print(perf_str)
    list_perf.append(perf_str.split(' '))

    # 结果比较
    if mpi_result != serial_result:
        print('结果不同')
    else:
        print('结果相同')
with open('output/serial.txt', 'w') as f:
    f.write(serial_result)
with open('output/mpi.txt', 'w') as f:
    f.write(mpi_result)
with open('output/perf.txt', 'w') as f:
    f.write('进程数 总执行时间/ms 计算比例 通信比例\n')
    for procs_1, perf in enumerate(list_perf):
        f.write('{} {} {} {}\n'.format(procs_1+1, perf[0], perf[1], perf[2]))