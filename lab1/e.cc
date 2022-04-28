#include <gmpxx.h>
#include <omp.h>
#include <sys/time.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <iomanip>

using namespace std;

#define N 12
#define begin_LOOPs 96000000
#define step_LOOPs 96000
#define PAD 8

ofstream resultout, profileout;

typedef struct timeval timeval;

size_t time_use(timeval start, timeval end) {
    return (end.tv_sec-start.tv_sec)*1000000+end.tv_usec-start.tv_usec;
}

void init_out() {
    resultout.unsetf(ios::fixed);
    profileout.unsetf(ios::fixed);
    resultout << "线程数,循环次数,结果\n";
    profileout << "线程数,循环次数,执行时间(us)\n";
}

void out_result(int n_proc, size_t LOOPs, mpf_t sum) {
    resultout << n_proc << "," << LOOPs << ",";
    mp_exp_t tmp = 0;
    char *out_str = mpf_get_str(nullptr, &tmp, 10, 100, sum);
    resultout << out_str[0] << '.' << out_str+1 << endl;
    free(out_str);
}

void out_profile(int n_proc, size_t LOOPs, size_t T) {
    profileout << n_proc << "," << LOOPs << "," << T << endl;
}

void no_parallel_gmp(size_t LOOPs) {
    size_t T;
    timeval start, end;

    gettimeofday(&start, nullptr);
    mpf_set_default_prec(334);
    mpf_t sum;
    mpf_init(sum);
    mpf_set_d(sum, 1.0);
    size_t k = 1;
    mpf_t part;
    mpf_init(part);
    mpf_set_d(part, 1.0);
    for (; k < LOOPs; k++) {
        mpf_div_ui(part, part, k);
        mpf_add(sum, sum, part);
    }
    gettimeofday(&end, nullptr);
    T = time_use(start, end);
    out_result(1, LOOPs, sum);
    out_profile(1, LOOPs, T);
    mpf_clear(sum);
    mpf_clear(part);
}

void parallel_gmp_no_class(int n_proc, size_t LOOPs) {
    size_t T;
    timeval start, end;

    gettimeofday(&start, nullptr);
    mpf_set_default_prec(334);
    mpf_t result[2*N];
    for (int i = 0; i < 2*N; i++) {
        mpf_init(result[i]);
    }
    mpf_t sum;
    mpf_init(sum);
    mpf_set_d(sum, 1.0);
    mpf_t tmp;
    mpf_init(tmp);

    omp_set_num_threads(n_proc);
    size_t block = LOOPs/n_proc;

    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        size_t start_bound = id * block+1, end_bound = (id+1)*block+1;
        mpf_t part_res;
        mpf_init(part_res);
        mpf_set_d(part_res, 0.0);
        mpf_t fact_reci;
        mpf_init(fact_reci);
        mpf_set_d(fact_reci, 1.0);

        for (size_t i = start_bound; i < end_bound; i++) {
            mpf_div_ui(fact_reci, fact_reci, i);
            mpf_add(part_res, part_res, fact_reci);
        }

        mpf_set(result[2*id], part_res);
        mpf_set(result[2*id+1], fact_reci);

        mpf_clear(part_res);
        mpf_clear(fact_reci);
    }

    mpf_add(sum, result[0], sum);
    for (int i = 1; i < n_proc; i++) {
        mpf_mul(tmp, result[2*i-1], result[2*i]);
        mpf_add(sum, sum, tmp);
        mpf_mul(result[2*i+1], result[2*i-1], result[2*i+1]);
    }

    gettimeofday(&end, nullptr);
    T = time_use(start, end);

    out_result(n_proc, LOOPs, sum);
    out_profile(n_proc, LOOPs, T);

    mpf_clear(tmp);
    mpf_clear(sum);
    for (int i = 0; i < 2*N; i++) {
        mpf_clear(result[i]);
    }
}

size_t add_W(int n_proc) {
    return (n_proc-1)*step_LOOPs+begin_LOOPs;
}

int main(void)
{
    resultout.open("result.csv");
    profileout.open("profile.csv");
    init_out();
    parallel_gmp_no_class(12, begin_LOOPs);
    parallel_gmp_no_class(10, begin_LOOPs);
    parallel_gmp_no_class(8, begin_LOOPs);
    parallel_gmp_no_class(6, begin_LOOPs);
    parallel_gmp_no_class(4, begin_LOOPs);
    parallel_gmp_no_class(2, begin_LOOPs);
    parallel_gmp_no_class(1, begin_LOOPs);
    resultout.close();
    profileout.close();
    return 0;
}
