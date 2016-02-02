/* 
 * File:   main.cpp
 * Author: Abuenameh
 *
 * Created on 17 November 2014, 22:05
 */

#include <cstdlib>
#include <queue>

using namespace std;

#include <boost/lexical_cast.hpp>
#include <boost/thread.hpp>
#include <boost/progress.hpp>
#include <boost/random.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/date_time.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/string.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

using namespace boost;
using namespace boost::random;
using namespace boost::filesystem;
using namespace boost::posix_time;
//using namespace boost::interprocess;

using boost::interprocess::shared_memory_object;
using boost::interprocess::managed_shared_memory;
using boost::interprocess::create_only;
using boost::interprocess::open_only;
//using boost::interprocess::allocator;
using boost::interprocess::basic_string;
using boost::interprocess::interprocess_mutex;
using boost::interprocess::interprocess_condition;
//using boost::interprocess::interprocess_;

#include <boost/process.hpp>

using namespace boost::process;
using namespace boost::process::initializers;

#include <nlopt.hpp>

using namespace nlopt;

#include "gutzwiller.hpp"
#include "mathematica.hpp"
#include "casadimath.hpp"
#include "orderparameter.hpp"
#include "ode.hpp"
#include "jac.hpp"


#include <casadi/interfaces/sundials/cvodes_interface.hpp>

typedef managed_shared_memory::segment_manager segment_manager_t;

typedef interprocess::allocator<void, segment_manager_t> void_allocator;

typedef interprocess::allocator<double, segment_manager_t> double_allocator;
typedef interprocess::vector<double, double_allocator> double_vector;

typedef interprocess::allocator<complex<double>, segment_manager_t> complex_allocator;
typedef interprocess::vector<complex<double>, complex_allocator> complex_vector;
typedef interprocess::allocator<complex_vector, segment_manager_t> complex_vector_allocator;
typedef interprocess::vector<complex_vector, complex_vector_allocator> complex_vector_vector;
typedef interprocess::allocator<complex_vector_vector, segment_manager_t> complex_vector_vector_allocator;
typedef interprocess::vector<complex_vector_vector, complex_vector_vector_allocator> complex_vector_vector_vector;

typedef interprocess::allocator<char, segment_manager_t> char_allocator;
typedef interprocess::basic_string<char, std::char_traits<char>, char_allocator> char_string;

struct worker_input {
    double Wi;
    double Wf;
    double mu;
    double_vector xi;
    double U0;
    double_vector J0;
    double_vector x0;
    complex_vector_vector f0;
    char_string integrator;
    double dt;
    int ngrid;

    worker_input(const void_allocator& void_alloc) : xi(void_alloc), J0(void_alloc), x0(void_alloc), f0(void_alloc), integrator(void_alloc) {
    }
};

struct worker_tau {
    interprocess_mutex mutex;
    interprocess_condition cond_empty;
    interprocess_condition cond_full;

    double tau;

    bool full;

    worker_tau() : full(false) {
    }
};

struct worker_output {
    double Ei;
    double Ef;
    double Q;
    double p;
    double_vector ts;
    double_vector Es;
    complex_vector b0;
    complex_vector bf;
    complex_vector_vector f0;
    complex_vector_vector ff;
    complex_vector_vector_vector fs;
    char_string runtime;
    bool success;

    interprocess_mutex mutex;
    interprocess_condition cond_empty;
    interprocess_condition cond_full;

    bool full;

    worker_output(const void_allocator& void_alloc) : ts(void_alloc), Es(void_alloc), b0(void_alloc), bf(void_alloc), f0(void_alloc), ff(void_alloc), fs(void_alloc), runtime(void_alloc), full(false) {
    }
};

struct results {
    double tau;
    double Ei;
    double Ef;
    double Q;
    double p;
    double U0;
    vector<double> ts;
    vector<double> Es;
    vector<double> J0;
    vector<complex<double>> b0;
    vector<complex<double>> bf;
    vector<vector<complex<double>>> f0;
    vector<vector<complex<double>>> ff;
    vector<vector<vector<complex<double>>>> fs;
    string runtime;

    //    results(const void_allocator& void_alloc) : Es(void_alloc), J0(void_alloc), b0(void_alloc), bf(void_alloc), f0(void_alloc), ff(void_alloc), runtime(void_alloc) {
    //    }
};

double UWi(double W) {
    return -2 * (g24 * g24) / Delta * (Ng * Ng * W * W) / ((Ng * Ng + W * W) * (Ng * Ng + W * W));
}

complex<double> dot(vector<complex<double>>&v, vector<complex<double>>&w) {
    complex<double> res = 0;
    for (int i = 0; i < v.size(); i++) {
        res += ~v[i] * w[i];
    }
    return res;
}

struct input {
    double tau;
};

class SumFunction : public Callback2 {
public:

    SumFunction(vector<Function>& fs_) : fs(fs_) {
        nf = fs.size();
        nin = fs[0].nIn();
        nout = fs[0].nOut();
        for (int i = 0; i < nin; i++) {
            inSparsity.push_back(fs[0].inputSparsity(i));
        }
        for (int i = 0; i < nout; i++) {
            outSparsity.push_back(fs[0].outputSparsity(i));
        }
    }

    int nIn() {
        return nin;
    }

    int nOut() {
        return nout;
    }

    Sparsity inputSparsity(int i) {
        return inSparsity[i];
    }

    Sparsity outputSparsity(int i) {
        return outSparsity[i];
    }

    std::vector<DMatrix> operator()(const std::vector<DMatrix>& arg) {
        vector<DMatrix> res(nout);
        for (int i = 0; i < nout; i++) {
            res[i] = DMatrix(outSparsity[i], 0);
        }
        for (int i = 0; i < nf; i++) {
            vector<DMatrix> resi = fs[i](arg);
            for (int j = 0; j < nout; j++) {
                res[j] += resi[j];
            }
        }
        //                cout << res << endl;
        //                exit(0);
        return res;
    }

private:
    vector<Function>& fs;
    int nf;
    int nin;
    int nout;
    vector<Sparsity> inSparsity;
    vector<Sparsity> outSparsity;
};

class ODEFunction : public Callback2 {
public:

    ODEFunction() {
        inSparsity.push_back(Sparsity::dense(2 * L * dim));
        inSparsity.push_back(Sparsity::dense(0, 0));
        inSparsity.push_back(Sparsity::dense(L + 4));
        inSparsity.push_back(Sparsity::dense(1));

        outSparsity.push_back(Sparsity::dense(2 * L * dim));
        outSparsity.push_back(Sparsity::dense(0, 0));
        outSparsity.push_back(Sparsity::dense(0, 0));
    }

    int nIn() {
        return DAE_NUM_IN;
    }

    int nOut() {
        return DAE_NUM_OUT;
    }

    Sparsity inputSparsity(int i) {
        return inSparsity[i];
    }

    Sparsity outputSparsity(int i) {
        return outSparsity[i];
    }

    std::vector<DMatrix> operator()(const std::vector<DMatrix>& arg) {
        vector<double> fin = arg[0].nonzeros();
        vector<double> p = arg[2].nonzeros();
        double t = arg[3].toScalar();

        double Wi = p[L];
        double Wf = p[L + 1];
        double mu = p[L + 2];
        double tau = p[L + 3];

        double Wt, Wtp;
        if (t < tau) {
            Wt = Wi + (Wf - Wi) * t / tau;
            Wtp = (Wf - Wi) / tau;
        } else {
            Wt = Wf + (Wi - Wf) * (t - tau) / tau;
            Wtp = (Wi - Wf) / tau;
        }

        vector<double> J(L), Jp(L), dU(L);
        double U0 = UW(Wt);
        double U0p = UWp(Wt, Wtp);
        for (int i = 0; i < L; i++) {
            J[i] = JWij(Wt * p[i], Wt * p[mod(i + 1)]);
            Jp[i] = JWijp(Wt * p[i], Wt * p[mod(i + 1)], Wtp * p[i], Wtp * p[mod(i + 1)]);
            dU[i] = UW(Wt * p[i]) - U0;
        }

        vector<double> odes(2 * L * dim);
        for (int i = 0; i < 2 * L * dim; i++) {
            odes[i] = ode(i, fin, J, Jp, U0, U0p, dU, mu);
        }

        return vector<DMatrix>{DMatrix(odes), DMatrix(), DMatrix()};
    }

private:
    vector<Sparsity> inSparsity;
    vector<Sparsity> outSparsity;
};

class JacFunction : public Callback2 {
public:

    JacFunction() {
        inSparsity.push_back(Sparsity::dense(2 * L * dim));
        inSparsity.push_back(Sparsity::dense(0, 0));
        inSparsity.push_back(Sparsity::dense(L + 4));
        inSparsity.push_back(Sparsity::dense(1));
        inSparsity.push_back(Sparsity::dense(1));
        inSparsity.push_back(Sparsity::dense(1));

        outSparsity.push_back(Sparsity::dense(2 * L * dim, 2 * L * dim));
        outSparsity.push_back(Sparsity::dense(0, 0));
        outSparsity.push_back(Sparsity::dense(0, 0));
    }

    int nIn() {
        return DAE_NUM_IN + 2;
    }

    int nOut() {
        return DAE_NUM_OUT;
    }

    Sparsity inputSparsity(int i) {
        return inSparsity[i];
    }

    Sparsity outputSparsity(int i) {
        return outSparsity[i];
    }

    std::vector<DMatrix> operator()(const std::vector<DMatrix>& arg) {
        vector<double> fin = arg[0].nonzeros();
        vector<double> p = arg[2].nonzeros();
        double t = arg[3].toScalar();

        double Wi = p[L];
        double Wf = p[L + 1];
        double mu = p[L + 2];
        double tau = p[L + 3];

        double Wt, Wtp;
        if (t < tau) {
            Wt = Wi + (Wf - Wi) * t / tau;
            Wtp = (Wf - Wi) / tau;
        } else {
            Wt = Wf + (Wi - Wf) * (t - tau) / tau;
            Wtp = (Wi - Wf) / tau;
        }

        vector<double> J(L), Jp(L), dU(L);
        double U0 = UW(Wt);
        double U0p = UWp(Wt, Wtp);
        for (int i = 0; i < L; i++) {
            J[i] = JWij(Wt * p[i], Wt * p[mod(i + 1)]);
            Jp[i] = JWijp(Wt * p[i], Wt * p[mod(i + 1)], Wtp * p[i], Wtp * p[mod(i + 1)]);
            dU[i] = UW(Wt * p[i]) - U0;
        }

        vector<vector<double>> jac(2 * L * dim/*, vector<double>(2 * L * dim)*/);
        for (int i = 0; i < 2 * L * dim; i++) {
            vector<double> jaci(2 * L * dim);
            jacobian(jaci, i, fin, J, Jp, U0, U0p, dU, mu);
            jac[i] = jaci;
        }

        return vector<DMatrix>{DMatrix(jac), DMatrix(), DMatrix()};
    }

private:
    vector<Sparsity> inSparsity;
    vector<Sparsity> outSparsity;
};

boost::mutex progress_mutex;
boost::mutex inputs_mutex;
boost::mutex problem_mutex;

boost::random::mt19937 rng;
boost::random::uniform_real_distribution<> uni(-1, 1);

void threadfunc(std::string prog, double tauf, queue<input>& inputs, vector<results>& res, progress_display& progress, int thread, managed_shared_memory& segment, string shm_name) {

    void_allocator void_alloc(segment.get_segment_manager());

    std::string tau_name = "tau" + lexical_cast<std::string>(thread);
    worker_tau* tau = segment.construct<worker_tau>(tau_name.c_str())();

    std::string output_name = "output" + lexical_cast<std::string>(thread);
    worker_output* output = segment.construct<worker_output>(output_name.c_str())(void_alloc);

    std::vector<std::string> args;
    args.push_back(prog);
    args.push_back("-1");
    args.push_back(shm_name);
    args.push_back(tau_name);
    args.push_back(output_name);

    execute(set_args(args));

    for (;;) {
        input in;
        {
            boost::mutex::scoped_lock lock(inputs_mutex);
            if (inputs.empty()) {
                interprocess::scoped_lock<interprocess_mutex> lock(tau->mutex);

                if (tau->full) {
                    tau->cond_full.wait(lock);
                }
                tau->tau = -1;
                tau->cond_empty.notify_one();
                tau->full = true;
                break;
            }
            in = inputs.front();
            inputs.pop();
        }

        {
            interprocess::scoped_lock<interprocess_mutex> lock(tau->mutex);

            if (tau->full) {
                tau->cond_full.wait(lock);
            }
            tau->tau = in.tau;
            tau->cond_empty.notify_one();
            tau->full = true;
        }

        //        results pointRes(void_alloc);
        results pointRes;
        pointRes.tau = in.tau;

        {
            interprocess::scoped_lock<interprocess_mutex> lock(output->mutex);

            if (!output->full) {
                output->cond_empty.wait(lock);
            }
            pointRes.Ei = output->Ei;
            pointRes.Ef = output->Ef;
            pointRes.Q = output->Q;
            pointRes.p = output->p;
            pointRes.ts = vector<double>(output->ts.begin(), output->ts.end());
            pointRes.Es = vector<double>(output->Es.begin(), output->Es.end());
            pointRes.b0 = vector<complex<double>>(output->b0.begin(), output->b0.end());
            pointRes.bf = vector<complex<double>>(output->bf.begin(), output->bf.end());
            vector<vector<complex<double>>> ff;
            for (int i = 0; i < L; i++) {
                ff.push_back(vector<complex<double>>(output->ff[i].begin(), output->ff[i].end()));
            }
            pointRes.ff = ff;
            vector<vector < vector<complex<double>>>> fs;
            for (int i = 0; i < output->fs.size(); i++) {
                vector<vector<complex<double>>> fsi;
                for (int j = 0; j < L; j++) {
                    fsi.push_back(vector<complex<double>>(output->fs[i][j].begin(), output->fs[i][j].end()));
                }
                fs.push_back(fsi);
            }
            pointRes.fs = fs;
            pointRes.runtime = string(output->runtime.begin(), output->runtime.end());
            //            pointRes.Es = output->Es;
            //            pointRes.b0 = output->b0;
            //            pointRes.bf = output->bf;
            //            pointRes.f0 = output->f0;
            //            pointRes.ff = output->ff;
            //            pointRes.runtime = output->runtime;
            output->full = false;
            output->cond_full.notify_one();
        }

        {
            boost::mutex::scoped_lock lock(inputs_mutex);
            res.push_back(pointRes);
        }

        {
            boost::mutex::scoped_lock lock(progress_mutex);
            ++progress;
        }
    }

    segment.destroy_ptr<worker_output>(output);
}

double energymin(const vector<double>& x, vector<double>& grad, void* data) {
    Function& func = *(Function*) data;
    vector<DMatrix> res = func(vector<DMatrix>{x, DMatrix()});
    grad = res[0].nonzeros();
    return res[1].toScalar();
}

worker_input* initialize(double Wi, double Wf, double mu, vector<double>& xi, managed_shared_memory& segment) {

    SX f = SX::sym("f", 2 * L * dim);
    SX dU = SX::sym("dU", L);
    SX J = SX::sym("J", L);
    SX U0 = SX::sym("U0");

    U0 = UW(Wi);
    for (int i = 0; i < L; i++) {
        J[i] = JWij(Wi * xi[i], Wi * xi[mod(i + 1)]);
        dU[i] = UW(Wi * xi[i]) - U0;
    }

    SX E = energy(f, J, U0, dU, mu).real();
    SX Enorm = energynorm(f, J, U0, dU, mu).real();

    SX g = SX::sym("g", L);
    for (int i = 0; i < L; i++) {
        g[i] = 0;
        for (int n = 0; n < dim; n++) {
            g[i] += f[2 * (i * dim + n)] * f[2 * (i * dim + n)] + f[2 * (i * dim + n) + 1] * f[2 * (i * dim + n) + 1];
        }
    }

    SXFunction nlp("nlp", nlpIn("x", f), nlpOut("f", E, "g", g));
    NlpSolver solver("solver", "ipopt", nlp, make_dict("hessian_approximation", "limited-memory", "linear_solver", "ma86", "print_level", 0, "print_time", false, "obj_scaling_factor", 1));
    SXFunction nlpnorm("nlpnorm", nlpIn("x", f), nlpOut("f", Enorm));
    NlpSolver solvernorm("solvernorm", "ipopt", nlp, make_dict("hessian_approximation", "limited-memory", "linear_solver", "ma86", "print_level", 0, "print_time", false, "obj_scaling_factor", 1));

    Function grad = nlpnorm.gradient();

    boost::random::mt19937 rng;
    boost::random::uniform_real_distribution<> uni(-1, 1);

    vector<double> xrand(2 * L*dim, 1);
    rng.seed();
    for (int i = 0; i < 2 * L * dim; i++) {
        xrand[i] = uni(rng);
    }

    map<string, DMatrix> arg;
    arg["lbx"] = -1;
    arg["ubx"] = 1;
    arg["x0"] = xrand;
    arg["lbg"] = 1;
    arg["ubg"] = 1;

    map<string, DMatrix> res = solver(arg);
    vector<double> x0ipopt = res["x"].nonzeros();
    double E0ipopt = nlpnorm(vector<DMatrix>{x0ipopt, DMatrix()})[0].toScalar(); //res["f"].toScalar();
    cout << "E0ipopt = " << E0ipopt << endl;
    //        vector<double> x0 = xrand;
    //        cout << "xrand = " << ::math(xrand) << endl;
    //        cout << "x0 = " << ::math(x0) << endl;
    //    cout << "E0 = " << ::math(res["f"].toScalar()) << endl;

    opt lopt(LD_LBFGS, 2 * L * dim);
    lopt.set_lower_bounds(-1);
    lopt.set_upper_bounds(1);
    lopt.set_min_objective(energymin, &grad);
    double E0nlopt;
    vector<double> x0nlopt = xrand;
    lopt.optimize(x0nlopt, E0nlopt);
    cout << "Enlopt = " << E0nlopt << endl;

        opt gopt(GD_MLSL, 2*L*dim);
        gopt.set_lower_bounds(-1);
        gopt.set_upper_bounds(1);
        gopt.set_maxtime(600);
        gopt.set_min_objective(energymin, &grad);
        gopt.set_local_optimizer(lopt);
        gopt.set_population(100);
        double E0nlopt2;
        vector<double> x0nlopt2 = xrand;
        gopt.optimize(x0nlopt2, E0nlopt2);
        cout << "Enlopt2(1) = " << E0nlopt2 << endl;
        lopt.optimize(x0nlopt2, E0nlopt2);
        cout << "Enlopt2(2) = " << E0nlopt2 << endl;
        
        exit(0);

    vector<double> x0;
    if (E0ipopt < E0nlopt) {
        x0 = x0ipopt;
    } else {
        x0 = x0nlopt;
    }

    vector<complex<double>> x0i(dim);
    for (int i = 0; i < L; i++) {
        for (int n = 0; n <= nmax; n++) {
            x0i[n] = complex<double>(x0[2 * (i * dim + n)], x0[2 * (i * dim + n) + 1]);
        }
        double nrm = sqrt(abs(dot(x0i, x0i)));
        for (int n = 0; n <= nmax; n++) {
            x0[2 * (i * dim + n)] /= nrm;
            x0[2 * (i * dim + n) + 1] /= nrm;
        }
    }
    //    cout << "nlp: " << ::math(nlp(vector<DMatrix>{x0, vector<double>()})[0].toScalar()) << endl;

    void_allocator void_alloc(segment.get_segment_manager());
    worker_input* input = segment.construct<worker_input>("input")(void_alloc);
    input->U0 = UW(Wi);
    for (int i = 0; i < L; i++) {
        input->J0.push_back(JWij(Wi * xi[i], Wi * xi[mod(i + 1)]));
    }
    for (int i = 0; i < 2 * L * dim; i++) {
        input->x0.push_back(x0[i]);
    }
    for (int i = 0; i < L; i++) {
        complex_vector f0i(dim, void_alloc);
        for (int n = 0; n <= nmax; n++) {
            f0i[n] = complex<double>(x0[2 * (i * dim + n)], x0[2 * (i * dim + n) + 1]);
        }
        input->f0.push_back(f0i);
    }
    input->Wi = Wi;
    input->Wf = Wf;
    input->mu = mu;
    for (int i = 0; i < L; i++) {
        input->xi.push_back(xi[i]);
    }

    return input;
}

complex<double> dot(complex_vector& v, complex_vector& w) {
    complex<double> res = 0;
    for (int i = 0; i < v.size(); i++) {
        res += ~v[i] * w[i];
    }
    return res;
}

string escape(string str) {
    return replace_all_copy(str, "\"", "\\\"");
}

void evolve(SXFunction& E0, SXFunction& Et, Function& ode_func, Function& jac_func, vector<double>& p, worker_input* input, worker_output* output, managed_shared_memory& segment) {
    double tau = p[L + 3];
    //    double tauf = tau;//2e-6;
    //    double dt = 0.9e-9*scale;
    double dt = input->dt;
    Integrator integrator_rk("integrator", "rk", ode_func, make_dict("t0", 0, "tf", 2 * tau, "number_of_finite_elements", ceil((2 * tau) / dt)));
    Integrator integrator_cvodes("integrator", "cvodes", ode_func, make_dict("t0", 0, "tf", 2 * tau, "exact_jacobian", false, "max_num_steps", 100000));
    Integrator integrator_cvodesjac("integrator", "cvodesjac", ode_func, make_dict("t0", 0, "tf", 2 * tau, "exact_jacobian", true, "full_jacobian", jac_func, "max_num_steps", 100000));
    Integrator integrator_rk1("integrator", "rk", ode_func, make_dict("t0", 0, "tf", tau, "number_of_finite_elements", ceil((tau) / dt)));
    Integrator integrator_cvodes1("integrator", "cvodes", ode_func, make_dict("t0", 0, "tf", tau, "exact_jacobian", false, "max_num_steps", 100000));
    Integrator integrator_cvodesjac1("integrator", "cvodesjac", ode_func, make_dict("t0", 0, "tf", tau, "exact_jacobian", /*false*/true, "full_jacobian", jac_func, "max_num_steps", 100000));
    Integrator integrator_rk2("integrator", "rk", ode_func, make_dict("t0", tau, "tf", 2 * tau, "number_of_finite_elements", ceil((tau) / dt)));
    Integrator integrator_cvodes2("integrator", "cvodes", ode_func, make_dict("t0", tau, "tf", 2 * tau, "exact_jacobian", false, "max_num_steps", 100000));
    Integrator integrator_cvodesjac2("integrator", "cvodesjac", ode_func, make_dict("t0", tau, "tf", 2 * tau, "exact_jacobian", /*false*/true, "full_jacobian", jac_func, "max_num_steps", 100000));
    Integrator integrator;
    Integrator integrator1;
    Integrator integrator2;
    if (input->integrator == "rk") {
        integrator = integrator_rk;
        integrator1 = integrator_rk1;
        integrator2 = integrator_rk2;
    }
    if (input->integrator == "cvodes") {
        integrator = integrator_cvodes;
        integrator1 = integrator_cvodes1;
        integrator2 = integrator_cvodes2;
    }
    if (input->integrator == "cvodesjac") {
        integrator = integrator_cvodesjac;
        integrator1 = integrator_cvodesjac1;
        integrator2 = integrator_cvodesjac2;
    }
    ptime start_time = microsec_clock::local_time();

    std::vector<double> x0;
    for (int i = 0; i < 2 * L * dim; i++) {
        x0.push_back(input->x0[i]);
    }

    void_allocator void_alloc(segment.get_segment_manager());

    vector<double> grid1;
    vector<double> grid2;
    int ngrid = input->ngrid;
    double griddt = tau / (ngrid - 1);
    for (int i = 0; i < ngrid - 1; i++) {
        grid1.push_back(i * griddt);
        grid2.push_back(tau + i * griddt);
    }
    grid1.push_back(tau);
    grid2.push_back(2 * tau);

    vector<vector<double>> xs;
    vector<double> xf;
    bool half = true;
    output->ts.clear();
    output->Es.clear();
    output->fs.clear();
    if (half) {
        Simulator sim1("simulator1", integrator1, grid1);
        map<string, DMatrix> sres1 = sim1(make_map("x0", DMatrix(x0), "p", p));
        DMatrix xfs1 = sres1["xf"];
        for (int i = 0; i < ngrid; i++) {
            output->ts.push_back(grid1[i]);
            vector<double> xfi = xfs1[Slice(2 * i * L*dim, 2 * (i + 1) * L * dim)].nonzeros();
            xs.push_back(xfi);
            output->Es.push_back(E0(vector<DMatrix>{xfi})[0].toScalar());
            complex_vector_vector fsi(void_alloc);
            for (int i = 0; i < L; i++) {
                complex_vector fsii(void_alloc);
                for (int n = 0; n <= nmax; n++) {
                    fsii.push_back(complex<double>(xfi[2 * (i * dim + n)], xfi[2 * (i * dim + n) + 1]));
                }
                double nrm = sqrt(abs(dot(fsii, fsii)));
                for (int n = 0; n <= nmax; n++) {
                    fsii[n] /= nrm;
                }
                fsi.push_back(fsii);
            }
            output->fs.push_back(fsi);
        }
        Simulator sim2("simulator2", integrator2, grid2);
        map<string, DMatrix> sres2 = sim2(make_map("x0", DMatrix(xs.back()), "p", p));
        DMatrix xfs2 = sres2["xf"];
        for (int i = 0; i < ngrid; i++) {
            output->ts.push_back(grid2[i]);
            vector<double> xfi = xfs2[Slice(2 * i * L*dim, 2 * (i + 1) * L * dim)].nonzeros();
            xs.push_back(xfi);
            output->Es.push_back(E0(vector<DMatrix>{xfi})[0].toScalar());
            complex_vector_vector fsi(void_alloc);
            for (int i = 0; i < L; i++) {
                complex_vector fsii(void_alloc);
                for (int n = 0; n <= nmax; n++) {
                    fsii.push_back(complex<double>(xfi[2 * (i * dim + n)], xfi[2 * (i * dim + n) + 1]));
                }
                double nrm = sqrt(abs(dot(fsii, fsii)));
                for (int n = 0; n <= nmax; n++) {
                    fsii[n] /= nrm;
                }
                fsi.push_back(fsii);
            }
            output->fs.push_back(fsi);
            xf = xs.back();
        }
        //        map<string, DMatrix> res = integrator1(make_map("x0", DMatrix(x0), "p", p));
        //        xf = res["xf"].nonzeros();
        //        res = integrator2(make_map("x0", DMatrix(xf), "p", p));
        //        xf = res["xf"].nonzeros();
    } else {
        map<string, DMatrix> res = integrator(make_map("x0", DMatrix(x0), "p", p));
        /*vector<double>*/ xf = res["xf"].nonzeros();
    }

    complex_vector_vector ff(void_alloc);
    for (int i = 0; i < L; i++) {
        complex_vector ffi(void_alloc);
        for (int n = 0; n <= nmax; n++) {
            ffi.push_back(complex<double>(xf[2 * (i * dim + n)], xf[2 * (i * dim + n) + 1]));
        }
        double nrm = sqrt(abs(dot(ffi, ffi)));
        for (int n = 0; n <= nmax; n++) {
            ffi[n] /= nrm;
        }
        ff.push_back(ffi);
    }
    output->ff = ff;

    complex_vector_vector& f0 = input->f0;

    output->b0.clear();
    output->bf.clear();
    for (int i = 0; i < L; i++) {
        output->b0.push_back(b(f0, i, input->J0, input->U0));
        output->bf.push_back(b(ff, i, input->J0, input->U0));
    }

    output->Ei = E0(vector<DMatrix>{x0})[0].toScalar();
    output->Ef = E0(vector<DMatrix>{xf})[0].toScalar();
    output->Q = output->Ef - output->Ei;

    vector<double> pi(L);
    output->p = 0;
    for (int i = 0; i < L; i++) {
        pi[i] = 1 - norm(dot(ff[i], f0[i]));
        output->p += pi[i];
    }
    output->p /= L;

    ptime stop_time = microsec_clock::local_time();
    time_period period(start_time, stop_time);
    output->runtime = escape(to_simple_string(period.length())).c_str();
}

void fail(std::string error, worker_output* output, managed_shared_memory& segment) {
    void_allocator void_alloc(segment.get_segment_manager());
    output->Ei = numeric_limits<double>::quiet_NaN();
    output->Ef = numeric_limits<double>::quiet_NaN();
    output->Q = numeric_limits<double>::quiet_NaN();
    output->p = numeric_limits<double>::quiet_NaN();
    output->ts = double_vector(1, numeric_limits<double>::quiet_NaN(), void_alloc);
    output->Es = double_vector(1, numeric_limits<double>::quiet_NaN(), void_alloc);
    complex_vector nan_vector(L, numeric_limits<double>::quiet_NaN(), void_alloc);
    output->b0 = nan_vector;
    output->bf = nan_vector;
    complex_vector_vector nan_vector_vector(L, complex_vector(dim, numeric_limits<double>::quiet_NaN(), void_alloc), void_alloc);
    output->f0 = nan_vector_vector;
    output->ff = nan_vector_vector;
    complex_vector_vector_vector nan_vector_vector_vector(1, complex_vector_vector(L, complex_vector(dim, numeric_limits<double>::quiet_NaN(), void_alloc), void_alloc), void_alloc);
    output->fs = nan_vector_vector_vector;
    output->runtime = escape(error).c_str();
}

SXFunction get_ode() {

    SX p = SX::sym("p", L + 4);
    SX Wi = p[L];
    SX Wf = p[L + 1];
    SX mu = p[L + 2];
    SX tau = p[L + 3];

    SX f = SX::sym("f", 2 * L * dim);
    SX dU = SX::sym("dU", L);
    SX J = SX::sym("J", L);
    SX U0 = SX::sym("U0");
    SX t = SX::sym("t");

    SX Wt = if_else(t < tau, Wi + (Wf - Wi) * t / (tau), Wf + (Wi - Wf) * (t - tau) / (tau));

    U0 = UW(Wt);
    for (int i = 0; i < L; i++) {
        J[i] = JWij(Wt * p[i], Wt * p[mod(i + 1)]);
        dU[i] = UW(Wt * p[i]) - U0;
    }

    complex<SX> E = energy(f, J, U0, dU, mu);

    complex<SX> S = canonical(f, J, U0, dU, mu);
    SXFunction St("St",{t},
    {
        S.real(), S.imag()
    });
    complex<SX> Sdt = complex<SX>(St.gradient(0, 0)(vector<SX>{t})[0], St.gradient(0, 1)(vector<SX>{t})[0]);

    complex<SX> HS = Sdt - complex<SX>(0, 1) * E;
    SXFunction HSf("HS",{f},
    {
        HS.real(), HS.imag()
    });
    complex<SX> HSdf = complex<SX>(HSf.gradient(0, 0)(vector<SX>{f})[0], HSf.gradient(0, 1)(vector<SX>{f})[0]);
    SX ode = SX::sym("ode", 2 * L * dim);
    for (int j = 0; j < L * dim; j++) {
        ode[2 * j] = 0.5 * (HSdf.real().elem(2 * j) - HSdf.imag().elem(2 * j + 1));
        ode[2 * j + 1] = 0.5 * (HSdf.real().elem(2 * j + 1) + HSdf.imag().elem(2 * j));
    }

    SXFunction ode_func = SXFunction("ode", daeIn("t", t, "x", f, "p", p), daeOut("ode", ode));

    return ode_func;
}

void worker(worker_input* input, worker_tau* tau_in, worker_output* output, managed_shared_memory& segment) {//std::string tau_name, std::string output_name, managed_shared_memory& segment) {

    double Wi = input->Wi;
    double Wf = input->Wf;
    double mu = input->mu;
    double_vector xi = input->xi;

    vector<double> p(L + 4);
    for (int i = 0; i < L; i++) {
        p[i] = xi[i];
    }
    p[L] = Wi;
    p[L + 1] = Wf;
    p[L + 2] = mu;

    SX f = SX::sym("f", 2 * L * dim);
    SX dU = SX::sym("dU", L);
    SX J = SX::sym("J", L);
    SX U0 = SX::sym("U0");
    SX t = SX::sym("t");

    //    SX tau = SX::sym("tau");
    //    SX tau = p[L+3];
    SX psx = SX::sym("p", L + 4);
    SX tau = psx[L + 3];

    SX Wt = if_else(t < tau, Wi + (Wf - Wi) * t / (tau), Wf + (Wi - Wf) * (t - tau) / (tau));

    U0 = UW(Wt);
    for (int i = 0; i < L; i++) {
        J[i] = JWij(Wt * xi[i], Wt * xi[mod(i + 1)]);
        dU[i] = UW(Wt * xi[i]) - U0;
    }

    SX E = energy(f, J, U0, dU, mu).real();
    SXFunction Ef = SXFunction("E",{f, t, tau},
    {
        E
    });
    SXFunction E0 = SXFunction("E0",{f}, Ef(vector<SX>{f, 0, 1}));

        ExternalFunction ode_func("ode");
        Function jac_func;

//    Function ode_func, jac_func;
//
//    ODEFunction odef;
//    JacFunction jacf;
//    chdir("odes");
//    vector<Function> odes;
//    odes.push_back(ExternalFunction("ode_S"));
//    for (int ei = 0; ei < 7; ei++) {
//        for (int i = 0; i < L; i++) {
//            for (int n = 0; n <= nmax; n++) {
//                string funcname = "ode_E_" + to_string(ei) + "_" + to_string(i) + "_" + to_string(n);
//                odes.push_back(ExternalFunction(funcname));
//            }
//        }
//    }
//    SumFunction sf(odes);
//    chdir("..");
//    if (input->integrator == "cvodesjac") {
//        ode_func = odef.create();
//        jac_func = jacf.create();
//    } else {
//        ode_func = sf.create();
//    }

    //    SXFunction ode_func = get_ode();

    //    ODEFunction odef;
    //    Function ode_func = odef.create();
    //    JacFunction jacf;
    //    Function jac_func = jacf.create();

    double taui;
    for (;;) {
        {
            interprocess::scoped_lock<interprocess_mutex> lock(tau_in->mutex);
            if (!tau_in->full) {
                tau_in->cond_empty.wait(lock);
            }
            taui = tau_in->tau;
            tau_in->full = false;
            tau_in->cond_full.notify_one();
            if (taui < 0) {
                break;
            }
        }

        p[L + 3] = taui;
        try {
            evolve(E0, Ef, ode_func, jac_func, p, input, output, segment);
        } catch (std::exception& e) {
            fail(e.what(), output, segment);
        }

        {
            interprocess::scoped_lock<interprocess_mutex> lock(output->mutex);

            if (output->full) {
                output->cond_full.wait(lock);
            }
            output->cond_empty.notify_one();
            output->full = true;
        }
    }

    segment.destroy_ptr<worker_tau>(tau_in);
}

void build_ode() {

    SX p = SX::sym("p", L + 4);
    SX Wi = p[L];
    SX Wf = p[L + 1];
    SX mu = p[L + 2];
    SX tau = p[L + 3];

    SX f = SX::sym("f", 2 * L * dim);
    SX dU = SX::sym("dU", L);
    SX J = SX::sym("J", L);
    SX U0 = SX::sym("U0");
    SX t = SX::sym("t");

    SX Wt = if_else(t < tau, Wi + (Wf - Wi) * t / (tau), Wf + (Wi - Wf) * (t - tau) / (tau));

    U0 = UW(Wt);
    for (int i = 0; i < L; i++) {
        J[i] = JWij(Wt * p[i], Wt * p[mod(i + 1)]);
        dU[i] = UW(Wt * p[i]) - U0;
    }

    complex<SX> E = energy(f, J, U0, dU, mu);

    complex<SX> S = canonical(f, J, U0, dU, mu);
    SXFunction St("St",{t},
    {
        S.real(), S.imag()
    });
    complex<SX> Sdt = complex<SX>(St.gradient(0, 0)(vector<SX>{t})[0], St.gradient(0, 1)(vector<SX>{t})[0]);

    complex<SX> HS = Sdt - complex<SX>(0, 1) * E;
    SXFunction HSf("HS",{f},
    {
        HS.real(), HS.imag()
    });
    complex<SX> HSdf = complex<SX>(HSf.gradient(0, 0)(vector<SX>{f})[0], HSf.gradient(0, 1)(vector<SX>{f})[0]);
    SX ode = SX::sym("ode", 2 * L * dim);
    for (int j = 0; j < L * dim; j++) {
        ode[2 * j] = 0.5 * (HSdf.real().elem(2 * j) - HSdf.imag().elem(2 * j + 1));
        ode[2 * j + 1] = 0.5 * (HSdf.real().elem(2 * j + 1) + HSdf.imag().elem(2 * j));
    }

    SXFunction ode_func = SXFunction("ode", daeIn("t", t, "x", f, "p", p), daeOut("ode", ode));

    ode_func.generate("ode");
}

typedef complex<SX> (*energyfunc) (int i, int n, SX& fin, SX& J, SX& U0, SX& dU, SX& mu);
energyfunc energyfuncs[] = {energy1, energy2, energy3, energy4, energy5, energy6, energy7};

void build_odes() {

    SX p = SX::sym("p", L + 4);
    SX Wi = p[L];
    SX Wf = p[L + 1];
    SX mu = p[L + 2];
    SX tau = p[L + 3];

    SX f = SX::sym("f", 2 * L * dim);
    SX dU = SX::sym("dU", L);
    SX J = SX::sym("J", L);
    SX U0 = SX::sym("U0");
    SX t = SX::sym("t");

    SX Wt = if_else(t < tau, Wi + (Wf - Wi) * t / (tau), Wf + (Wi - Wf) * (t - tau) / (tau));

    U0 = UW(Wt);
    for (int i = 0; i < L; i++) {
        J[i] = JWij(Wt * p[i], Wt * p[mod(i + 1)]);
        dU[i] = UW(Wt * p[i]) - U0;
    }

    chdir("odes");
    for (int ei = 0; ei < 7; ei++) {
        energyfunc energy = energyfuncs[ei];
        for (int i = 0; i < L; i++) {
            for (int n = 0; n <= nmax; n++) {
                string func_name = "ode_E_" + to_string(ei) + "_" + to_string(i) + "_" + to_string(n);

                complex<SX> E = energy(i, n, f, J, U0, dU, mu);

                complex<SX> HS = -complex<SX>(0, 1) * E;
                SXFunction HSf("HS",{f},
                {
                    HS.real(), HS.imag()
                });
                complex<SX> HSdf = complex<SX>(HSf.gradient(0, 0)(vector<SX>{f})[0], HSf.gradient(0, 1)(vector<SX>{f})[0]);
                SX ode = SX::sym("ode", 2 * L * dim);
                for (int j = 0; j < L * dim; j++) {
                    ode[2 * j] = 0.5 * (HSdf.real().elem(2 * j) - HSdf.imag().elem(2 * j + 1));
                    ode[2 * j + 1] = 0.5 * (HSdf.real().elem(2 * j + 1) + HSdf.imag().elem(2 * j));
                }
                SXFunction ode_func = SXFunction(func_name, daeIn("t", t, "x", f, "p", p), daeOut("ode", ode));
                //                ode_func.generate(funcname);

                CodeGenerator gen;
                gen.add(ode_func);
                //                gen.add(ode_func.fullJacobian());
                gen.generate(func_name);
            }
        }
    }

    complex<SX> S = canonical(f, J, U0, dU, mu);
    SXFunction St("St",{t},
    {
        S.real(), S.imag()
    });
    complex<SX> Sdt = complex<SX>(St.gradient(0, 0)(vector<SX>{t})[0], St.gradient(0, 1)(vector<SX>{t})[0]);


    complex<SX> HS = Sdt;
    SXFunction HSf("HS",{f},
    {
        HS.real(), HS.imag()
    });
    complex<SX> HSdf = complex<SX>(HSf.gradient(0, 0)(vector<SX>{f})[0], HSf.gradient(0, 1)(vector<SX>{f})[0]); //HSf.gradient()(vector<SX>{f})[0];
    SX ode = SX::sym("ode", 2 * L * dim);
    for (int j = 0; j < L * dim; j++) {
        ode[2 * j] = 0.5 * (HSdf.real().elem(2 * j) - HSdf.imag().elem(2 * j + 1));
        ode[2 * j + 1] = 0.5 * (HSdf.real().elem(2 * j + 1) + HSdf.imag().elem(2 * j));
    }
    SXFunction ode_func = SXFunction("ode_S", daeIn("t", t, "x", f, "p", p), daeOut("ode", ode));
    //    ode_func.generate("odes");

    CodeGenerator gen;
    gen.add(ode_func);
    //    gen.add(ode_func.fullJacobian());
    gen.generate("ode_S");

    chdir("..");
}

template<class T>
bool paircomp(const pair<double, T>& a, const pair<double, T>& b) {
    return a.first < b.first;
}

/*
 * 
 */
int main(int argc, char** argv) {

    //    vector<double> fin(2*L*dim, 0.25);
    //    vector<double> p({1.01015958087519, 0.914144976064563, 1.04162956443615, 1.0679898084607, 0.958180948719382, 3e11, 1e11, 1.8974531961544957e7, 1e-8});
    //    vector<DMatrix> arg({fin, DMatrix(), p, DMatrix()});
    //
    //    ODEFunction odef;
    //    Function odefun = odef.create();
    ////    cout << odefun(arg) << endl;
    //    
    //    JacFunction jacfn;
    //    Function jacfun = jacfn.create();
    ////    cout << jacfun(arg) << endl;
    //    return 0;
    //    
    //    vector<double> Jf({1.0227310362932056e7,1.024675659410313e7,1.0353495227480633e7,1.0297352485677032e7,
    //   1.0262727637339968e7});
    //   vector<double> dUf({-662648.4069630802,6.324989883779682e6,-2.614157014239885e6,-4.138882871558778e6,
    //   2.9101352455864474e6});
    //   double U0f = 3.7949063923089914e7;
    //   double muf = 1.8974531961544957e7;
    //   vector<double> Jpf({-9.5757307865558e13,-9.350483272606312e13,-8.11329513149425e13,
    //   -8.767128019117625e13,-9.170625884987675e13});
    //   double U0pf = 4.40274854172645e15;
    //   vector<vector<double>> jacf(2*L*dim, vector<double>(2*L*dim, 0));
    //   for (int i = 0; i < 2*L*dim; i++) {
    ////           cout << ode(i, fin, Jf, Jpf, U0f, U0pf, dUf, muf) << endl;
    //           jacobian(jacf[i], i, fin, Jf, Jpf, U0f, U0pf, dUf, muf);
    //           for (int j = 0; j < 2*L*dim; j++) {
    //               if (jacf[i][j] != 0) {
    //                   cout << "(" << i+1 << "," << j+1 << ") -> " << jacf[i][j] << endl;
    //               }
    //           }
    //   }
    //    return 0;

    //    Function ode_func = get_ode();
    //    CodeGenerator gen;
    //    gen.add(ode_func);
    //    Function jac = ode_func.fullJacobian();
    //    gen.add(ode_func.fullJacobian());
    //    gen.generate("ode");
    //    return 0;

    //    build_ode();
    //    return 0;

    //        build_ode();
    //        return 0;

    ptime begin = microsec_clock::local_time();

    random::mt19937 rng;
    random::uniform_real_distribution<> uni(-1, 1);

    int seed = lexical_cast<int>(argv[1]);

    if (seed != -1) {

        double Wi = lexical_cast<double>(argv[2]);
        double Wf = lexical_cast<double>(argv[3]);

        double mu = lexical_cast<double>(argv[4]);

        double Ui = UWi(Wi);

        double D = lexical_cast<double>(argv[5]);

        double taui = lexical_cast<double>(argv[6]);
        double tauf = lexical_cast<double>(argv[7]);
        int ntaus = lexical_cast<int>(argv[8]);

        int numthreads = lexical_cast<int>(argv[9]);

        int resi = lexical_cast<int>(argv[10]);

        //        int integrator = lexical_cast<int>(argv[11]);
        std::string intg = argv[11];

        double dt = lexical_cast<double>(argv[12]);

        int ngrid = lexical_cast<int>(argv[13]);

#ifdef AMAZON
        //    path resdir("/home/ubuntu/Results/Canonical Transformation Dynamical Gutzwiller");
        path resdir("/home/ubuntu/Dropbox/Amazon EC2/Simulation Results/CTDG");
#else
        path resdir("/Users/Abuenameh/Documents/Simulation Results/Canonical Transformation Dynamical Gutzwiller 2");
        //        path resdir("/Users/Abuenameh/Documents/Simulation Results/Dynamical Gutzwiller Hartmann Comparison");
#endif
        if (!exists(resdir)) {
            cerr << "Results directory " << resdir << " does not exist!" << endl;
            exit(1);
        }
        ostringstream oss;
        oss << "res." << resi << ".txt";
        path resfile = resdir / oss.str();
        //#ifndef AMAZON
        while (exists(resfile)) {
            resi++;
            oss.str("");
            oss << "res." << resi << ".txt";
            resfile = resdir / oss.str();
        }
        //#endif
        if (seed < 0) {
            resi = seed;
            oss.str("");
            oss << "res." << resi << ".txt";
            resfile = resdir / oss.str();
        }
        vector<double> xi(L, 1);
        rng.seed(seed);
        if (seed > -1) {
            for (int j = 0; j < L; j++) {
                xi[j] = (1 + D * uni(rng));
            }
        }

        //        double Ui = UWi(Wi);
        double mui = mu * Ui;

        filesystem::ofstream os(resfile);
        printMath(os, "int", resi, intg);
        printMath(os, "seed", resi, seed);
        printMath(os, "Delta", resi, D);
        printMath(os, "dt", resi, dt);
        printMath(os, "mures", resi, mui);
        printMath(os, "Ures", resi, Ui);
        printMath(os, "xires", resi, xi);
        os << flush;

        printMath(os, "Wires", resi, Wi);
        printMath(os, "Wfres", resi, Wf);
        os << flush;

        cout << "Res: " << resi << endl;

        string shm_name = "SharedMemory" + to_string(time(NULL));

        struct shm_remove {
            string shm_name;

            shm_remove(string shm_name_) : shm_name(shm_name_) {
                shared_memory_object::remove(shm_name.c_str()); //("SharedMemory");
            }

            ~shm_remove() {
                shared_memory_object::remove(shm_name.c_str()); //("SharedMemory");
            }
        } remover(shm_name);

        int size = 2000 * (sizeof (worker_input) + numthreads * (sizeof (worker_tau) + sizeof (worker_output))); //2 * (((2 * L * dim + L + 1) + numthreads * (4 * L * dim + 5 * L + 6)) * sizeof (double) +numthreads * 2 * sizeof (ptime)/*sizeof(time_period)*/);
//        int size = 1000 * (sizeof (worker_input) + numthreads * (sizeof (worker_tau) + sizeof (worker_output))); //2 * (((2 * L * dim + L + 1) + numthreads * (4 * L * dim + 5 * L + 6)) * sizeof (double) +numthreads * 2 * sizeof (ptime)/*sizeof(time_period)*/);

        managed_shared_memory segment(create_only, shm_name.c_str(), size);
        //        managed_shared_memory segment(create_only, "SharedMemory", size);

        worker_input* w_input = initialize(Wi, Wf, mui, xi, segment);
        //        return 0;

        void_allocator void_alloc(segment.get_segment_manager());
        char_string integrator(intg.begin(), intg.end(), void_alloc);
        w_input->integrator = integrator;
        w_input->dt = dt;
        w_input->ngrid = ngrid;

        queue<input> inputs;
        if (ntaus == 1) {
            input in;
            in.tau = taui;
            inputs.push(in);
        } else {
            for (int i = 0; i < ntaus; i++) {
                input in;
                double tau = taui + i * (tauf - taui) / (ntaus - 1);
                in.tau = tau;
                inputs.push(in);
            }
        }

        vector<results> res;

        progress_display progress(inputs.size());

        thread_group threads;
        for (int i = 0; i < numthreads; i++) {
            threads.create_thread(bind(&threadfunc, argv[0], tauf, boost::ref(inputs), boost::ref(res), boost::ref(progress), i, boost::ref(segment), shm_name));
        }
        threads.join_all();

        vector<double> taures;
        vector<vector<double>> tsres;
        vector<vector<double>> Esres;
        vector<double> Eires;
        vector<double> Efres;
        vector<double> Qres;
        vector<double> pres;
        vector<vector<complex<double>>> b0res;
        vector<vector<complex<double>>> bfres;
        vector<vector < vector<complex<double>>>> ffres;
        vector<vector<vector < vector<complex<double>>>>> fsres;
        vector<std::string> runtimeres;

        vector<pair<double, double>> tauresp;
        vector<pair<double, vector<double>>> tsresp;
        vector<pair<double, vector<double>>> Esresp;
        vector<pair<double, double>> Eiresp;
        vector<pair<double, double>> Efresp;
        vector<pair<double, double>> Qresp;
        vector<pair<double, double>> presp;
        vector<pair<double, vector<complex<double>>>> b0resp;
        vector<pair<double, vector<complex<double>>>> bfresp;
        vector<pair<double, vector < vector<complex<double>>>>> ffresp;
        vector<pair<double, vector<vector < vector<complex<double>>>>>> fsresp;
        vector<pair<double, std::string>> runtimeresp;

        for (results& ires : res) {
            tauresp.push_back(make_pair(ires.tau, ires.tau));
            tsresp.push_back(make_pair(ires.tau, ires.ts));
            Esresp.push_back(make_pair(ires.tau, ires.Es));
            Eiresp.push_back(make_pair(ires.tau, ires.Ei));
            Efresp.push_back(make_pair(ires.tau, ires.Ef));
            Qresp.push_back(make_pair(ires.tau, ires.Q));
            presp.push_back(make_pair(ires.tau, ires.p));
            b0resp.push_back(make_pair(ires.tau, ires.b0));
            bfresp.push_back(make_pair(ires.tau, ires.bf));
            runtimeresp.push_back(make_pair(ires.tau, ires.runtime));
            ffresp.push_back(make_pair(ires.tau, ires.ff));
            fsresp.push_back(make_pair(ires.tau, ires.fs));
        }

        sort(tauresp.begin(), tauresp.end(), paircomp<double>);
        sort(tsresp.begin(), tsresp.end(), paircomp<vector<double>>);
        sort(Esresp.begin(), Esresp.end(), paircomp<vector<double>>);
        sort(Eiresp.begin(), Eiresp.end(), paircomp<double>);
        sort(Efresp.begin(), Efresp.end(), paircomp<double>);
        sort(Qresp.begin(), Qresp.end(), paircomp<double>);
        sort(presp.begin(), presp.end(), paircomp<double>);
        sort(b0resp.begin(), b0resp.end(), paircomp<vector<complex<double>>>);
        sort(bfresp.begin(), bfresp.end(), paircomp<vector<complex<double>>>);
        sort(ffresp.begin(), ffresp.end(), paircomp<vector < vector<complex<double>>>>);
        sort(runtimeresp.begin(), runtimeresp.end(), paircomp<string>);

        for (int i = 0; i < tauresp.size(); i++) {
            taures.push_back(tauresp[i].second);
            tsres.push_back(tsresp[i].second);
            Esres.push_back(Esresp[i].second);
            Eires.push_back(Eiresp[i].second);
            Efres.push_back(Efresp[i].second);
            Qres.push_back(Qresp[i].second);
            pres.push_back(presp[i].second);
            b0res.push_back(b0resp[i].second);
            bfres.push_back(bfresp[i].second);
            ffres.push_back(ffresp[i].second);
            fsres.push_back(fsresp[i].second);
            runtimeres.push_back(runtimeresp[i].second);
            //            taures.push_back(ires.tau);
            //            Eires.push_back(ires.Ei);
            //            Efres.push_back(ires.Ef);
            //            Qres.push_back(ires.Q);
            //            pres.push_back(ires.p);
            ////            Esres.push_back(ires.Es);
            //            b0res.push_back(ires.b0);
            //            bfres.push_back(ires.bf);
            //            runtimeres.push_back(replace_all_copy(ires.runtime, "\"", "\\\""));
            //            //            vector<double> Es(ires.Es.begin(), ires.Es.end());
            //            //            Esres.push_back(Es);
            //            //            vector<complex<double>> b0(ires.b0.begin(), ires.b0.end());
            //            //            vector<complex<double>> bf(ires.bf.begin(), ires.bf.end());
            //            //            b0res.push_back(b0);
            //            //            bfres.push_back(bf);
            //            //            f0res.push_back(ires.f0);
            //            ffres.push_back(ires.ff);
            //            //            std::string runtime(ires.runtime.begin(), ires.runtime.end());
            //            //            runtimeres.push_back(replace_all_copy(runtime, "\"", "\\\""));
        }

        printMath(os, "taures", resi, taures);
        printMath(os, "tsres", resi, tsres);
        printMath(os, "Esres", resi, Esres);
        printMath(os, "Eires", resi, Eires);
        printMath(os, "Efres", resi, Efres);
        printMath(os, "Qres", resi, Qres);
        printMath(os, "pres", resi, pres);
        printMath(os, "U0res", resi, w_input->U0);
        printMath(os, "J0res", resi, w_input->J0);
        printMath(os, "b0res", resi, b0res);
        printMath(os, "bfres", resi, bfres);
        printMath(os, "f0res", resi, w_input->f0);
        printMath(os, "ffres", resi, ffres);
        printMath(os, "fsres", resi, fsres);
        printMath(os, "runtime", resi, runtimeres);

        ptime end = microsec_clock::local_time();
        time_period period(begin, end);
        cout << endl << period.length() << endl << endl;

        os << "totalruntime[" << resi << "]=\"" << period.length() << "\";" << endl;

        segment.destroy<worker_input>("input");

    } else {
        //        managed_shared_memory segment(open_only, "SharedMemory");
        managed_shared_memory segment(open_only, argv[2]);

        worker_input* input = segment.find<worker_input>("input").first;
        worker_tau* tau = segment.find<worker_tau>(argv[3]).first;
        worker_output* output = segment.find<worker_output>(argv[4]).first;

        worker(input, tau, output, segment);

    }

    return 0;

}

