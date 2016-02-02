/* 
 * File:   orderparameter.hpp
 * Author: Abuenameh
 *
 * Created on November 4, 2015, 3:31 AM
 */

#ifndef ORDERPARAMETER_HPP
#define	ORDERPARAMETER_HPP

#include <complex>
#include <vector>

using namespace std;

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/vector.hpp>

using namespace boost;

#include "gutzwiller.hpp"

typedef interprocess::managed_shared_memory::segment_manager segment_manager_t;

typedef interprocess::allocator<double, segment_manager_t> double_allocator;
typedef interprocess::vector<double, double_allocator> double_vector;

typedef interprocess::allocator<complex<double>, segment_manager_t> complex_allocator;
typedef interprocess::vector<complex<double>, complex_allocator> complex_vector;
typedef interprocess::allocator<complex_vector, segment_manager_t> complex_vector_allocator;
typedef interprocess::vector<complex_vector, complex_vector_allocator> complex_vector_vector;

complex<double> b0(complex_vector_vector& f, int i);
complex<double> b1(complex_vector_vector& f, int i, double_vector& J, double U);
complex<double> bf1(complex_vector_vector& f, int k, int i, int j, int a, int b, int n, int m, int p, int q);
complex<double> bf2(complex_vector_vector& f, int k, int i, int j, int a, int b, int n, int m, int p, int q);
complex<double> bf3(complex_vector_vector& f, int k, int i, int j, int a, int b, int n, int m, int p, int q);
complex<double> bf4(complex_vector_vector& f, int k, int i, int j, int a, int b, int n, int m, int p, int q);
complex<double> bf5(complex_vector_vector& f, int k, int i, int j, int a, int b, int n, int m, int p, int q);
complex<double> bf6(complex_vector_vector& f, int k, int i, int j, int a, int b, int n, int m, int p, int q);
complex<double> bf7(complex_vector_vector& f, int k, int i, int j, int a, int b, int n, int m, int p, int q);
complex<double> bf8(complex_vector_vector& f, int k, int i, int j, int a, int b, int n, int m, int p, int q);
complex<double> bf(complex_vector_vector& f, int k, int i, int j, int a, int b, int n, int m, int p, int q);
complex<double> b2(complex_vector_vector& f, int k, double_vector& J, double U);
complex<double> b3(complex_vector_vector& f, int k, double_vector& J, double U);
complex<double> b(complex_vector_vector& f, int k, double_vector& J, double U);

#endif	/* ORDERPARAMETER_HPP */

