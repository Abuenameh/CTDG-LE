#include "casadimath.hpp"
#include "gutzwiller.hpp"

//namespace casadi {
//
//    inline bool isnan(SX& sx) {
//        return sx.at(0).isNan();
//    }
//
//    inline bool isinf(SX sx) {
//        return sx.at(0).isInf();
//    }
//}

complex<SX> operator*(double x, complex<SX> sx) {
    return complex<SX>(x, 0) * sx;
}

complex<SX> operator*(complex<SX> sx, double x) {
    return complex<SX>(x, 0) * sx;
}

complex<SX> operator*(complex<SX> csx, SX sx) {
    return csx * complex<SX>(sx, 0);
}

complex<SX> energy(int i, SX& fin, SX& J, SX& U0, SX& dU, SX& mu) {
    vector<vector<complex < SX>>> f(L, vector<complex < SX >> (dim, complex<SX>(0, 0)));
    for (int j = 0; j < L; j++) {
        for (int k = 0; k < dim; k++) {
            int l = 2 * (dim * j + k);
            f[j][k] = complex<SX>(fin[l], fin[l + 1]);
        }
    }

    complex<SX> E, Ei, Ej1, Ej2, Ej1j2, Ej1k1, Ej2k2;
    E = complex<SX>(0, 0);
    Ei = complex<SX>(0, 0);
    Ej1 = complex<SX>(0, 0);
    Ej2 = complex<SX>(0, 0);
    Ej1j2 = complex<SX>(0, 0);
    Ej1k1 = complex<SX>(0, 0);
    Ej2k2 = complex<SX>(0, 0);
    int k1 = mod(i - 2);
    int j1 = mod(i - 1);
    int j2 = mod(i + 1);
    int k2 = mod(i + 2);

#include "casadii.incl"

    E += Ei;
    E += Ej1;
    E += Ej2;
    E += Ej1j2;
    E += Ej1k1;
    E += Ej2k2;
    return E;
}


#include "casadimath.incl"

complex<SX> energy(SX& fin, SX& J, SX& U0, SX& dU, double mu) {
    vector<vector<complex < SX>>> f(L, vector<complex < SX >> (dim, complex<SX>(0, 0)));
    vector<SX> norm2(L, 0);
    for (int j = 0; j < L; j++) {
        for (int k = 0; k < dim; k++) {
            int l = 2 * (j * dim + k);
            f[j][k] = complex<SX>(fin[l], fin[l + 1]);
        }
        for (int m = 0; m <= nmax; m++) {
            norm2[j] += f[j][m].real() * f[j][m].real() + f[j][m].imag() * f[j][m].imag();
        }
    }

    double theta = 0;
    double costh = cos(theta);
    double sinth = sin(theta);
    double cos2th = cos(2 * theta);
    double sin2th = sin(2 * theta);

    complex<SX> E = complex<SX>(0, 0);

    complex<SX> Ei, Ej1, Ej2, Ej1j2, Ej1k1, Ej2k2;

#include "casadi.incl"

    return E; //E.real();
}

complex<SX> energy(SX& fin, SX& J, SX& U0, SX& dU, SX& mu) {
    vector<vector<complex < SX>>> f(L, vector<complex < SX >> (dim, complex<SX>(0, 0)));
    vector<SX> norm2(L, 0);
    for (int j = 0; j < L; j++) {
        for (int k = 0; k < dim; k++) {
            int l = 2 * (j * dim + k);
            f[j][k] = complex<SX>(fin[l], fin[l + 1]);
        }
        for (int m = 0; m <= nmax; m++) {
            norm2[j] += f[j][m].real() * f[j][m].real() + f[j][m].imag() * f[j][m].imag();
        }
    }

    double theta = 0;
    double costh = cos(theta);
    double sinth = sin(theta);
    double cos2th = cos(2 * theta);
    double sin2th = sin(2 * theta);

    complex<SX> E = complex<SX>(0, 0);

    complex<SX> Ei, Ej1, Ej2, Ej1j2, Ej1k1, Ej2k2;

#include "casadi.incl"

    return E; //E.real();
}

complex<SX> energynorm(SX& fin, SX& J, SX& U0, SX& dU, double mu) {
    vector<vector<complex < SX>>> f(L, vector<complex < SX >> (dim, complex<SX>(0, 0)));
    vector<SX> norm2(L, 0);
    for (int j = 0; j < L; j++) {
        for (int k = 0; k < dim; k++) {
            int l = 2 * (j * dim + k);
            f[j][k] = complex<SX>(fin[l], fin[l + 1]);
        }
        for (int m = 0; m <= nmax; m++) {
            norm2[j] += f[j][m].real() * f[j][m].real() + f[j][m].imag() * f[j][m].imag();
        }
    }

    double theta = 0;
    double costh = cos(theta);
    double sinth = sin(theta);
    double cos2th = cos(2 * theta);
    double sin2th = sin(2 * theta);

    complex<SX> E = complex<SX>(0, 0);

    complex<SX> Ei, Ej1, Ej2, Ej1j2, Ej1k1, Ej2k2;

#include "casadinorm.incl"

    return E; //E.real();
}

complex<SX> energynorm(SX& fin, SX& J, SX& U0, SX& dU, SX& mu) {
    vector<vector<complex < SX>>> f(L, vector<complex < SX >> (dim, complex<SX>(0, 0)));
    vector<SX> norm2(L, 0);
    for (int j = 0; j < L; j++) {
        for (int k = 0; k < dim; k++) {
            int l = 2 * (j * dim + k);
            f[j][k] = complex<SX>(fin[l], fin[l + 1]);
        }
        for (int m = 0; m <= nmax; m++) {
            norm2[j] += f[j][m].real() * f[j][m].real() + f[j][m].imag() * f[j][m].imag();
        }
    }

    double theta = 0;
    double costh = cos(theta);
    double sinth = sin(theta);
    double cos2th = cos(2 * theta);
    double sin2th = sin(2 * theta);

    complex<SX> E = complex<SX>(0, 0);

    complex<SX> Ei, Ej1, Ej2, Ej1j2, Ej1k1, Ej2k2;

#include "casadinorm.incl"

    return E; //E.real();
}

complex<SX> canonical(int i, int n, SX& fin, SX& J, SX& U0, SX& dU, SX mu) {

    vector<vector<complex < SX>>> f(L, vector<complex < SX >> (dim, complex<SX>(0, 0)));
    vector<SX> norm2(L, 0);
    for (int j = 0; j < L; j++) {
        for (int k = 0; k < dim; k++) {
            int l = 2 * (j * dim + k);
            f[j][k] = complex<SX>(fin[l], fin[l + 1]);
        }
        for (int m = 0; m <= nmax; m++) {
            norm2[j] += f[j][m].real() * f[j][m].real() + f[j][m].imag() * f[j][m].imag();
        }
    }

    complex<SX> S = complex<SX>(0, 0);

    complex<SX> Sj1, Sj2;

    int j1 = mod(i - 1);
    int j2 = mod(i + 1);

    Sj1 = complex<SX>(0, 0);
    Sj2 = complex<SX>(0, 0);

    if (n < nmax) {
        for (int m = 1; m <= nmax; m++) {
            if (n != m - 1) {
                Sj1 += -J[j1] * g(n, m) * (1 / eps(U0, n, m))
                        * ~f[i][n + 1] * ~f[j1][m - 1] * f[i][n] * f[j1][m];
                Sj2 += -J[i] * g(n, m) * (1 / eps(U0, n, m))
                        * ~f[i][n + 1] * ~f[j2][m - 1] * f[i][n] * f[j2][m];

            }
        }
    }

    S += Sj1;
    S += Sj2;

    return S; //S.imag();
}

complex<SX> canonical(int i, SX& fin, SX& J, SX& U0, SX& dU, SX mu) {

    complex<SX> S = complex<SX>(0, 0);
        for (int n = 0; n <= nmax; n++) {
            S += canonical(i, n, fin, J, U0, dU, mu);
        }
    return S;
}

complex<SX> canonical(SX& fin, SX& J, SX& U0, SX& dU, SX mu) {

    complex<SX> S = complex<SX>(0, 0);
    for (int i = 0; i < L; i++) {
        for (int n = 0; n <= nmax; n++) {
            S += canonical(i, n, fin, J, U0, dU, mu);
        }
    }
    return S;
}

