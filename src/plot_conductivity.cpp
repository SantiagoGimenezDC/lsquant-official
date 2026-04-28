/*
 * plot_conductivity.cpp
 *
 * Translates the Python post-processing script plot_conductivity.py.
 * Reads unitcell.txt, params.txt, dos.txt, dos_fine.txt, and the per-timestep
 * muxy_NNNNN.txt files, then computes and writes:
 *
 *   dos_plot.txt          E  DOS
 *   MSD_vs_t.txt          columns: E(0) E(1) ... for each time step
 *   D_vs_t.txt            same layout for diffusivity
 *   sigma_vs_t.txt        same layout for conductivity
 *   sigma_sc.txt          E  sigma_sc
 *   tp.txt                E  tau_p
 *   mu.txt                E  n2D  mu
 *
 * Compile:
 *   g++ -O2 -std=c++17 plot_conductivity.cpp -o plot_conductivity
 */

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iomanip>

// ─────────────────────────────────────────────────────────────────────────────
// Physical constants (SI)
// ─────────────────────────────────────────────────────────────────────────────
static constexpr double H_SI    = 6.62607015e-34;   // J·s
static constexpr double HBAR_SI = 1.05457182e-34;   // J·s
static constexpr double Q       = 1.60217663e-19;   // C
static constexpr double PI      = 3.14159265358979;

// hbar in eV·s
static constexpr double HBAR_EV = HBAR_SI / Q;
// Quantum of conductance G0 = 2e²/h
static constexpr double G0      = 2.0 * Q * Q / H_SI;


// ─────────────────────────────────────────────────────────────────────────────
// Namelist / params.txt parser  (same as in main code)
// ─────────────────────────────────────────────────────────────────────────────
using ParamMap = std::unordered_map<std::string, std::string>;

static ParamMap readNamelist(const std::string& filename)
{
    std::ifstream f(filename);
    if (!f) throw std::runtime_error("Cannot open: " + filename);

    ParamMap P;
    std::string line;
    auto trim = [](std::string s) {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(),
                [](unsigned char c){ return !std::isspace(c); }));
        s.erase(std::find_if(s.rbegin(), s.rend(),
                [](unsigned char c){ return !std::isspace(c); }).base(), s.end());
        return s;
    };

    while (std::getline(f, line)) {
        auto bang = line.find('!');
        if (bang != std::string::npos) line = line.substr(0, bang);
        line = trim(line);
        if (line.empty() || line[0] == '$') continue;
        if (!line.empty() && line.back() == ',') line.pop_back();

        auto eq = line.find('=');
        if (eq == std::string::npos) continue;

        std::string key = trim(line.substr(0, eq));
        std::string val = trim(line.substr(eq + 1));
        std::transform(key.begin(), key.end(), key.begin(), ::tolower);

        if (val.size() >= 2 && val.front() == '\'' && val.back() == '\'')
            val = val.substr(1, val.size() - 2);
        else
            for (char& c : val) if (c=='d'||c=='D') { c='e'; break; }

        P[key] = val;
    }
    return P;
}

template<typename T>
static T getParam(const ParamMap& P, const std::string& key)
{
    std::string lkey = key;
    std::transform(lkey.begin(), lkey.end(), lkey.begin(), ::tolower);
    auto it = P.find(lkey);
    if (it == P.end())
        throw std::runtime_error("Missing parameter: " + key);
    T val;
    std::istringstream ss(it->second);
    if (!(ss >> val))
        throw std::runtime_error("Cannot parse parameter: " + key);
    return val;
}


// ─────────────────────────────────────────────────────────────────────────────
// Low-noise N=11 central-difference differentiator
// (Holoborodko, http://www.holoborodko.com/pavel)
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<double> gradient_lownoise_N11(const std::vector<double>& y, double dx)
{
    int n = static_cast<int>(y.size());
    std::vector<double> d(n, 0.0);
    if (n < 2) return d;

    for (int i = 0; i < n; ++i) {
        if      (i == 0)     d[i] = (y[i+1]-y[i]) / dx;
        else if (i == n-1)   d[i] = (y[i]-y[i-1]) / dx;
        else if (i==1||i==n-2)
            d[i] = (y[i+1]-y[i-1]) / (2*dx);
        else if (i==2||i==n-3)
            d[i] = (2*(y[i+1]-y[i-1]) + (y[i+2]-y[i-2])) / (8*dx);
        else if (i==3||i==n-4)
            d[i] = (5*(y[i+1]-y[i-1]) + 4*(y[i+2]-y[i-2]) + (y[i+3]-y[i-3])) / (32*dx);
        else if (i==4||i==n-5)
            d[i] = (14*(y[i+1]-y[i-1]) + 14*(y[i+2]-y[i-2])
                  + 6*(y[i+3]-y[i-3]) + (y[i+4]-y[i-4])) / (128*dx);
        else
            d[i] = (42*(y[i+1]-y[i-1]) + 48*(y[i+2]-y[i-2])
                  + 27*(y[i+3]-y[i-3]) + 8*(y[i+4]-y[i-4])
                  + (y[i+5]-y[i-5])) / (512*dx);
    }
    return d;
}


// ─────────────────────────────────────────────────────────────────────────────
// Cumulative trapezoidal integration  (matches scipy.integrate.cumtrapz)
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<double> cumtrapz(const std::vector<double>& y,
                                    const std::vector<double>& x)
{
    int n = static_cast<int>(y.size());
    std::vector<double> out(n, 0.0);
    for (int i = 1; i < n; ++i)
        out[i] = out[i-1] + 0.5*(y[i]+y[i-1])*(x[i]-x[i-1]);
    return out;
}


// ─────────────────────────────────────────────────────────────────────────────
// Linear interpolation of src defined on x_src, evaluated at x_dst
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<double> interp(const std::vector<double>& x_dst,
                                  const std::vector<double>& x_src,
                                  const std::vector<double>& y_src)
{
    int n = static_cast<int>(x_dst.size());
    std::vector<double> out(n);
    for (int i = 0; i < n; ++i) {
        double xq = x_dst[i];
        // Find bracket
        auto it = std::lower_bound(x_src.begin(), x_src.end(), xq);
        if (it == x_src.begin()) { out[i] = y_src[0]; continue; }
        if (it == x_src.end())   { out[i] = y_src.back(); continue; }
        int j = static_cast<int>(it - x_src.begin());
        double t = (xq - x_src[j-1]) / (x_src[j] - x_src[j-1]);
        out[i] = y_src[j-1] + t*(y_src[j] - y_src[j-1]);
    }
    return out;
}


// ─────────────────────────────────────────────────────────────────────────────
// Write a 2-column file
// ─────────────────────────────────────────────────────────────────────────────
static void write2col(const std::string& fname,
                      const std::vector<double>& a,
                      const std::vector<double>& b)
{
    std::ofstream f(fname);
    if (!f) throw std::runtime_error("Cannot write: " + fname);
    f << std::scientific << std::setprecision(8);
    for (size_t i = 0; i < a.size(); ++i)
        f << a[i] << "  " << b[i] << "\n";
    std::cout << "  Written " << fname << "\n";
}

// Write a matrix where rows = energies, columns = time steps
// Header line contains the time values
static void writeMatrix(const std::string& fname,
                        const std::vector<double>& E,
                        const std::vector<double>& time,
                        const std::vector<std::vector<double>>& M)
{
    std::ofstream f(fname);
    if (!f) throw std::runtime_error("Cannot write: " + fname);
    f << std::scientific << std::setprecision(8);

    // Header: time values
    f << "#E";
    for (double t : time) f << "  " << t;
    f << "\n";

    // Rows: one per energy
    for (size_t iE = 0; iE < E.size(); ++iE) {
        f << E[iE];
        for (size_t it = 0; it < time.size(); ++it)
            f << "  " << M[iE][it];
        f << "\n";
    }
    std::cout << "  Written " << fname << "\n";
}


// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main()
{
    // ── Read unitcell.txt ─────────────────────────────────────────────────────
    std::ifstream fsamp("unitcell.txt");
    if (!fsamp) throw std::runtime_error("Cannot open unitcell.txt");

    double A1x, A1y, A2x, A2y, acc;
    int    natoms;
    {
        std::string line;
        std::getline(fsamp, line); std::istringstream(line) >> A1x >> A1y;
        std::getline(fsamp, line); std::istringstream(line) >> A2x >> A2y;
        std::getline(fsamp, line); std::istringstream(line) >> acc;
        std::getline(fsamp, line);   // skip ncutoff
        std::getline(fsamp, line); std::istringstream(line) >> natoms;
    }
    fsamp.close();

    double L1   = std::sqrt(A1x*A1x + A1y*A1y);
    double L2   = std::sqrt(A2x*A2x + A2y*A2y);

    // ── Read params.txt ───────────────────────────────────────────────────────
    auto   P     = readNamelist("params.txt");
    double tstep = getParam<double>(P, "tstep");     // [fs]
    int    nT    = getParam<int>   (P, "nT");
    double teV   = getParam<double>(P, "teV");       // hopping [eV]
    double Emin  = getParam<double>(P, "Emin");
    double Emax  = getParam<double>(P, "Emax");
    double dE    = getParam<double>(P, "dE");
    int    nrow  = getParam<int>   (P, "nrow");
    int    ncol  = getParam<int>   (P, "ncol");

    int    nE    = static_cast<int>((Emax - Emin) / dE) + 1;
    double Lmax  = std::min(ncol*L1, nrow*L2);  // sample dimension [Å]

    // Graphene Fermi velocity  vF = (3/2) * acc[m] * t[eV] / hbar[eV·s]
    double vf    = 1.5 * (acc * 1e-10) * teV / HBAR_EV;

    // Area per atom [m²]
    double A_m2  = (A1x*A2y - A1y*A2x) * 1e-20 / natoms;

    // Time vector [fs]
    std::vector<double> time(nT + 1);
    for (int i = 0; i <= nT; ++i) time[i] = i * tstep;

    // Energy grid
    std::vector<double> E(nE);
    for (int i = 0; i < nE; ++i) E[i] = Emin + i * dE;

    std::cout << "\nSample info:\n";
    std::cout << "  natoms = " << natoms << "\n";
    std::cout << "  Lmax   = " << Lmax   << " Å\n";
    std::cout << "  nE     = " << nE     << ",  nT = " << nT << "\n\n";


    // ── Read DOS [1/J/m²] ─────────────────────────────────────────────────────
    auto readTwoCol = [](const std::string& fn)
        -> std::pair<std::vector<double>, std::vector<double>>
    {
        std::ifstream f(fn);
        if (!f) throw std::runtime_error("Cannot open: " + fn);
        std::vector<double> xa, ya;
        double a, b;
        while (f >> a >> b) { xa.push_back(a); ya.push_back(b); }
        return {xa, ya};
    };

    auto [E_dos,    dos_raw]     = readTwoCol("dos.txt");
    auto [E_fine,   dosfine_raw] = readTwoCol("dos_fine.txt");

    // Convert DOS from file units to [1/J/m²]: dos = 2*dos_file / A_m2 / Q
    std::vector<double> dos(E_dos.size()), dosfine(E_fine.size());
    for (size_t i = 0; i < dos.size();     ++i) dos[i]     = 2.0*dos_raw[i]     / A_m2 / Q;
    for (size_t i = 0; i < dosfine.size(); ++i) dosfine[i] = 2.0*dosfine_raw[i] / A_m2 / Q;

    // Charge neutrality point = minimum of fine DOS
    int iCNP = static_cast<int>(
        std::min_element(dosfine.begin(), dosfine.end()) - dosfine.begin());
    std::cout << "  CNP at E = " << E_fine[iCNP]
              << " eV  (index " << iCNP << ")\n";

    // Carrier density n2D [1/m²] via cumulative trapezoidal integration
    // Subtract CNP value so n2D = 0 at the Dirac point
    auto n2Dfine_raw = cumtrapz(dosfine, E_fine);   // dosfine in 1/J/m², E in eV → integrate over eV→multiply by Q
    // Actually integrate over eV: n [1/m²] = ∫ dos[1/J/m²] * Q[J/eV] dE[eV]
    std::vector<double> n2Dfine(n2Dfine_raw.size());
    for (size_t i = 0; i < n2Dfine.size(); ++i)
        n2Dfine[i] = (n2Dfine_raw[i] - n2Dfine_raw[iCNP]) * Q;

    // Interpolate onto the zoomed energy grid
    auto n2D = interp(E, E_fine, n2Dfine);

    std::cout << "  n2D at CNP = " << n2Dfine[iCNP] << " m⁻²\n\n";

    // Interpolate DOS onto energy grid E (needed for conductivity)
    auto dos_E = interp(E, E_dos, dos);


    // ── Read MSD files [Å²] ───────────────────────────────────────────────────
    // dL2[iE][it] = dX²(iE,it) + dY²(iE,it)
    std::vector<std::vector<double>> dL2(nE, std::vector<double>(nT+1, 0.0));

    for (int it = 1; it <= nT; ++it) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "muxy_%05d.txt", it);
        std::ifstream f(buf);
        if (!f) throw std::runtime_error(std::string("Cannot open ") + buf);

        // Header line: "# normdx normdy"
        std::string header;
        std::getline(f, header);
        double normdx, normdy;
        std::istringstream(header.substr(1)) >> normdx >> normdy;

        // Rows: index  Re(mux) Im(mux)  Re(muy) Im(muy)
        // The MSD in Å² is normdx² * Re(mux) for x, etc.
        // Here we follow the Fortran convention: dX2 = real part of mux * normdx²
        int    idx;
        double re_mux, im_mux, re_muy, im_muy;
        int    iE_row = 0;
        while (f >> idx >> re_mux >> im_mux >> re_muy >> im_muy) {
            if (iE_row < nE) {
                double dX2 = normdx * normdx * re_mux;
                double dY2 = normdy * normdy * re_muy;
                dL2[iE_row][it] = dX2 + dY2;
            }
            ++iE_row;
        }
    }


    // ── Compute D(t), sigma(t), and semiclassical quantities ─────────────────
    std::vector<std::vector<double>> D    (nE, std::vector<double>(nT+1, 0.0));
    std::vector<std::vector<double>> sigma(nE, std::vector<double>(nT+1, 0.0));
    std::vector<double> D_sc    (nE, 0.0);
    std::vector<double> sigma_sc(nE, 0.0);
    std::vector<double> tp      (nE, 0.0);

    for (int iE = 0; iE < nE; ++iE) {
        // Diffusivity D = d(dL2)/dt / 4  [m²/s]
        // Input dL2 in Å², time in fs → convert to m²/s: ×1e-20/1e-15 = ×1e-5
        auto dL2_slice = dL2[iE];
        auto grad      = gradient_lownoise_N11(dL2_slice, tstep);
        for (int it = 0; it <= nT; ++it)
            D[iE][it] = grad[it] / 4.0 * 1e-20 / 1e-15;
        D[iE][0] = 0.0;

        // Conductivity σ(t) = e² * DOS * D  [1/Ω]
        for (int it = 0; it <= nT; ++it)
            sigma[iE][it] = Q * Q * dos_E[iE] * D[iE][it];

        // Semiclassical: max of D and σ over time
        D_sc[iE]     = *std::max_element(D[iE].begin(),     D[iE].end());
        sigma_sc[iE] = *std::max_element(sigma[iE].begin(), sigma[iE].end());

        // Momentum relaxation time τ_p = 2*D_sc / vF²  [fs]
        tp[iE] = 2.0 * D_sc[iE] / (vf * vf) * 1e15;
    }

    // ── Mobility [cm²/V·s] from generalised Einstein relation ─────────────────
    std::vector<double> mu(nE, 0.0);
    for (int iE = 0; iE < nE; ++iE) {
        double dndE = dos_E[iE];   // [1/J/m²]
        double n    = std::abs(n2D[iE]);
        if (n > 0.0)
            mu[iE] = D_sc[iE] * dndE / n * Q * 1e4;  // ×1e4: m²→cm²
    }


    // ── Write output files ────────────────────────────────────────────────────
    std::cout << "\nWriting output files...\n";

    // DOS: E [eV]  dos [1/eV/nm²]
    {
        std::vector<double> dos_plot(nE);
        for (int i = 0; i < nE; ++i)
            dos_plot[i] = dos_E[i] * Q / 1e18;   // 1/J/m² → 1/eV/nm²
        write2col("dos_plot.txt", E, dos_plot);
    }

    // MSD vs time: rows=energies, cols=time steps  [µm²]
    {
        std::vector<std::vector<double>> MSD_um(nE, std::vector<double>(nT+1));
        for (int iE = 0; iE < nE; ++iE)
            for (int it = 0; it <= nT; ++it)
                MSD_um[iE][it] = dL2[iE][it] / (1e4*1e4);   // Å² → µm²
        writeMatrix("MSD_vs_t.txt", E, time, MSD_um);
    }

    // Diffusivity vs time [cm²/s]
    {
        std::vector<std::vector<double>> D_cm(nE, std::vector<double>(nT+1));
        for (int iE = 0; iE < nE; ++iE)
            for (int it = 0; it <= nT; ++it)
                D_cm[iE][it] = D[iE][it] * 1e4;   // m²/s → cm²/s
        writeMatrix("D_vs_t.txt", E, time, D_cm);
    }

    // Conductivity vs time [G0]
    {
        std::vector<std::vector<double>> sig_G0(nE, std::vector<double>(nT+1));
        for (int iE = 0; iE < nE; ++iE)
            for (int it = 0; it <= nT; ++it)
                sig_G0[iE][it] = sigma[iE][it] / G0;
        writeMatrix("sigma_vs_t.txt", E, time, sig_G0);
    }

    // Semiclassical conductivity [G0]
    {
        std::vector<double> ssc(nE);
        for (int i = 0; i < nE; ++i) ssc[i] = sigma_sc[i] / G0;
        write2col("sigma_sc.txt", E, ssc);
    }

    // Momentum relaxation time [fs]
    write2col("tp.txt", E, tp);

    // Mobility [cm²/V·s]  (also writes n2D in cm⁻²)
    {
        std::ofstream f("mu.txt");
        f << std::scientific << std::setprecision(8);
        for (int i = 0; i < nE; ++i)
            f << E[i] << "  " << n2D[i]/1e4 << "  " << mu[i] << "\n";
        std::cout << "  Written mu.txt\n";
    }

    std::cout << "\nDone.\n";
    return 0;
}
