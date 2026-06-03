/*******************************************************************************
 * core/poly_fit.hpp
 * Header-only polynomial least-squares fit helper.
 *
 * Extracted from calibration_routines.cpp untuk eliminasi duplikasi kode
 * yang identik di mode 3, 5, dan 6 (Gauss elimination 3×3).
 *
 * API:
 *   poly_fit::fit(Z, Y, max_deg)
 *     → std::vector<double> koefisien [a_n, ..., a_1, a_0]
 *     → kompatibel dengan HeightMath::polyval() (konvensi numpy.polyval)
 *
 * Degree dipilih otomatis: min(n_pts - 1, max_deg)
 *   n=1 → degree 0 (konstan)
 *   n=2 → degree 1 (linear):    K = a*Z + b
 *   n≥3 → degree 2 (quadratic): K = a*Z² + b*Z + c
 *
 * Tidak ada dependensi eksternal — hanya <vector>, <cmath>, <cassert>.
 ******************************************************************************/
#pragma once

#include <cassert>
#include <cmath>
#include <vector>

namespace poly_fit {

/**
 * @brief Polynomial least-squares fit: Y ≈ p(Z), degree = min(n-1, max_deg).
 *
 * Normal equations solved via partial-pivoting Gauss elimination.
 * Kompatibel dengan HeightMath::polyval() — koefisien urutan turun:
 *   result[0]*Z^deg + result[1]*Z^(deg-1) + ... + result[deg]
 *
 * @param Z       Variabel bebas (panjang n)
 * @param Y       Variabel terikat (panjang n, sama dengan Z)
 * @param max_deg Degree maksimum yang diijinkan (default 2)
 * @return        Koefisien polinomial, urutan menurun (seperti numpy.polyfit)
 *                Mengembalikan {Y[0]} jika hanya 1 titik.
 */
inline std::vector<double> fit(const std::vector<double>& Z,
                                const std::vector<double>& Y,
                                int max_deg = 2)
{
    assert(Z.size() == Y.size());
    const int n = static_cast<int>(Z.size());
    if (n == 0) return {0.0};
    if (n == 1) return {Y[0]};

    const int deg = std::min(n - 1, max_deg);

    /* ── Degree 0: constant ───────────────────────────────────────────────── */
    if (deg == 0) {
        double sum = 0.0;
        for (int i = 0; i < n; ++i) sum += Y[i];
        return {sum / n};
    }

    /* ── Degree 1: linear p(Z) = a*Z + b ─────────────────────────────────── */
    if (deg == 1) {
        double sz = 0, sz2 = 0, sk = 0, szk = 0;
        for (int i = 0; i < n; ++i) {
            sz  += Z[i];
            sz2 += Z[i] * Z[i];
            sk  += Y[i];
            szk += Z[i] * Y[i];
        }
        const double denom = n * sz2 - sz * sz;
        const double a     = (std::fabs(denom) > 1e-15)
                             ? (n * szk - sz * sk) / denom
                             : 0.0;
        const double b     = (sk - a * sz) / n;
        return {a, b};  // polyval order: a*Z + b
    }

    /* ── Degree 2: quadratic p(Z) = a*Z² + b*Z + c ──────────────────────── */
    // Accumulate sums for normal equations.
    // Unknowns: [a, b, c] (highest power first).
    // Normal equations:
    //   Row 0: Σz⁴·a + Σz³·b + Σz²·c = Σ(y·z²)
    //   Row 1: Σz³·a + Σz²·b + Σz·c  = Σ(y·z)
    //   Row 2: Σz²·a + Σz·b  + n·c   = Σy
    double Sz1 = 0, Sz2 = 0, Sz3 = 0, Sz4 = 0;
    double Sy0 = 0, Syz1 = 0, Syz2 = 0;
    for (int i = 0; i < n; ++i) {
        const double z = Z[i], y = Y[i];
        Sz1  += z;
        Sz2  += z * z;
        Sz3  += z * z * z;
        Sz4  += z * z * z * z;
        Sy0  += y;
        Syz1 += y * z;
        Syz2 += y * z * z;
    }

    double A[3][4] = {
        {Sz4, Sz3, Sz2, Syz2},
        {Sz3, Sz2, Sz1, Syz1},
        {Sz2, Sz1, static_cast<double>(n), Sy0}
    };

    // Gauss elimination with partial pivoting (Jordanification → read directly)
    for (int col = 0; col < 3; ++col) {
        // Find pivot row
        int pivot = col;
        for (int r = col + 1; r < 3; ++r)
            if (std::fabs(A[r][col]) > std::fabs(A[pivot][col]))
                pivot = r;
        if (pivot != col)
            for (int c = 0; c < 4; ++c)
                std::swap(A[col][c], A[pivot][c]);

        if (std::fabs(A[col][col]) < 1e-15) continue;

        // Eliminate column from ALL other rows (Gauss-Jordan)
        for (int r = 0; r < 3; ++r) {
            if (r == col) continue;
            const double f = A[r][col] / A[col][col];
            for (int c = col; c < 4; ++c)
                A[r][c] -= f * A[col][c];
        }
    }

    // Read solution directly from diagonal (Gauss-Jordan gives identity matrix)
    const double a_coef = A[0][3] / A[0][0];  // coeff of Z²
    const double b_coef = A[1][3] / A[1][1];  // coeff of Z
    const double c_coef = A[2][3] / A[2][2];  // constant
    return {a_coef, b_coef, c_coef};           // polyval order: a*Z² + b*Z + c
}

} // namespace poly_fit
