// mhd_cuda.cu -- CUDA C++ ports of MHDTurbulence's five core kernels
// (StateVevtor, NumericalFlux1/2/3, UpdateConsv) from the optimize-ugly
// branch (2831841). Compiled with `nvc++ -cuda` so kernels, the OpenACC
// runtime and the device allocations share one NVHPC runtime; launchers
// run on the stream returned by acc_get_cuda_stream so they interleave
// with the untouched OpenACC regions. Pointers arrive via acc_deviceptr;
// the OpenACC runtime keeps ownership of all data.
//
// Deviations from the Fortran arithmetic (ULP-level, gated on t-prof
// agreement like every adopted change in this campaign):
//  - the MUSCL->HLLD interface carries primitives directly; the Fortran
//    round-trips through conservative variables and re-derives v from
//    rho*v with 6 divides per face (plus v*rho vs rho*v and rotated
//    summation orders for directions 2/3),
//  - UpdateConsv hoists 1/dx out of the per-field flux sums.
// Requires ncomp=1 (nhyd=12, mflx=10); the glue refuses anything else.
// Index maps (0-based):
//  svc:  0 d, 1 v1, 2 v2, 3 v3, 4 e/rho, 5 p, 6 cs, 7 b1, 8 b2, 9 b3,
//        10 bp, 11 X
//  flux: 0 dn, 1 rv1, 2 rv2, 3 rv3, 4 et, 5 b1, 6 b2, 7 b3, 8 bp, 9 X
// The Fortran rotates (vn, t1, t2) per direction before HLLD and
// unrotates on store; here that rotation is a compile-time template.
// HLLD is a variable-for-variable transcription of the if/else Fortran
// (guards, NaN semantics, and factor orders preserved); eel/eer are
// TOTAL energies (leftst(muet)), as there.

#include <cmath>
#include <cstdint>

#define DEVINL __device__ __forceinline__

// ======================================================================
// StateVevtor: primitive -> state-vector conversion.
// Fortran loops i,j,k = 1..n-1 (includes the inner ghost ring).
// ======================================================================
__global__ void __launch_bounds__(256) statevevtor_cuda_kernel(
    const double* __restrict__ d_, const double* __restrict__ v1_,
    const double* __restrict__ v2_, const double* __restrict__ v3_,
    const double* __restrict__ ei_, const double* __restrict__ b1_,
    const double* __restrict__ b2_, const double* __restrict__ b3_,
    const double* __restrict__ bp_, const double* __restrict__ xcomp_,
    double* __restrict__ p_, double* __restrict__ svc_,
    double gam, int in_, int jn_, int kn_,
    int i1, int i2, int j1, int j2, int k1, int k2) {
  long long nxi = i2 - i1 + 1, nxj = j2 - j1 + 1;
  long long total = nxi * nxj * (k2 - k1 + 1);
  for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
       t < total; t += (long long)gridDim.x * blockDim.x) {
    int i = (int)(t % nxi);
    long long tj = t / nxi;
    int j = (int)(tj % nxj);
    int k = (int)(tj / nxj);
    long long ci = (long long)(i1 - 1 + i) +
                   (long long)in_ * (j1 - 1 + j) +
                   (long long)in_ * jn_ * (k1 - 1 + k);
    double dv  = d_[ci];
    double v1v = v1_[ci], v2v = v2_[ci], v3v = v3_[ci];
    double eiv = ei_[ci];
    double b1v = b1_[ci], b2v = b2_[ci], b3v = b3_[ci], bpv = bp_[ci];
    double xv  = xcomp_[ci];
    double g1  = gam - 1.0;
    double2* q = (double2*)(svc_ + ci * 12);
    q[0] = make_double2(dv,   v1v);
    q[1] = make_double2(v2v,  v3v);
    q[2] = make_double2(eiv / dv, eiv * g1);
    q[3] = make_double2(sqrt(gam * g1 * eiv / dv), b1v);
    q[4] = make_double2(b2v,  b3v);
    q[5] = make_double2(bpv,  xv);
    p_[ci] = eiv * g1;
  }
}

extern "C" int statevevtor_cuda(const uintptr_t d_, const uintptr_t v1_, const uintptr_t v2_,
    const uintptr_t v3_, const uintptr_t ei_, const uintptr_t b1_, const uintptr_t b2_,
    const uintptr_t b3_, const uintptr_t bp_, const uintptr_t xcomp_, uintptr_t p_,
    uintptr_t svc_, double gam, int in_, int jn_, int kn_,
    int i1, int i2, int j1, int j2, int k1, int k2, uintptr_t stream) {
  long long total = (long long)(i2 - i1 + 1) * (j2 - j1 + 1) * (k2 - k1 + 1);
  int blocks = (int)((total + 255) / 256);
  cudaStream_t s = (cudaStream_t)stream;
  statevevtor_cuda_kernel<<<blocks, 256, 0, s>>>(
      (const double*)d_, (const double*)v1_, (const double*)v2_,
      (const double*)v3_, (const double*)ei_, (const double*)b1_,
      (const double*)b2_, (const double*)b3_, (const double*)bp_,
      (const double*)xcomp_, (double*)p_, (double*)svc_,
      gam, in_, jn_, kn_, i1, i2, j1, j2, k1, k2);
  return (int)cudaGetLastError();
}

// ======================================================================
// NumericalFlux1/2/3
// ======================================================================

DEVINL double vl_limiter(double dvp, double dvm) {
  // vanLeer: 2*dvp*dvm/(dvp+dvm) for dvp*dvm > 0, else 0. Factor order
  // mirrors the Fortran expression's left-to-right evaluation.
  double pr = dvp * dvm;
  return (pr > 0.0) ? ((2.0 * dvp) * dvm) / (dvp + dvm) : 0.0;
}

// Rotated component indices per direction (0-based svc fields).
template <int DIR> struct DirMap {};
template <> struct DirMap<0> { static const int vn = 1, t1 = 2, t2 = 3,
                                        bn = 7, bt1 = 8, bt2 = 9; };
template <> struct DirMap<1> { static const int vn = 2, t1 = 3, t2 = 1,
                                        bn = 8, bt1 = 9, bt2 = 7; };
template <> struct DirMap<2> { static const int vn = 3, t1 = 1, t2 = 2,
                                        bn = 9, bt1 = 7, bt2 = 8; };

// HLLD in the rotated frame (u = normal, v = t1, w = t2). All
// star-state variables live at function scope, mirroring the Fortran.
DEVINL void hlld_rot(
    // left state (primitives + conservative momenta + flux + fast speed)
    double rol, double vxl, double vyl, double vzl, double eel, double ptl,
    double byl, double bzl, double scl, double rxl, double ryl, double rzl,
    double cfl, double bnL, double bpL,
    double frol, double frxl, double fryl, double frzl, double feel,
    double fbyl, double fbzl, double fscl,
    // right state
    double ror, double vxr, double vyr, double vzr, double eer, double ptr,
    double byr, double bzr, double scr, double rxr, double ryr, double rzr,
    double cfr, double bnR, double bpR,
    double fror, double frxr, double fryr, double frzr, double feer,
    double fbyr, double fbzr, double fscr,
    // outputs (normal / t1 / t2 fluxes + et + tracer; b1/bp GLM slots
    // are filled by the caller)
    double& n_dn, double& n_vn, double& n_vt1, double& n_vt2, double& n_et,
    double& n_bt1, double& n_bt2, double& n_sc) {
  const double eps = 1e-30;
  double rolst, vxlst, vylst, vzlst, bylst, bzlst, vdbstl, eelst, sclst;
  double rylst, rzlst;
  double rorst, vxrst, vyrst, vzrst, byrst, bzrst, vdbstr, eerst, scrst;
  double rxrst, ryrst, rzrst;
  double vyldst, vyrdst, vzldst, vzrdst, byldst, byrdst, bzldst, bzrdst;
  double ryldst, rzldst, ryrdst, rzrdst, eeldst, eerdst;
  double slst, srst;

  // Step 1: wave speeds
  const double sl = fmin(vxl, vxr) - fmax(cfl, cfr);
  const double sr = fmax(vxl, vxr) + fmax(cfl, cfr);

  // Step 4: middle and Alfven wave
  const double sdl = fmin(-1e-20, sl - vxl);
  const double sdr = fmax(1e-20, sr - vxr);
  const double rosdl = rol * sdl;
  const double rosdr = ror * sdr;
  const double temp = 1.0 / (rosdr - rosdl);
  const double sm = (rosdr * vxr - rosdl * vxl - ptr + ptl) * temp;
  const double sdml = fmin(-1e-20, sl - sm);
  const double isdml = 1.0 / sdml;
  const double sdmr = fmax(1e-20, sr - sm);
  const double isdmr = 1.0 / sdmr;

  // Step 5: intermediate states
  const double ptst = (rosdr * ptl - rosdl * ptr + rosdl * rosdr * (vxr - vxl)) * temp;
  const double bxs = 0.5 * (bnL + bnR);
  const double bxsq = bxs * bxs;

  // 5A: U*_L
  const double temp_fstL = rosdl * sdml - bxsq;
  if (fabs(temp_fstL) >= eps) {
    const double itf = 1.0 / temp_fstL;
    rolst = rosdl * isdml;
    vxlst = sm;
    double tmp = bxs * (sdl - sdml) * itf;
    vylst = vyl - byl * tmp;
    vzlst = vzl - bzl * tmp;
    tmp = (rosdl * sdl - bxsq) * itf;
    bylst = byl * tmp;
    bzlst = bzl * tmp;
    vdbstl = vxlst * bxs + vylst * bylst + vzlst * bzlst;
    eelst = (sdl * eel - ptl * vxl + ptst * sm +
             bxs * (vxl * bxs + vyl * byl + vzl * bzl - vdbstl)) * isdml;
  } else {
    rolst = rol; vxlst = vxl; vylst = vyl; vzlst = vzl;
    bylst = byl; bzlst = bzl;
    vdbstl = vxlst * bxs + vylst * bylst + vzlst * bzlst;
    eelst = eel;
  }
  sclst = rolst / rol * scl;
  rylst = rolst * vylst;
  rzlst = rolst * vzlst;

  // 5B: U*_R
  const double temp_fstR = rosdr * sdmr - bxsq;
  if (fabs(temp_fstR) >= eps) {
    const double itf = 1.0 / temp_fstR;
    rorst = rosdr * isdmr;
    vxrst = sm;
    double tmp = bxs * (sdr - sdmr) * itf;
    vyrst = vyr - byr * tmp;
    vzrst = vzr - bzr * tmp;
    tmp = (rosdr * sdr - bxsq) * itf;
    byrst = byr * tmp;
    bzrst = bzr * tmp;
    vdbstr = vxrst * bxs + vyrst * byrst + vzrst * bzrst;
    eerst = ((sdr * eer - ptr * vxr + ptst * sm) * isdmr +
             bxs * (vxr * bxs + vyr * byr + vzr * bzr - vdbstr) * isdmr);
  } else {
    rorst = ror; vxrst = vxr; vyrst = vyr; vzrst = vzr;
    byrst = byr; bzrst = bzr;
    vdbstr = vxrst * bxs + vyrst * byrst + vzrst * bzrst;
    eerst = eer;
  }
  scrst = rorst / ror * scr;
  rxrst = rorst * vxrst;
  ryrst = rorst * vyrst;
  rzrst = rorst * vzrst;

  // 5C: U**_L and U**_R
  const double abbx = fabs(bxs);
  if (abbx >= eps) {
    const double sqrtrol = sqrt(rolst);
    const double sqrtror = sqrt(rorst);
    const double signbx = (bxs < 0.0) ? -1.0 : 1.0;
    const double invsumro = 1.0 / (sqrtrol + sqrtror);
    double tmp = invsumro * (sqrtrol * vylst + sqrtror * vyrst +
                             signbx * (byrst - bylst));
    vyldst = tmp; vyrdst = tmp;
    ryldst = rolst * tmp; ryrdst = rorst * tmp;
    tmp = invsumro * (sqrtrol * vzlst + sqrtror * vzrst +
                      signbx * (bzrst - bzlst));
    vzldst = tmp; vzrdst = tmp;
    rzldst = rolst * tmp; rzrdst = rorst * tmp;
    tmp = invsumro * (sqrtrol * byrst + sqrtror * bylst +
                      signbx * sqrtrol * sqrtror * (vyrst - vylst));
    byldst = tmp; byrdst = tmp;
    tmp = invsumro * (sqrtrol * bzrst + sqrtror * bzlst +
                      signbx * sqrtrol * sqrtror * (vzrst - vzlst));
    bzldst = tmp; bzrdst = tmp;
    const double temp2 = sm * bxs + vyldst * byldst + vzldst * bzldst;
    eeldst = eelst - sqrtrol * signbx * (vdbstl - temp2);
    eerdst = eerst + sqrtror * signbx * (vdbstr - temp2);
    slst = sm - abbx / sqrtrol;
    srst = sm + abbx / sqrtror;
  } else {
    vyldst = vylst; vyrdst = vyrst; ryldst = rylst; ryrdst = ryrst;
    vzldst = vzlst; vzrdst = vzrst; rzldst = rzlst; rzrdst = rzrst;
    byldst = bylst; byrdst = byrst; bzldst = bzlst; bzrdst = bzrst;
    eeldst = eelst; eerdst = eerst;
    slst = 0.0; srst = 0.0;
  }

  // Step 6: flux. The Fortran's `if (sm >= 0) ... else if (sm < 0)`
  // leaves nflux untouched for NaN sm; the else-if here matches that.
  const double msl = fmin(sl, 0.0);
  const double msr = fmax(sr, 0.0);
  const double mslst = fmin(slst, 0.0);
  const double msrst = fmax(srst, 0.0);
  if (sm >= 0.0) {
    n_dn  = frol + (rolst - rol) * msl;
    n_et  = feel + (eelst - eel) * msl + (eeldst - eelst) * mslst;
    n_vn  = frxl + (rolst * vxlst - rxl) * msl;
    n_vt1 = fryl + (rylst - ryl) * msl + (ryldst - rylst) * mslst;
    n_vt2 = frzl + (rzlst - rzl) * msl + (rzldst - rzlst) * mslst;
    n_bt1 = fbyl + (bylst - byl) * msl + (byldst - bylst) * mslst;
    n_bt2 = fbzl + (bzlst - bzl) * msl + (bzldst - bzlst) * mslst;
    n_sc  = fscl + msl * (sclst - scl);
  } else if (sm < 0.0) {
    n_dn  = fror + (rorst - ror) * msr;
    n_et  = feer + (eerst - eer) * msr + (eerdst - eerst) * msrst;
    n_vn  = frxr + (rxrst - rxr) * msr;
    n_vt1 = fryr + (ryrst - ryr) * msr + (ryrdst - ryrst) * msrst;
    n_vt2 = frzr + (rzrst - rzr) * msr + (rzrdst - rzrst) * msrst;
    n_bt1 = fbyr + (byrst - byr) * msr + (byrdst - byrst) * msrst;
    n_bt2 = fbzr + (bzrst - bzr) * msr + (bzrdst - bzrst) * msrst;
    n_sc  = fscr + msr * (scrst - scr);
  }
}

// One face's MUSCL + HLLD + GLM, direction-generic via DIR.
template <int DIR>
DEVINL void face_flux(const double* __restrict__ svc,
                      double* __restrict__ nfx, long long ci_face,
                      long long coff, double chg) {
  // Load the 4 stencil cells (contiguous 12-field rows).
  double A0[12], A1[12], A2[12], A3[12];
  {
    const double2* q0 = (const double2*)(svc + (ci_face - 2 * coff) * 12);
    const double2* q1 = (const double2*)(svc + (ci_face - coff) * 12);
    const double2* q2 = (const double2*)(svc + ci_face * 12);
    const double2* q3 = (const double2*)(svc + (ci_face + coff) * 12);
    #pragma unroll
    for (int k = 0; k < 6; ++k) {
      double2 w;
      w = q0[k]; A0[2*k] = w.x; A0[2*k+1] = w.y;
      w = q1[k]; A1[2*k] = w.x; A1[2*k+1] = w.y;
      w = q2[k]; A2[2*k] = w.x; A2[2*k+1] = w.y;
      w = q3[k]; A3[2*k] = w.x; A3[2*k+1] = w.y;
    }
  }

  // MUSCL slopes: left state sits at the right edge of cell i-1, right
  // state at the left edge of cell i (Fortran Plefte/Prigte).
  double PL[12], PR[12];
  {
    #pragma unroll
    for (int f = 0; f < 12; ++f) {
      double dvL = vl_limiter(A2[f] - A1[f], A1[f] - A0[f]);
      double dvR = vl_limiter(A3[f] - A2[f], A2[f] - A1[f]);
      PL[f] = A1[f] + 0.5 * dvL;
      PR[f] = A2[f] - 0.5 * dvR;
    }
  }

  // Rotated primitives.
  const double rol = PL[0], ror = PR[0];
  const double vxl = PL[DirMap<DIR>::vn], vxr = PR[DirMap<DIR>::vn];
  const double vyl = PL[DirMap<DIR>::t1], vyr = PR[DirMap<DIR>::t1];
  const double vzl = PL[DirMap<DIR>::t2], vzr = PR[DirMap<DIR>::t2];
  const double bnl = PL[DirMap<DIR>::bn], bnr = PR[DirMap<DIR>::bn];
  const double bt1l = PL[DirMap<DIR>::bt1], bt1r = PR[DirMap<DIR>::bt1];
  const double bt2l = PL[DirMap<DIR>::bt2], bt2r = PR[DirMap<DIR>::bt2];
  const double bpl = PL[10], bpr = PR[10];

  // Per-side conservative momenta, total energy, fluxes, fast speed.
  const double etotl = PL[4] * rol + 0.5 * rol * (vxl * vxl + vyl * vyl + vzl * vzl)
                     + 0.5 * (bnl * bnl + bt1l * bt1l + bt2l * bt2l);
  const double etotr = PR[4] * ror + 0.5 * ror * (vxr * vxr + vyr * vyr + vzr * vzr)
                     + 0.5 * (bnr * bnr + bt1r * bt1r + bt2r * bt2r);
  const double ptl = PL[5] + 0.5 * (bnl * bnl + bt1l * bt1l + bt2l * bt2l);
  const double ptr = PR[5] + 0.5 * (bnr * bnr + bt1r * bt1r + bt2r * bt2r);

  const double frol = rol * vxl;
  const double frxl = rol * vxl * vxl + ptl - bnl * bnl;
  const double fryl = rol * vyl * vxl - bt1l * bnl;
  const double frzl = rol * vzl * vxl - bt2l * bnl;
  const double feel = (etotl + ptl) * vxl -
                      (vxl * bnl + vyl * bt1l + vzl * bt2l) * bnl;
  const double fbyl = bt1l * vxl - vyl * bnl;
  const double fbzl = bt2l * vxl - vzl * bnl;
  const double fscl = rol * PL[11] * vxl;

  const double fror = ror * vxr;
  const double frxr = ror * vxr * vxr + ptr - bnr * bnr;
  const double fryr = ror * vyr * vxr - bt1r * bnr;
  const double frzr = ror * vzr * vxr - bt2r * bnr;
  const double feer = (etotr + ptr) * vxr -
                      (vxr * bnr + vyr * bt1r + vzr * bt2r) * bnr;
  const double fbyr = bt1r * vxr - vyr * bnr;
  const double fbzr = bt2r * vxr - vzr * bnr;
  const double fscr = ror * PR[11] * vxr;

  const double cssl = PL[6] * PL[6];
  const double ctsl = cssl + (bnl * bnl + bt1l * bt1l + bt2l * bt2l) / rol;
  const double cfl = sqrt((ctsl + sqrt(ctsl * ctsl - 4.0 * cssl * bnl * bnl / rol)) / 2.0);
  const double cssr = PR[6] * PR[6];
  const double ctsr = cssr + (bnr * bnr + bt1r * bt1r + bt2r * bt2r) / ror;
  const double cfr = sqrt((ctsr + sqrt(ctsr * ctsr - 4.0 * cssr * bnr * bnr / ror)) / 2.0);

  double n_dn, n_vn, n_vt1, n_vt2, n_et, n_bt1, n_bt2, n_sc;
  hlld_rot(rol, vxl, vyl, vzl, etotl, ptl, bt1l, bt2l, rol * PL[11],
           vxl * rol, vyl * rol, vzl * rol, cfl, bnl, bpl,
           frol, frxl, fryl, frzl, feel, fbyl, fbzl, fscl,
           ror, vxr, vyr, vzr, etotr, ptr, bt1r, bt2r, ror * PR[11],
           vxr * ror, vyr * ror, vzr * ror, cfr, bnr, bpr,
           fror, frxr, fryr, frzr, feer, fbyr, fbzr, fscr,
           n_dn, n_vn, n_vt1, n_vt2, n_et, n_bt1, n_bt2, n_sc);

  // GLM finite-volume terms (Fortran: the mubu slot carries the normal
  // B in the rotated frame, mubp carries psi).
  const double glm_bn = 0.5 * (bpl + bpr) - 0.5 * chg * (bnr - bnl);
  const double glm_bp = (0.5 * (bnl + bnr) - 0.5 / chg * (bpr - bpl)) * chg * chg;

  // Unrotate and store (10 contiguous fields, 80 B, 16 B-aligned).
  double out[10];
  if (DIR == 0) {
    out[0] = n_dn;  out[1] = n_vn;  out[2] = n_vt1; out[3] = n_vt2;
    out[4] = n_et;  out[5] = glm_bn; out[6] = n_bt1; out[7] = n_bt2;
    out[8] = glm_bp; out[9] = n_sc;
  } else if (DIR == 1) {
    out[0] = n_dn;  out[1] = n_vt2; out[2] = n_vn;  out[3] = n_vt1;
    out[4] = n_et;  out[5] = n_bt2; out[6] = glm_bn; out[7] = n_bt1;
    out[8] = glm_bp; out[9] = n_sc;
  } else {
    out[0] = n_dn;  out[1] = n_vt1; out[2] = n_vt2; out[3] = n_vn;
    out[4] = n_et;  out[5] = n_bt1; out[6] = n_bt2; out[7] = glm_bn;
    out[8] = glm_bp; out[9] = n_sc;
  }
  double2* q = (double2*)(nfx + ci_face * 10);
  #pragma unroll
  for (int k = 0; k < 5; ++k) q[k] = make_double2(out[2*k], out[2*k+1]);
}

template <int DIR>
__global__ void flux_kernel(const double* __restrict__ svc,
                            double* __restrict__ nfx,
                            const double* __restrict__ chg_p,
                            int in_, int jn_, int kn_,
                            int i1, int i2, int j1, int j2, int k1, int k2) {
  // Face counts: f = face-running axis, then the two transverse axes.
  long long cnt_f, cnt_2, cnt_3;
  long long coff;    // cell-linear step per face step
  if (DIR == 0) {
    cnt_f = i2 - i1 + 1; cnt_2 = j2 - j1 + 1; cnt_3 = k2 - k1 + 1;
    coff = 1;
  } else if (DIR == 1) {
    cnt_f = j2 - j1 + 1; cnt_2 = i2 - i1 + 1; cnt_3 = k2 - k1 + 1;
    coff = in_;
  } else {
    cnt_f = k2 - k1 + 1; cnt_2 = i2 - i1 + 1; cnt_3 = j2 - j1 + 1;
    coff = (long long)in_ * jn_;
  }
  const long long total = cnt_f * cnt_2 * cnt_3;
  const double chg = *chg_p;
  for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
       t < total; t += (long long)gridDim.x * blockDim.x) {
    const long long f = t % cnt_f;
    const long long t2 = t / cnt_f;
    const long long a = t2 % cnt_2;
    const long long b = t2 / cnt_2;
    // 0-based face coords (fi, fj, fk) and the face's cell-linear index.
    long long fi, fj, fk;
    if (DIR == 0)      { fi = f;  fj = a;  fk = b; }
    else if (DIR == 1) { fj = f;  fi = a;  fk = b; }
    else               { fk = f;  fi = a;  fj = b; }
    const long long ci_face = fi + (long long)in_ * fj +
                              (long long)in_ * jn_ * fk;
    face_flux<DIR>(svc, nfx, ci_face, coff, chg);
  }
}

template <int DIR>
static int launch_flux(const void* svc, void* nfx, const void* chg_p,
                       int in_, int jn_, int kn_,
                       int i1, int i2, int j1, int j2, int k1, int k2,
                       void* stream) {
  long long cnt_f = (DIR == 0) ? (i2 - i1 + 1) :
                    (DIR == 1) ? (j2 - j1 + 1) : (k2 - k1 + 1);
  long long cnt_2 = (DIR == 0) ? (j2 - j1 + 1) :
                    (DIR == 1) ? (i2 - i1 + 1) : (i2 - i1 + 1);
  long long cnt_3 = (DIR == 0) ? (k2 - k1 + 1) :
                    (DIR == 1) ? (k2 - k1 + 1) : (j2 - j1 + 1);
  long long total = cnt_f * cnt_2 * cnt_3;
  int blocks = (int)((total + 255) / 256);
  cudaStream_t s = (cudaStream_t)stream;
  flux_kernel<DIR><<<blocks, 256, 0, s>>>(
      (const double*)svc, (double*)nfx, (const double*)chg_p,
      in_, jn_, kn_, i1, i2, j1, j2, k1, k2);
  return (int)cudaGetLastError();
}

extern "C" int flux1_cuda(const uintptr_t svc, uintptr_t nfx, const uintptr_t chg_p,
    int in_, int jn_, int kn_, int i1, int i2, int j1, int j2, int k1, int k2,
    uintptr_t stream) {
  return launch_flux<0>(svc, nfx, chg_p, in_, jn_, kn_, i1, i2, j1, j2, k1, k2, stream);
}
extern "C" int flux2_cuda(const uintptr_t svc, uintptr_t nfx, const uintptr_t chg_p,
    int in_, int jn_, int kn_, int i1, int i2, int j1, int j2, int k1, int k2,
    uintptr_t stream) {
  return launch_flux<1>(svc, nfx, chg_p, in_, jn_, kn_, i1, i2, j1, j2, k1, k2, stream);
}
extern "C" int flux3_cuda(const uintptr_t svc, uintptr_t nfx, const uintptr_t chg_p,
    int in_, int jn_, int kn_, int i1, int i2, int j1, int j2, int k1, int k2,
    uintptr_t stream) {
  return launch_flux<2>(svc, nfx, chg_p, in_, jn_, kn_, i1, i2, j1, j2, k1, k2, stream);
}

// ======================================================================
// UpdateConsv: u += dt * (flux divergence in 3 directions [+ gravity
// source on the momenta]). Thread = one cell; the face windows are
// contiguous 10-field rows read as double2, writes are warp-coalesced
// along i. Fortran loops i=is..ie, j=js..je, k=ks..ke.
// ======================================================================
__global__ void __launch_bounds__(256) updateconsv_cuda_kernel(
    const double* __restrict__ nflux1, const double* __restrict__ nflux2,
    const double* __restrict__ nflux3,
    double* __restrict__ d_, double* __restrict__ et_,
    double* __restrict__ mv1_, double* __restrict__ mv2_,
    double* __restrict__ mv3_,
    double* __restrict__ b1_, double* __restrict__ b2_,
    double* __restrict__ b3_, double* __restrict__ bp_,
    double* __restrict__ dxcomp_,
    const double* __restrict__ grv1, const double* __restrict__ grv2,
    const double* __restrict__ grv3,
    const double* __restrict__ x1a, const double* __restrict__ x2a,
    const double* __restrict__ x3a,
    double dt, int in_, int jn_, int kn_,
    int i1, int i2, int j1, int j2, int k1, int k2) {
  long long nxi = i2 - i1 + 1, nxj = j2 - j1 + 1;
  long long total = nxi * nxj * (k2 - k1 + 1);
  for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
       t < total; t += (long long)gridDim.x * blockDim.x) {
    int i = (int)(t % nxi);
    long long tj = t / nxi;
    int j = (int)(tj % nxj);
    int k = (int)(tj / nxj);
    long long ci = (long long)(i1 - 1 + i) +
                   (long long)in_ * (j1 - 1 + j) +
                   (long long)in_ * jn_ * (k1 - 1 + k);
    const double invx = 1.0 / (x1a[i1 - 1 + i + 1] - x1a[i1 - 1 + i]);
    const double invy = 1.0 / (x2a[j1 - 1 + j + 1] - x2a[j1 - 1 + j]);
    const double invz = 1.0 / (x3a[k1 - 1 + k + 1] - x3a[k1 - 1 + k]);

    // Six 10-field windows: faces on both sides in each direction.
    // Staged one direction at a time to keep the register footprint at
    // ~20 live doubles while the per-thread 16-B pieces still cover
    // every sector of each 80-B window across the unrolled loads.
    double acc[10];
    #pragma unroll
    for (int q = 0; q < 10; ++q) acc[q] = 0.0;
    {
      const double2* wa = (const double2*)(nflux1 + ci * 10);
      const double2* wb = (const double2*)(nflux1 + (ci + 1) * 10);
      double a[10], b[10];
      #pragma unroll
      for (int q = 0; q < 5; ++q) {
        double2 w; w = wa[q]; a[2*q] = w.x; a[2*q+1] = w.y;
        w = wb[q]; b[2*q] = w.x; b[2*q+1] = w.y;
      }
      #pragma unroll
      for (int q = 0; q < 10; ++q) acc[q] += (a[q] - b[q]) * invx;
    }
    {
      const double2* wa = (const double2*)(nflux2 + ci * 10);
      const double2* wb = (const double2*)(nflux2 + (ci + (long long)in_) * 10);
      double a[10], b[10];
      #pragma unroll
      for (int q = 0; q < 5; ++q) {
        double2 w; w = wa[q]; a[2*q] = w.x; a[2*q+1] = w.y;
        w = wb[q]; b[2*q] = w.x; b[2*q+1] = w.y;
      }
      #pragma unroll
      for (int q = 0; q < 10; ++q) acc[q] += (a[q] - b[q]) * invy;
    }
    {
      const double2* wa = (const double2*)(nflux3 + ci * 10);
      const double2* wb = (const double2*)(nflux3 + (ci + (long long)in_ * jn_) * 10);
      double a[10], b[10];
      #pragma unroll
      for (int q = 0; q < 5; ++q) {
        double2 w; w = wa[q]; a[2*q] = w.x; a[2*q+1] = w.y;
        w = wb[q]; b[2*q] = w.x; b[2*q+1] = w.y;
      }
      #pragma unroll
      for (int q = 0; q < 10; ++q) acc[q] += (a[q] - b[q]) * invz;
    }

    const double g1v = grv1[ci], g2v = grv2[ci], g3v = grv3[ci];

    // field 0: d
    d_[ci] = d_[ci] + dt * acc[0];
    // fields 1-3: mv1-3 with gravity sources (inside dt, as in Fortran)
    mv1_[ci] = mv1_[ci] + dt * (g1v + acc[1]);
    mv2_[ci] = mv2_[ci] + dt * (g2v + acc[2]);
    mv3_[ci] = mv3_[ci] + dt * (g3v + acc[3]);
    // field 4: et
    et_[ci] = et_[ci] + dt * acc[4];
    // fields 5-8: b1-3, bp
    b1_[ci] = b1_[ci] + dt * acc[5];
    b2_[ci] = b2_[ci] + dt * acc[6];
    b3_[ci] = b3_[ci] + dt * acc[7];
    bp_[ci] = bp_[ci] + dt * acc[8];
    // field 9: tracer (DXcomp, ncomp=1)
    dxcomp_[ci] = dxcomp_[ci] + dt * acc[9];
  }
}

extern "C" int updateconsv_cuda(const uintptr_t nflux1, const uintptr_t nflux2,
    const uintptr_t nflux3, uintptr_t d_, uintptr_t et_, uintptr_t mv1_, uintptr_t mv2_,
    uintptr_t mv3_, uintptr_t b1_, uintptr_t b2_, uintptr_t b3_, uintptr_t bp_, uintptr_t dxcomp_,
    const uintptr_t grv1, const uintptr_t grv2, const uintptr_t grv3,
    const uintptr_t x1a, const uintptr_t x2a, const uintptr_t x3a,
    double dt, int in_, int jn_, int kn_,
    int i1, int i2, int j1, int j2, int k1, int k2, uintptr_t stream) {
  long long total = (long long)(i2 - i1 + 1) * (j2 - j1 + 1) * (k2 - k1 + 1);
  int blocks = (int)((total + 255) / 256);
  cudaStream_t s = (cudaStream_t)stream;
  updateconsv_cuda_kernel<<<blocks, 256, 0, s>>>(
      (const double*)nflux1, (const double*)nflux2, (const double*)nflux3,
      (double*)d_, (double*)et_, (double*)mv1_, (double*)mv2_, (double*)mv3_,
      (double*)b1_, (double*)b2_, (double*)b3_, (double*)bp_,
      (double*)dxcomp_,
      (const double*)grv1, (const double*)grv2, (const double*)grv3,
      (const double*)x1a, (const double*)x2a, (const double*)x3a,
      dt, in_, jn_, kn_, i1, i2, j1, j2, k1, k2);
  return (int)cudaGetLastError();
}
