//! R's default Mersenne-Twister (`set.seed` + `runif`).
//!
//! Harmony draws `arma::fill::randu` through R's `unif_rand`. Matching that
//! stream is what lets k-means++ land on the same cells as CRAN Harmony when
//! both sides are given the same integer seed.

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER: u32 = 0x8000_0000;
const LOWER: u32 = 0x7fff_ffff;
const I2_32: f64 = 1.0 / 4_294_967_296.0;

/// R `Mersenne-Twister` after `set.seed(seed)`.
#[derive(Clone, Debug)]
pub struct RRng {
    mt: [u32; N],
    mti: usize,
}

impl RRng {
    /// Same state as R's `set.seed(seed)` with the default kind.
    pub fn set_seed(seed: u32) -> Self {
        let mut s = seed;
        for _ in 0..50 {
            s = 69069u32.wrapping_mul(s).wrapping_add(1);
        }
        let mut mt = [0u32; N];
        for slot in &mut mt {
            s = 69069u32.wrapping_mul(s).wrapping_add(1);
            *slot = s;
        }
        // First tempered word after the initial twist is discarded; the rest
        // match `runif`.
        let mut rng = Self { mt, mti: N };
        let _ = rng.next_u32_raw();
        rng
    }

    pub fn from_harmony_seed(seed: u64) -> Self {
        Self::set_seed(seed as u32)
    }

    fn twist(&mut self) {
        for i in 0..N {
            let x = (self.mt[i] & UPPER) | (self.mt[(i + 1) % N] & LOWER);
            let mut xa = x >> 1;
            if x & 1 != 0 {
                xa ^= MATRIX_A;
            }
            self.mt[i] = self.mt[(i + M) % N] ^ xa;
        }
        self.mti = 0;
    }

    fn next_u32_raw(&mut self) -> u32 {
        if self.mti >= N {
            self.twist();
        }
        let mut y = self.mt[self.mti];
        self.mti += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// R `runif(1)` in `[0, 1)`.
    pub fn unif(&mut self) -> f64 {
        self.next_u32_raw() as f64 * I2_32
    }

    /// `(float) unif_rand()` as Armadillo `randu` does on a Harmony `float` build.
    #[allow(dead_code)]
    pub fn unif_f32(&mut self) -> f32 {
        self.unif() as f32
    }

    /// `floor(unif * n)` index in `0..n`, matching Armadillo's usual integer draw.
    pub fn index(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        let j = (self.unif() * n as f64).floor() as usize;
        j.min(n - 1)
    }

    /// Armadillo / RcppArmadillo `shuffle`: `j = i + floor(U*(n-i))`.
    pub fn shuffle<T>(&mut self, xs: &mut [T]) {
        let n = xs.len();
        for i in 0..n {
            let span = n - i;
            let j = i + self.index(span);
            xs.swap(i, j);
        }
    }
}

/// Cell-order shuffle used by `update_r`.
pub trait ShuffleRng {
    fn shuffle_cells(&mut self, order: &mut [usize]);
}

impl ShuffleRng for RRng {
    fn shuffle_cells(&mut self, order: &mut [usize]) {
        self.shuffle(order);
    }
}

impl ShuffleRng for rand_chacha::ChaCha8Rng {
    fn shuffle_cells(&mut self, order: &mut [usize]) {
        use rand::seq::SliceRandom;
        order.shuffle(self);
    }
}

/// Uniform-(0, 1) source used by Harmony-style k-means++.
pub trait UnitUniform {
    fn unit(&mut self) -> f64;
}

impl UnitUniform for RRng {
    fn unit(&mut self) -> f64 {
        self.unif()
    }
}

impl UnitUniform for rand_chacha::ChaCha8Rng {
    fn unit(&mut self) -> f64 {
        use rand::Rng;
        self.gen::<f64>().clamp(1e-12, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_r_runif_after_set_seed_1() {
        let want = [
            0.2655086631421,
            0.37212389963679016,
            0.5728533633518964,
            0.9082077899947762,
            0.2016819310374558,
        ];
        let mut rng = RRng::set_seed(1);
        for w in want {
            let g = rng.unif();
            assert!((g - w).abs() < 1e-15, "runif {g} vs {w}");
        }
    }

    #[test]
    fn matches_r_runif_after_set_seed_42() {
        let want = [
            0.9148060434963554,
            0.9370754132978618,
            0.2861395347863436,
        ];
        let mut rng = RRng::set_seed(42);
        for w in want {
            let g = rng.unif();
            assert!((g - w).abs() < 1e-15, "runif {g} vs {w}");
        }
    }
}
