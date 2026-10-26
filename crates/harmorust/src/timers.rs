//! Optional phase timers for the Harmony loop.
//!
//! Enabled when `HARMORUST_TIMERS=1`. Times are wall-clock milliseconds and
//! are printed after a run; they never change numerical results.

use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct PhaseTimers {
    pub kmeans_ms: f64,
    pub softmax_ms: f64,
    pub update_r_ms: f64,
    pub objective_ms: f64,
    pub ridge_ms: f64,
}

impl PhaseTimers {
    pub fn enabled() -> bool {
        matches!(
            std::env::var("HARMORUST_TIMERS").as_deref(),
            Ok("1") | Ok("true") | Ok("yes")
        )
    }

    pub fn add(&mut self, field: &str, ms: f64) {
        match field {
            "kmeans" => self.kmeans_ms += ms,
            "softmax" => self.softmax_ms += ms,
            "update_r" => self.update_r_ms += ms,
            "objective" => self.objective_ms += ms,
            "ridge" => self.ridge_ms += ms,
            _ => {}
        }
    }

    pub fn total_ms(&self) -> f64 {
        self.kmeans_ms + self.softmax_ms + self.update_r_ms + self.objective_ms + self.ridge_ms
    }
}

pub struct ScopeTimer {
    start: Instant,
}

impl ScopeTimer {
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    pub fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }
}
