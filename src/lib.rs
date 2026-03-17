//! Library surface for GPU-accelerated bird's-eye-view processing.
//!
//! This crate exposes reusable compute components (`backend`, `imgproc`, `math`) and keeps
//! application-specific transport logic in binaries.

pub mod backend;
pub mod imgproc;
pub mod math;
mod nodes;
