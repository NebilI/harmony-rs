use thiserror::Error;

#[derive(Debug, Error)]
pub enum HarmonyError {
    #[error("{0}")]
    Message(String),
}

impl HarmonyError {
    pub fn msg(msg: impl Into<String>) -> Self {
        Self::Message(msg.into())
    }
}

pub type Result<T> = std::result::Result<T, HarmonyError>;
