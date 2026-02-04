use thiserror::Error;

#[derive(Error, Debug)]
pub enum FerrumError {
    #[error("IO Error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Parsing Error: {0}")]
    ParseError(String),

    #[error("Chemistry Error: {0}")]
    ChemError(String),

    #[error("Feature not implemented: {0}")]
    NotImplemented(String),
    
    #[error("JSON Error: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, FerrumError>;
