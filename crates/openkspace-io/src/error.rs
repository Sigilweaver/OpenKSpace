use thiserror::Error;

#[derive(Debug, Error)]
pub enum IoError {
    #[error("HDF5 error: {0}")]
    Hdf5(#[from] hdf5_metno::Error),

    #[error("XML header parse error: {0}")]
    Xml(String),

    #[error("Unsupported format or structure: {0}")]
    Unsupported(String),

    #[error("Missing required field in header: {0}")]
    MissingField(&'static str),

    #[error("Inconsistent data: {0}")]
    Inconsistent(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

pub type IoResult<T> = Result<T, IoError>;
