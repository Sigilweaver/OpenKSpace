//! Minimal ISMRMRD XML header parser.
//!
//! We only extract the fields required for basic cartesian recon. The full
//! schema is large; we deliberately do a narrow extraction here to avoid
//! pulling in a heavy XSD-driven parser.

use crate::error::{IoError, IoResult};
use quick_xml::events::Event;
use quick_xml::Reader;

#[derive(Debug, Clone, Copy, Default)]
pub struct MatrixSize {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct FieldOfView {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct EncodingLimit {
    pub minimum: u32,
    pub maximum: u32,
    pub center: u32,
}

#[derive(Debug, Clone, Default)]
pub struct EncodingInfo {
    pub encoded_matrix: MatrixSize,
    pub recon_matrix: MatrixSize,
    pub encoded_fov: FieldOfView,
    pub recon_fov: FieldOfView,
    pub trajectory: String,      // "cartesian", "radial", ...
    pub ky_limit: EncodingLimit, // kspace_encoding_step_1
    pub kz_limit: EncodingLimit, // kspace_encoding_step_2
    pub slice_limit: EncodingLimit,
}

#[derive(Debug, Clone, Default)]
pub struct IsmrmrdHeader {
    pub system_vendor: String,
    pub system_model: String,
    pub field_strength_t: f32,
    pub receiver_channels: u32,
    pub encoding: EncodingInfo,
}

impl IsmrmrdHeader {
    /// Parse an ISMRMRD XML header string.
    ///
    /// The parser walks tokens and captures only the fields needed for
    /// cartesian reconstruction.
    pub fn parse(xml: &str) -> IoResult<Self> {
        let mut reader = Reader::from_str(xml);
        reader.config_mut().trim_text(true);

        let mut hdr = IsmrmrdHeader::default();

        // Breadcrumb stack of open element local-names.
        let mut path: Vec<String> = Vec::with_capacity(16);
        let mut buf = Vec::new();

        loop {
            match reader.read_event_into(&mut buf) {
                Err(e) => {
                    return Err(IoError::Xml(format!(
                        "xml at {}: {e}",
                        reader.buffer_position()
                    )))
                }
                Ok(Event::Eof) => break,
                Ok(Event::Start(e)) => {
                    let name = local_name(e.name().as_ref());
                    path.push(name);
                }
                Ok(Event::End(_)) => {
                    path.pop();
                }
                Ok(Event::Empty(_)) => {
                    // self-closing tag: no text to capture
                }
                Ok(Event::Text(t)) => {
                    let text = t.unescape().map_err(|e| IoError::Xml(e.to_string()))?;
                    let text = text.trim();
                    if text.is_empty() {
                        continue;
                    }
                    apply_field(&path, text, &mut hdr);
                }
                _ => {}
            }
            buf.clear();
        }

        Ok(hdr)
    }

    /// Matrix size used for the final image (recon space).
    #[inline]
    pub fn recon_size(&self) -> MatrixSize {
        self.encoding.recon_matrix
    }

    /// Matrix size of the acquired k-space (encoded space).
    #[inline]
    pub fn encoded_size(&self) -> MatrixSize {
        self.encoding.encoded_matrix
    }
}

fn local_name(bytes: &[u8]) -> String {
    // quick-xml gives us `prefix:local` or just `local` -- we only care about local.
    let s = std::str::from_utf8(bytes).unwrap_or("");
    match s.rfind(':') {
        Some(i) => s[i + 1..].to_string(),
        None => s.to_string(),
    }
}

/// Dispatches a text value into the struct based on the path of open tags.
fn apply_field(path: &[String], text: &str, hdr: &mut IsmrmrdHeader) {
    // Walk the tail of the path to match. Use endswith-style matching so we
    // ignore the top-level <ismrmrdHeader> wrapper.
    let tail: Vec<&str> = path.iter().map(String::as_str).collect();

    // ---- acquisitionSystemInformation --------------------------------------
    if ends_with(&tail, &["acquisitionSystemInformation", "systemVendor"]) {
        hdr.system_vendor = text.to_string();
    } else if ends_with(&tail, &["acquisitionSystemInformation", "systemModel"]) {
        hdr.system_model = text.to_string();
    } else if ends_with(
        &tail,
        &["acquisitionSystemInformation", "systemFieldStrength_T"],
    ) {
        hdr.field_strength_t = text.parse().unwrap_or(0.0);
    } else if ends_with(&tail, &["acquisitionSystemInformation", "receiverChannels"]) {
        hdr.receiver_channels = text.parse().unwrap_or(0);
    }
    // ---- encoding/encodedSpace/matrixSize ----------------------------------
    else if ends_with(&tail, &["encoding", "encodedSpace", "matrixSize", "x"]) {
        hdr.encoding.encoded_matrix.x = text.parse().unwrap_or(0);
    } else if ends_with(&tail, &["encoding", "encodedSpace", "matrixSize", "y"]) {
        hdr.encoding.encoded_matrix.y = text.parse().unwrap_or(0);
    } else if ends_with(&tail, &["encoding", "encodedSpace", "matrixSize", "z"]) {
        hdr.encoding.encoded_matrix.z = text.parse().unwrap_or(0);
    }
    // ---- encoding/encodedSpace/fieldOfView_mm ------------------------------
    else if ends_with(&tail, &["encoding", "encodedSpace", "fieldOfView_mm", "x"]) {
        hdr.encoding.encoded_fov.x = text.parse().unwrap_or(0.0);
    } else if ends_with(&tail, &["encoding", "encodedSpace", "fieldOfView_mm", "y"]) {
        hdr.encoding.encoded_fov.y = text.parse().unwrap_or(0.0);
    } else if ends_with(&tail, &["encoding", "encodedSpace", "fieldOfView_mm", "z"]) {
        hdr.encoding.encoded_fov.z = text.parse().unwrap_or(0.0);
    }
    // ---- encoding/reconSpace/matrixSize ------------------------------------
    else if ends_with(&tail, &["encoding", "reconSpace", "matrixSize", "x"]) {
        hdr.encoding.recon_matrix.x = text.parse().unwrap_or(0);
    } else if ends_with(&tail, &["encoding", "reconSpace", "matrixSize", "y"]) {
        hdr.encoding.recon_matrix.y = text.parse().unwrap_or(0);
    } else if ends_with(&tail, &["encoding", "reconSpace", "matrixSize", "z"]) {
        hdr.encoding.recon_matrix.z = text.parse().unwrap_or(0);
    }
    // ---- encoding/reconSpace/fieldOfView_mm --------------------------------
    else if ends_with(&tail, &["encoding", "reconSpace", "fieldOfView_mm", "x"]) {
        hdr.encoding.recon_fov.x = text.parse().unwrap_or(0.0);
    } else if ends_with(&tail, &["encoding", "reconSpace", "fieldOfView_mm", "y"]) {
        hdr.encoding.recon_fov.y = text.parse().unwrap_or(0.0);
    } else if ends_with(&tail, &["encoding", "reconSpace", "fieldOfView_mm", "z"]) {
        hdr.encoding.recon_fov.z = text.parse().unwrap_or(0.0);
    }
    // ---- encoding/trajectory -----------------------------------------------
    else if ends_with(&tail, &["encoding", "trajectory"]) {
        hdr.encoding.trajectory = text.to_string();
    }
    // ---- encoding/encodingLimits -------------------------------------------
    else if path.len() >= 4
        && path[path.len() - 4] == "encoding"
        && path[path.len() - 3] == "encodingLimits"
    {
        let section = &path[path.len() - 2];
        let which = &path[path.len() - 1];
        let target: Option<&mut EncodingLimit> = match section.as_str() {
            "kspace_encoding_step_1" => Some(&mut hdr.encoding.ky_limit),
            "kspace_encoding_step_2" => Some(&mut hdr.encoding.kz_limit),
            "slice" => Some(&mut hdr.encoding.slice_limit),
            _ => None,
        };
        if let Some(t) = target {
            match which.as_str() {
                "minimum" => t.minimum = text.parse().unwrap_or(0),
                "maximum" => t.maximum = text.parse().unwrap_or(0),
                "center" => t.center = text.parse().unwrap_or(0),
                _ => {}
            }
        }
    }
}

#[inline]
fn ends_with(path: &[&str], suffix: &[&str]) -> bool {
    if path.len() < suffix.len() {
        return false;
    }
    path[path.len() - suffix.len()..] == *suffix
}

// ----------------------------------------------------------------------------
//                                   Tests
// ----------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"<?xml version="1.0"?>
    <ismrmrdHeader xmlns="http://www.ismrm.org/ISMRMRD">
      <acquisitionSystemInformation>
        <systemVendor>GE MEDICAL SYSTEMS</systemVendor>
        <systemModel>Orchestra SDK</systemModel>
        <systemFieldStrength_T>3.000000</systemFieldStrength_T>
        <receiverChannels>8</receiverChannels>
      </acquisitionSystemInformation>
      <encoding>
        <encodedSpace>
          <matrixSize><x>352</x><y>216</y><z>30</z></matrixSize>
          <fieldOfView_mm><x>280</x><y>280</y><z>4.5</z></fieldOfView_mm>
        </encodedSpace>
        <reconSpace>
          <matrixSize><x>512</x><y>512</y><z>0</z></matrixSize>
          <fieldOfView_mm><x>280</x><y>280</y><z>4.5</z></fieldOfView_mm>
        </reconSpace>
        <encodingLimits>
          <kspace_encoding_step_1><minimum>0</minimum><maximum>215</maximum><center>108</center></kspace_encoding_step_1>
          <kspace_encoding_step_2><minimum>0</minimum><maximum>0</maximum><center>0</center></kspace_encoding_step_2>
          <slice><minimum>0</minimum><maximum>29</maximum><center>15</center></slice>
        </encodingLimits>
        <trajectory>cartesian</trajectory>
      </encoding>
    </ismrmrdHeader>"#;

    #[test]
    fn parses_header_fields() {
        let h = IsmrmrdHeader::parse(SAMPLE).unwrap();
        assert_eq!(h.system_vendor, "GE MEDICAL SYSTEMS");
        assert_eq!(h.receiver_channels, 8);
        assert!((h.field_strength_t - 3.0).abs() < 1e-6);

        assert_eq!(h.encoding.encoded_matrix.x, 352);
        assert_eq!(h.encoding.encoded_matrix.y, 216);
        assert_eq!(h.encoding.encoded_matrix.z, 30);
        assert_eq!(h.encoding.recon_matrix.x, 512);
        assert_eq!(h.encoding.recon_matrix.y, 512);

        assert_eq!(h.encoding.ky_limit.maximum, 215);
        assert_eq!(h.encoding.slice_limit.maximum, 29);
        assert_eq!(h.encoding.trajectory, "cartesian");
    }
}
