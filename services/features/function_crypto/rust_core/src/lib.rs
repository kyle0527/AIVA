use pyo3::prelude::*;

#[pyfunction]
fn scan_crypto_weaknesses(code: &str) -> PyResult<Vec<(String, String)>> {
    let mut findings: Vec<(String,String)> = Vec::new();
    let lower = code.to_lowercase();
    if lower.contains("md5") { findings.push(("WEAK_ALGORITHM".into(),"Detected usage of MD5 algorithm".into())); }
    if lower.contains("sha1") { findings.push(("WEAK_ALGORITHM".into(),"Detected usage of SHA1 algorithm".into())); }
    if lower.contains(" des ") || lower.contains("des(") { findings.push(("WEAK_CIPHER".into(),"Detected usage of DES cipher".into())); }
    if lower.contains("rc4") { findings.push(("WEAK_CIPHER".into(),"Detected usage of RC4 cipher".into())); }
    if lower.contains("verify=false") { findings.push(("INSECURE_TLS".into(),"TLS certificate verification disabled".into())); }
    if lower.contains("tls1.0") || lower.contains("tlsv1") { findings.push(("INSECURE_TLS".into(),"Insecure TLS version in use".into())); }
    if lower.contains("private key") || (lower.contains("-----begin") && lower.contains("private key-----")) { findings.push(("HARDCODED_KEY".into(),"Private key material embedded".into())); }
    if lower.contains("random(") || lower.contains("srand(") || lower.contains("rand(") { findings.push(("WEAK_RANDOM".into(),"Potentially insecure RNG".into())); }
    Ok(findings)
}

#[pymodule]
fn crypto_engine(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(scan_crypto_weaknesses, m)?)?;
    Ok(())
}
