from services.features.function_crypto.detector.crypto_detector import CryptoDetector

def test_crypto_detector():
    d = CryptoDetector()
    code = "hash = MD5('x'); verify=False; -----BEGIN PRIVATE KEY-----; srand(1);"
    fs = d.detect(code, task_id="t1", scan_id="s1")
    assert len(fs) >= 3
