from services.features.function_postex.detector.postex_detector import PostExDetector

def test_postex_detector_priv():
    d = PostExDetector()
    fs = d.analyze("privilege_escalation","localhost","t1","s1",True,None)
    assert any(f.vulnerability.name.value == "Privilege Escalation" for f in fs)

def test_postex_detector_persist():
    d = PostExDetector()
    fs = d.analyze("persistence","localhost","t1","s1",True,None)
    assert len(fs) >= 1
