import requests
from urllib.parse import urlparse
class ApiClient:
    """HTTP client used by the Laptop 2 UI. It only talks to the Laptop 1 API;
    it never loads a model or trip data itself."""
    def __init__(self,base_url,timeout=10,predict_timeout=180):
        self.url=base_url.strip().rstrip("/")
        parsed=urlparse(self.url)
        if parsed.scheme not in ("http","https") or not parsed.hostname: raise ValueError("Enter a full HTTP API address, e.g. http://192.168.1.10:8000")
        if parsed.hostname=="0.0.0.0": raise ValueError("0.0.0.0 is a server bind address; enter Laptop 1's IPv4 address instead")
        self.timeout=timeout; self.predict_timeout=predict_timeout
    @staticmethod
    def _raise(response):
        if response.ok: return
        try: detail=response.json().get("detail") or response.json().get("error")
        except ValueError: detail=response.text[:200]
        raise requests.HTTPError(f"HTTP {response.status_code}: {detail}",response=response)
    def health(self):
        # A degraded backend answers 503 with a JSON explanation; return it so the
        # UI can distinguish "reachable but model not loaded" from "unreachable".
        response=requests.get(self.url+"/health",timeout=self.timeout)
        if response.status_code==503:
            try: return response.json()
            except ValueError: pass
        self._raise(response); return response.json()
    def trips(self):
        response=requests.get(self.url+"/trips",timeout=self.timeout); self._raise(response); return response.json()
    def predict(self,trip_id):
        response=requests.post(self.url+"/predict",json={"trip_id":trip_id},timeout=self.predict_timeout)
        self._raise(response); return response.json()
