import json, os, time
import requests

API = "https://sd629vuj4f7uh2cscrbe0.apigateway-cn-beijing.volceapi.com/api/"
SECRET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".secrets", "xpuoj.json")
CONTEST_ID = 13
PROBLEM_ORDER = 1
LANG = "triton-dist"

class Client:
    def __init__(self):
        self.s = requests.Session()
        creds = json.load(open(SECRET_PATH))
        r = self.s.post(API + "auth/login", json=creds, timeout=30)
        r.raise_for_status()
        data = r.json()
        if "token" not in data:
            raise RuntimeError(f"login failed: {data}")
        self.token = data["token"]
        self.username = data.get("username")
        self.s.headers.update({"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"})

    def _post(self, path, data=None):
        r = self.s.post(API + path, json=data or {}, timeout=60)
        return r

    def submit(self, code, language=LANG, contest_id=CONTEST_ID, problem_order=PROBLEM_ORDER,
               compile_and_run_options=None):
        content = {
            "language": language,
            "code": code,
            "compileAndRunOptions": compile_and_run_options or {},
        }
        r = self._post("contest/play/submit", {
            "contestId": contest_id,
            "problemOrder": problem_order,
            "content": content,
        })
        if r.status_code not in (200, 201):
            raise RuntimeError(f"submit HTTP {r.status_code}: {r.text}")
        data = r.json()
        if "submissionId" not in data:
            raise RuntimeError(f"submit rejected: {data}")
        return data["submissionId"]

    def list_submissions(self, take=10):
        r = self._post("contest/play/querySubmissions", {
            "contestId": CONTEST_ID, "problemOrder": PROBLEM_ORDER, "takeCount": take, "locale": "zh_CN"})
        r.raise_for_status()
        return r.json().get("submissions", [])

    def get_detail(self, submission_id):
        r = self._post("submission/getSubmissionDetail", {"submissionId": str(submission_id), "locale": "zh_CN"})
        return r

    def cancel(self, submission_id):
        # submissionId must be an integer here (unlike get_detail's string).
        # Canceling a stuck/TLE-ing submission frees the serial judge queue.
        return self._post("submission/cancelSubmission", {"submissionId": int(submission_id)})

    def poll(self, submission_id, interval=10, timeout=1800):
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            subs = self.list_submissions(5)
            for s in subs:
                if s["id"] == submission_id:
                    last = s
                    break
            if last and last.get("status") not in ("Pending", "Running"):
                return last
            time.sleep(interval)
        return last

if __name__ == "__main__":
    c = Client()
    subs = c.list_submissions(5)
    for s in subs:
        print(json.dumps({k: s[k] for k in ("id","status","score","displayScore","timeUsed","memoryUsed","submitTime")}, ensure_ascii=False))
