from pocket_capper.engine import remote


def test_dispatch_payload(monkeypatch):
    sent = {}

    class R:
        status_code = 204
        text = ""

    def fake_post(url, headers, json, timeout):
        sent.update(url=url, auth=headers["Authorization"], body=json)
        return R()

    monkeypatch.setenv("GH_DISPATCH_TOKEN", "tok")
    monkeypatch.setattr(remote.requests, "post", fake_post)
    assert remote.enabled()
    remote.dispatch(["nfl"], False)
    assert sent["url"].endswith("/repos/cwem-cloud/nfl-betting-model/actions/workflows/pocket_capper.yml/dispatches")
    assert sent["auth"] == "Bearer tok"
    assert sent["body"] == {"ref": "main", "inputs": {"task": "slate", "sports": "nfl", "props": "false"}}
