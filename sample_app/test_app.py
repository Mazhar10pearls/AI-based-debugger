from sample_app.app import app


def test_homepage():
    client = app.test_client()
    response = client.get("/")

    assert response.status_code == 200
    assert b"Hello from local AI DevOps app!" in response.data
