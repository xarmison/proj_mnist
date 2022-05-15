from fastapi.testclient import TestClient
from source.main import app
from numpy import argmax

# Instantiate the testing client with our app.
client = TestClient(app)

def test_root():
    r = client.get('/')
    assert r.status_code == 200

def test_inference():

    drawing_data = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAYAAAByDd+UAAABsklEQVRIS+2VPauBYRjH/w95yUsGmViVMpjsSoriA1DPRmIz2WxMRmI1kEU95RN4yQfwBSwMyipPeT3X3Zl0zvFc98FyzlX3dt39+v+vNwXA9eO9LZR/4LO9/qOW+nw+9Pt9OJ1ObDYblqu73Q7lcvnbP19a6vF4UKlUEI/HcTqdDAFNJhMCgQAulwuCwSAPqCgK3G43HA6HIRglkRuapoHA4XCYBzRM+Uy02WzCkWq1ilKphOFw+DoguREKhTCdTjGbzZDL5aDr+uuAZPtgMEA0GkUqlcJyufzRoF/NodlsRiwWw2g0QrvdRq1Ww/l8fh3Q5XJhMpmI5kokEliv1w/LL63QYrEgm82i1WqJZul2u7heHx8eaSCNzXw+B4Gpfvv9/qE6SpAC3qvrdDqGYNJAWXVSQKvVikKhgEajIXYm7VxOsC31+/1iyFerFTKZDA6HA4fHq6Hdbkez2YSqqkin02KzcIOlkK4BbZLxeIx8Po/j8cjlGVdItSsWi6jX60gmk1gsFmwYq2lom/R6PXF6IpGIlDoWkK6C1+sF7c/tdiuljgWUJtx9ZDXNM6BvB94AnrKdAYLjbpMAAAAASUVORK5CYII='

    r = client.post(f'/predict/?drawing_data={drawing_data}')

    assert r.status_code == 200
    assert r.json()['result'] == 1
    assert r.json()['error'] == ''
    assert len(r.json()['data']) == 10
    assert argmax(r.json()['data']) == 7
