import requests

BASE_URL = "http://127.0.0.1:8000"


def main() -> None:
    r = requests.get(f"{BASE_URL}/", timeout=30)
    print("GET / status:", r.status_code)
    print("GET / json:", r.json())

    data = {
        "age": 37,
        "workclass": "Private",
        "fnlgt": 178356,
        "education": "HS-grad",
        "education-num": 10,
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }

    r = requests.post(f"{BASE_URL}/data/", json=data, timeout=30)
    print("POST /data/ status:", r.status_code)
    print("POST /data/ json:", r.json())


if __name__ == "__main__":
    main()
