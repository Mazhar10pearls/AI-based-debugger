from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello from local AI DevOps app!"

if __name__ == "__main__":
    print("Running app...")
